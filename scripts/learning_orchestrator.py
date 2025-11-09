#!/usr/bin/env python3
"""
Incremental Learning Orchestrator for Allie

Manages the complete learning pipeline from data collection to model deployment.
"""

import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import threading
import time
import psutil
from dataclasses import dataclass, asdict
import os
import shutil

from experience_replay import ExperienceReplayBuffer
from ewc_regularization import ElasticWeightConsolidation, EWCOptimizer
from safety_quality_filters import LearningDataValidator
from evaluation_framework import LearningEvaluator, ModelEvaluator

logger = logging.getLogger(__name__)

@dataclass
class LearningEpisode:
    """Represents a single learning episode"""
    id: str
    status: str  # 'scheduled', 'running', 'completed', 'failed', 'cancelled'
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    data_stats: Dict[str, Any] = None
    training_stats: Dict[str, Any] = None
    evaluation_results: Dict[str, Any] = None
    error_message: Optional[str] = None

    def to_dict(self):
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data

class IncrementalLearningOrchestrator:
    """Main orchestrator for incremental learning episodes"""

    def __init__(self, config_path: str = "config/learning_config.json"):
        self.config = self._load_config(config_path)
        self.setup_directories()

        # Initialize components
        self.data_validator = LearningDataValidator()
        self.replay_buffer = ExperienceReplayBuffer(
            self.config['replay_buffer_path'],
            max_size=self.config['replay_buffer_size']
        )

        # Initialize evaluators
        self.model_evaluator = ModelEvaluator(
            base_model_path=self.config['base_model_path']
        )
        self.learning_evaluator = LearningEvaluator(self.model_evaluator)

        # Learning state
        self.current_episode: Optional[LearningEpisode] = None
        self.learning_history: List[LearningEpisode] = []
        self.is_learning_active = False

        # Load existing state
        self.load_state()

        logger.info("Incremental Learning Orchestrator initialized")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load learning configuration"""
        default_config = {
            'base_model_path': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
            'replay_buffer_path': 'data/replay_buffer.json',
            'replay_buffer_size': 5000,
            'data_dir': '../data',  # Relative to scripts/ directory
            'models_dir': 'models',
            'logs_dir': 'logs',
            'learning_thresholds': {
                'min_conversations': 3,  # Minimum approved conversations to trigger learning
                'min_quality_score': 0.7,
                'max_age_hours': 24
            },
            'training_params': {
                'batch_size': 4,
                'learning_rate': 2e-4,
                'num_epochs': 3,
                'ewc_lambda': 0.1,
                'replay_ratio': 0.3,
                'max_length': 512
            },
            'scheduling': {
                'preferred_start_hour': 0,  # Always allow learning for testing
                'preferred_end_hour': 23,  # Allow learning until 11 PM
                'max_concurrent_episodes': 1
            }
        }

        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)

        return default_config

    def setup_directories(self):
        """Create necessary directories"""
        for dir_path in [self.config['data_dir'], self.config['models_dir'], self.config['logs_dir']]:
            Path(dir_path).mkdir(exist_ok=True)

    def should_trigger_learning(self) -> Tuple[bool, str]:
        """
        Determine if learning should be triggered based on data availability and system state

        Returns:
            (should_trigger, reason)
        """
        # Check if already learning
        if self.is_learning_active:
            return False, "Learning episode already in progress"

        # Check data availability
        data_info = self._collect_available_data()
        conversation_data = data_info['conversations']
        if not conversation_data:
            return False, "No conversation data available"

        # Validate data quality
        validated_data = self.data_validator.validate_dataset(conversation_data)
        approved_count = validated_data['approved_count']
        total_count = validated_data['total_conversations']

        if approved_count < self.config['learning_thresholds']['min_conversations']:
            return False, f"Insufficient quality data: {approved_count}/{total_count} approved"

        # Check time constraints
        current_hour = datetime.now().hour
        preferred_start = self.config['scheduling']['preferred_start_hour']
        preferred_end = self.config['scheduling']['preferred_end_hour']

        if not (preferred_start <= current_hour < preferred_end):
            return False, f"Not in preferred learning window ({preferred_start}-{preferred_end})"

        # Check system resources
        if not self._check_system_resources():
            return False, "Insufficient system resources"

        return True, f"Ready to learn: {approved_count} quality conversations available"

    def start_learning_episode(self) -> str:
        """Start a new learning episode"""
        if self.is_learning_active:
            raise RuntimeError("Learning episode already in progress")

        episode_id = f"learn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.current_episode = LearningEpisode(
            id=episode_id,
            status='running',
            start_time=datetime.now()
        )

        self.is_learning_active = True

        # Run learning in background thread
        learning_thread = threading.Thread(
            target=self._run_learning_episode,
            name=f"learning-{episode_id}"
        )
        learning_thread.daemon = True
        learning_thread.start()

        logger.info(f"Started learning episode {episode_id}")
        return episode_id

    def _run_learning_episode(self):
        """Execute the learning episode (runs in background thread)"""
        try:
            # Phase 1: Data preparation
            self.current_episode.status = 'preparing_data'
            data_info = self._collect_available_data()
            conversation_data = data_info['conversations']
            validated_data = self.data_validator.validate_dataset(conversation_data)

            approved_conversations = [
                result['conversation'] for result in validated_data['results']
                if result['approved']
            ]

            self.current_episode.data_stats = {
                'total_conversations': len(conversation_data),
                'approved_conversations': len(approved_conversations),
                'avg_quality': validated_data['quality_stats']['mean']
            }

            # Phase 2: Prepare training data
            self.current_episode.status = 'preparing_training'
            training_data = self._prepare_training_data(approved_conversations)

            # Phase 3: Execute training
            self.current_episode.status = 'training'
            training_results = self._execute_training(training_data)

            # Phase 4: Evaluation
            self.current_episode.status = 'evaluating'
            evaluation_results = self._evaluate_learning(training_results)

            # Phase 5: Deployment
            self.current_episode.status = 'deploying'
            deployment_success = self._deploy_model(training_results['model_path'])

            # Phase 6: Cleanup
            self.current_episode.status = 'cleaning_up'
            self.cleanup_folder(self.config['models_dir'])
            self.cleanup_folder(str(Path(self.config['data_dir']) / 'backups'))

            # Success
            self.current_episode.status = 'completed'
            self.current_episode.end_time = datetime.now()
            self.current_episode.training_stats = training_results
            self.current_episode.evaluation_results = evaluation_results

            logger.info(f"Learning episode {self.current_episode.id} completed successfully")

        except Exception as e:
            logger.error(f"Learning episode {self.current_episode.id} failed: {e}")
            self.current_episode.status = 'failed'
            self.current_episode.error_message = str(e)
            self.current_episode.end_time = datetime.now()

        finally:
            self.is_learning_active = False
            self.learning_history.append(self.current_episode)
            self.save_state()

    def _collect_available_data(self) -> Dict[str, Any]:
        """Collect available conversation data for learning"""
        data_dir = Path(self.config['data_dir'])
        conversations = []

        # Load from backup.json (simple format)
        backup_file = data_dir / "backup.json"
        if backup_file.exists():
            try:
                with open(backup_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if self._normalize_conversation(item):
                                conversations.append(self._normalize_conversation(item))
                        logger.info(f"Loaded {len([c for c in data if self._normalize_conversation(c)])} conversations from backup.json")
            except Exception as e:
                logger.warning(f"Failed to load backup data: {e}")

        # Load from conversations.json (complex format)
        conv_file = data_dir / "conversations.json"
        if conv_file.exists():
            try:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        normalized_count = 0
                        for item in data:
                            normalized = self._normalize_conversation(item)
                            if normalized:
                                conversations.append(normalized)
                                normalized_count += 1
                        logger.info(f"Loaded {normalized_count} conversations from conversations.json")
            except Exception as e:
                logger.warning(f"Failed to load conversations data: {e}")

        logger.info(f"Total conversations available: {len(conversations)}")
        return {
            'conversations': conversations,
            'total_count': len(conversations),
            'data_sources': ['backup.json'] if backup_file.exists() else []
        }

    def _normalize_conversation(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize conversation to standard format"""
        # Handle simple format: {prompt, response}
        if 'prompt' in item and 'response' in item:
            return {
                'prompt': item['prompt'],
                'response': item['response']
            }

        # Handle simple format: {prompt, reply}
        if 'prompt' in item and 'reply' in item:
            return {
                'prompt': item['prompt'],
                'response': item['reply']
            }

        # Handle complex format with messages array
        if 'messages' in item and isinstance(item['messages'], list):
            messages = item['messages']
            # Extract user message and AI response
            user_msg = None
            ai_msg = None

            for msg in messages:
                if msg.get('role') == 'me' and not user_msg:
                    user_msg = msg.get('text', '')
                elif msg.get('role') == 'them' and not ai_msg:
                    ai_msg = msg.get('text', '')

            if user_msg and ai_msg:
                return {
                    'prompt': user_msg,
                    'response': ai_msg
                }

        # Skip invalid formats
        return None

    def _prepare_training_data(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare training data with replay sampling"""
        # Add new conversations to replay buffer
        for conv in conversations:
            validation_result = self.data_validator.validate_conversation(conv)
            quality_score = validation_result['scores']['quality']['overall'] if validation_result['approved'] else 0.0
            self.replay_buffer.add_conversation(conv, quality_score)

        # Sample replay data
        replay_ratio = self.config['training_params']['replay_ratio']
        replay_size = int(len(conversations) * replay_ratio)
        replay_data = self.replay_buffer.sample_replay_data(replay_size)

        # Combine datasets
        combined_data = conversations + replay_data

        # Convert to training format
        training_examples = []
        for conv in combined_data:
            training_examples.append({
                'prompt': conv.get('prompt', ''),
                'completion': conv.get('response', '')
            })

        return {
            'training_examples': training_examples,
            'new_conversations': len(conversations),
            'replay_conversations': len(replay_data),
            'total_examples': len(training_examples)
        }

    def _execute_training(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the training process with EWC regularization"""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import Dataset

        logger.info("Starting actual training with PEFT and EWC...")

        try:
            # Load base model and tokenizer
            model_name = self.config['base_model_path']
            logger.info(f"Loading model: {model_name}")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            # Load existing LoRA adapter if it exists
            adapter_path = Path(self.config.get('current_adapter_path', 'allie_finetuned'))
            if adapter_path.exists():
                logger.info(f"Loading existing adapter from {adapter_path}")
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(adapter_path))

            # Setup LoRA configuration
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )

            # Prepare model for training
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

            # Prepare dataset
            def tokenize_function(examples):
                prompts = [f"Human: {p}\nAssistant: {c}" for p, c in zip(examples['prompt'], examples['completion'])]
                return tokenizer(prompts, truncation=True, padding=True, max_length=self.config['training_params']['max_length'])

            dataset = Dataset.from_list(training_data['training_examples'])
            tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

            # Initialize EWC regularizer with correct device parameter
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            ewc_regularizer = ElasticWeightConsolidation(model, device=device)
            ewc_regularizer.ewc_lambda = self.config['training_params']['ewc_lambda']

            # Custom trainer with EWC loss
            class EWCTrainer(Trainer):
                def __init__(self, ewc_regularizer, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.ewc_regularizer = ewc_regularizer

                def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                    # Updated signature to match newer transformers API
                    # num_items_in_batch is passed by newer versions but not used here
                    outputs = model(**inputs)
                    loss = outputs.loss

                    # Add EWC regularization
                    ewc_loss = self.ewc_regularizer.ewc_loss(model)
                    total_loss = loss + ewc_loss

                    return (total_loss, outputs) if return_outputs else total_loss

            # Training arguments - use outputs dir to avoid cluttering backend/
            output_dir = Path(__file__).parent.parent / "outputs" / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=self.config['training_params']['num_epochs'],
                per_device_train_batch_size=self.config['training_params']['batch_size'],
                learning_rate=self.config['training_params']['learning_rate'],
                logging_steps=10,
                save_steps=100,
                eval_strategy="no",  # Changed from evaluation_strategy to eval_strategy
                save_strategy="no",
                load_best_model_at_end=False,
                push_to_hub=False,
                report_to="none",
                fp16=True,
                gradient_checkpointing=True,
            )

            # Initialize trainer
            trainer = EWCTrainer(
                ewc_regularizer=ewc_regularizer,
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
            )

            # Train the model
            logger.info("Starting training...")
            trainer.train()

            # Save the LoRA adapter
            output_dir = Path(self.config['models_dir']) / f"allie_v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            output_dir.mkdir(exist_ok=True)
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))

            # Update EWC with new parameters
            ewc_regularizer.update_fisher_information(tokenized_dataset, model, trainer)
            ewc_regularizer.save_consolidation_state(str(output_dir / "ewc_state.pt"))

            training_stats = {
                'epochs_completed': training_args.num_train_epochs,
                'final_loss': trainer.state.log_history[-1].get('train_loss', 0),
                'learning_rate': training_args.learning_rate,
                'samples_processed': len(tokenized_dataset),
                'model_path': str(output_dir),
                'training_time_seconds': time.time() - time.time(),  # Would need to track this properly
                'trainable_parameters': model.get_nb_trainable_parameters(),
                'lora_config': str(lora_config)
            }

            logger.info(f"Training completed successfully. Model saved to {output_dir}")
            return training_stats

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def _evaluate_learning(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the learning results"""
        # Placeholder evaluation
        evaluation_results = {
            'perplexity': 45.2,
            'coherence': 0.78,
            'relevance': 0.72,
            'safety_score': 0.95,
            'forgetting_detected': False,
            'learning_gains': {
                'coherence': 0.05,
                'relevance': 0.03
            }
        }

        return evaluation_results

    def _deploy_model(self, model_path: str) -> bool:
        """Deploy the trained model"""
        # Placeholder deployment
        logger.info(f"Deploying model from {model_path}")
        time.sleep(1)  # Simulate deployment time
        return True

    def _check_system_resources(self) -> bool:
        """Check if system has sufficient resources for learning"""
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 95:  # Increased threshold
                return False

            # Check memory
            memory = psutil.virtual_memory()
            if memory.percent > 95:  # Increased threshold
                return False

            # Check disk space (fix for Windows)
            try:
                disk = psutil.disk_usage('C:/')  # Use C:/ for Windows
                if disk.percent > 95:  # Increased threshold
                    return False
            except:
                # If disk check fails, don't block learning
                pass

            return True

        except Exception as e:
            logger.warning(f"Resource check failed: {e}")
            return True  # Default to allowing learning

    def cleanup_folder(self, folder_path: str, max_files: int = 30):
        """Clean up folder to keep only the most recent max_files files"""
        try:
            path = Path(folder_path)
            if not path.exists() or not path.is_dir():
                return

            # Get all files in the folder (not subdirectories)
            files = [f for f in path.iterdir() if f.is_file()]

            if len(files) <= max_files:
                return

            # Sort by modification time (newest first)
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Keep the most recent max_files, delete the rest
            files_to_delete = files[max_files:]

            for file_path in files_to_delete:
                try:
                    file_path.unlink()
                    logger.info(f"Cleaned up old file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete {file_path}: {e}")

            logger.info(f"Cleaned up {len(files_to_delete)} files from {folder_path}")

        except Exception as e:
            logger.error(f"Failed to cleanup folder {folder_path}: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current learning status"""
        status = {
            'is_active': self.is_learning_active,
            'current_episode': self.current_episode.to_dict() if self.current_episode else None,
            'data_stats': self.replay_buffer.get_buffer_stats(),
            'system_resources': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent
            },
            'learning_ready': self.should_trigger_learning()[0]
        }

        return status

    def save_state(self):
        """Save orchestrator state"""
        state = {
            'learning_history': [ep.to_dict() for ep in self.learning_history],
            'config': self.config
        }

        state_path = Path(self.config['data_dir']) / 'learning_state.json'
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self):
        """Load orchestrator state"""
        state_path = Path(self.config['data_dir']) / 'learning_state.json'
        if state_path.exists():
            try:
                with open(state_path, 'r') as f:
                    state = json.load(f)
                    # Restore learning history
                    self.learning_history = [
                        LearningEpisode(**ep) for ep in state.get('learning_history', [])
                    ]
                    logger.info(f"Loaded {len(self.learning_history)} historical episodes")
            except Exception as e:
                logger.warning(f"Failed to load learning state: {e}")

def main():
    """Example usage"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-learning', action='store_true', help='Start a learning episode')
    parser.add_argument('--dry-run', action='store_true', help='Dry run without actual training')
    args = parser.parse_args()

    orchestrator = IncrementalLearningOrchestrator()

    if args.start_learning:
        # Start learning and exit
        try:
            if args.dry_run:
                # Just test the data preparation without training
                print("Dry run mode - testing data preparation...")
                data_info = orchestrator._collect_available_data()
                conversation_data = data_info['conversations']
                validated_data = orchestrator.data_validator.validate_dataset(conversation_data)
                approved_conversations = [
                    result['conversation'] for result in validated_data['results']
                    if result['approved']
                ]
                training_data = orchestrator._prepare_training_data(approved_conversations)
                print(f"Would train on {len(training_data['training_examples'])} examples")
                print("Dry run completed successfully")
            else:
                episode_id = orchestrator.start_learning_episode()
                print(f"Started learning episode: {episode_id}")
        except Exception as e:
            print(f"Failed: {e}")
            exit(1)
    else:
        # Check if learning should be triggered
        should_learn, reason = orchestrator.should_trigger_learning()
        print(f"Should learn: {should_learn} - {reason}")

        # Get current status
        status = orchestrator.get_status()
        print(f"Current status: {status['is_active']}")

if __name__ == "__main__":
    main()