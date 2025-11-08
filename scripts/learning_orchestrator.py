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
            conversation_data = self._collect_available_data()
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
            quality_score = self.data_validator.assess_quality(conv)['overall']
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
        """Execute the training process with EWC"""
        # This is a placeholder - would implement actual training loop
        # For now, simulate training

        logger.info("Starting training simulation...")

        # Simulate training time
        time.sleep(5)

        training_stats = {
            'epochs_completed': self.config['training_params']['num_epochs'],
            'final_loss': 2.5,  # Simulated
            'learning_rate': self.config['training_params']['learning_rate'],
            'samples_processed': training_data['total_examples'],
            'model_path': f"models/allie_v_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'training_time_seconds': 5
        }

        logger.info("Training simulation completed")
        return training_stats

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
            if cpu_percent > 80:
                return False

            # Check memory
            memory = psutil.virtual_memory()
            if memory.percent > 85:
                return False

            # Check disk space
            disk = psutil.disk_usage('/')
            if disk.percent > 90:
                return False

            return True

        except Exception as e:
            logger.warning(f"Resource check failed: {e}")
            return True  # Default to allowing learning

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
    args = parser.parse_args()

    orchestrator = IncrementalLearningOrchestrator()

    if args.start_learning:
        # Start learning and exit
        try:
            episode_id = orchestrator.start_learning_episode()
            print(f"Started learning episode: {episode_id}")
        except Exception as e:
            print(f"Failed to start learning: {e}")
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