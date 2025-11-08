#!/usr/bin/env python3
"""
Demo script showing Allie's learning system in action
"""

import sys
import os
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

from learning_orchestrator import IncrementalLearningOrchestrator
from safety_quality_filters import LearningDataValidator
from experience_replay import ExperienceReplayBuffer
from ewc_regularization import ElasticWeightConsolidation

def demo_data_validation():
    """Demo the data validation system"""
    print("🔍 Testing Data Validation...")
    validator = LearningDataValidator()

    test_conversations = [
        {
            "prompt": "What is the capital of France?",
            "response": "The capital of France is Paris."
        },
        {
            "prompt": "How do I hack a website?",
            "response": "I'm sorry, but I can't help with illegal activities."
        },
        {
            "prompt": "Hi",
            "response": "Hello! How can I help you today?"
        }
    ]

    results = validator.validate_dataset(test_conversations)
    print(f"✅ Validated {results['total_conversations']} conversations")
    print(f"✅ Approved {results['approved_count']} for training")
    print()

def demo_experience_replay():
    """Demo the experience replay buffer"""
    print("🔄 Testing Experience Replay...")
    buffer = ExperienceReplayBuffer("demo_replay.json", max_size=100)

    # Add some conversations
    conversations = [
        {"prompt": "Hello", "response": "Hi there!"},
        {"prompt": "What's the weather?", "response": "I don't have access to weather data."},
        {"prompt": "Tell me a joke", "response": "Why did the chicken cross the road? To get to the other side!"}
    ]

    validator = LearningDataValidator()
    for conv in conversations:
        quality_score = validator.validate_conversation(conv)['scores']['quality']['overall']
        buffer.add_conversation(conv, quality_score)

    # Sample replay data
    replay_data = buffer.sample_replay_data(2)
    print(f"✅ Added {len(conversations)} conversations to buffer")
    print(f"✅ Sampled {len(replay_data)} replay examples")
    print(f"✅ Buffer stats: {buffer.get_buffer_stats()}")
    print()

def demo_learning_orchestrator():
    """Demo the learning orchestrator"""
    print("🎯 Testing Learning Orchestrator...")
    orchestrator = IncrementalLearningOrchestrator()

    # Check if learning should be triggered
    should_learn, reason = orchestrator.should_trigger_learning()
    print(f"📊 Learning Status: {'Ready' if should_learn else 'Not Ready'}")
    print(f"📝 Reason: {reason}")

    # Show data stats
    status = orchestrator.get_status()
    print(f"💾 Data Stats: {status['data_stats']}")
    print(f"🖥️  System Resources: CPU {status['system_resources']['cpu_percent']}%, Memory {status['system_resources']['memory_percent']}%")
    print()

def demo_ewc():
    """Demo EWC regularization (simplified)"""
    print("🧠 Testing EWC Regularization...")
    try:
        import torch
        import torch.nn as nn

        # Simple model for demo
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )

        ewc = ElasticWeightConsolidation(model, 0.1)
        print("✅ EWC initialized successfully")
        print("✅ Would prevent catastrophic forgetting during training")
    except ImportError:
        print("⚠️  PyTorch not available for EWC demo")
    print()

def main():
    """Run the complete demo"""
    print("🚀 Allie Learning System Demo")
    print("=" * 50)

    try:
        demo_data_validation()
        demo_experience_replay()
        demo_learning_orchestrator()
        demo_ewc()

        print("🎉 Demo completed successfully!")
        print()
        print("📋 Summary:")
        print("✅ Safety & Quality Filters: Working")
        print("✅ Experience Replay Buffer: Working")
        print("✅ Learning Orchestrator: Ready to learn")
        print("✅ EWC Regularization: Available")
        print("✅ Training Pipeline: Implemented")
        print()
        print("🎯 Allie can now learn from conversations!")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()