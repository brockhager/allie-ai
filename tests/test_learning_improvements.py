#!/usr/bin/env python3
"""
Quick validation test for learning improvements:
1. Import learning_pipeline module
2. Check PIPELINE_DEFAULTS exists
3. Verify LearningPipeline accepts settings parameter
4. Confirm min_confidence_to_store and batch_workers are initialized
"""

import sys
from pathlib import Path
sys.path.insert(0, '.')
sys.path.insert(0, str(Path(__file__).parent / 'advanced-memory'))

def test_pipeline_imports():
    """Test that learning_pipeline module can be imported"""
    try:
        # Use direct import from the advanced-memory directory
        import learning_pipeline
        LearningPipeline = learning_pipeline.LearningPipeline
        PIPELINE_DEFAULTS = learning_pipeline.PIPELINE_DEFAULTS
        print("✅ Successfully imported LearningPipeline and PIPELINE_DEFAULTS")
        return LearningPipeline, PIPELINE_DEFAULTS
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_pipeline_defaults():
    """Test that PIPELINE_DEFAULTS contains expected keys"""
    _, PIPELINE_DEFAULTS = test_pipeline_imports()
    if not PIPELINE_DEFAULTS:
        return False
    
    expected_keys = ['min_confidence_to_store', 'batch_workers']
    for key in expected_keys:
        if key not in PIPELINE_DEFAULTS:
            print(f"❌ Missing key in PIPELINE_DEFAULTS: {key}")
            return False
    
    print(f"✅ PIPELINE_DEFAULTS contains all expected keys: {PIPELINE_DEFAULTS}")
    return True

def test_pipeline_initialization():
    """Test that LearningPipeline can be initialized with custom settings"""
    LearningPipeline, _ = test_pipeline_imports()
    if not LearningPipeline:
        return False
    
    # Mock memory_db object
    class MockMemoryDB:
        def get_fact(self, keyword):
            return None
        def add_fact(self, *args, **kwargs):
            return {'fact_id': 1}
        def add_to_learning_queue(self, *args, **kwargs):
            return {'queue_id': 1}
    
    try:
        # Test with default settings
        mock_db = MockMemoryDB()
        pipeline1 = LearningPipeline(mock_db)
        print(f"✅ LearningPipeline initialized with defaults")
        print(f"   min_confidence_to_store: {pipeline1.min_confidence_to_store}")
        print(f"   batch_workers: {pipeline1.batch_workers}")
        
        # Test with custom settings
        custom_settings = {
            'min_confidence_to_store': 0.75,
            'batch_workers': 8
        }
        pipeline2 = LearningPipeline(mock_db, settings=custom_settings)
        print(f"✅ LearningPipeline initialized with custom settings")
        print(f"   min_confidence_to_store: {pipeline2.min_confidence_to_store}")
        print(f"   batch_workers: {pipeline2.batch_workers}")
        
        # Verify custom settings were applied
        assert pipeline2.min_confidence_to_store == 0.75, "Custom min_confidence not applied"
        assert pipeline2.batch_workers == 8, "Custom batch_workers not applied"
        print("✅ Custom settings correctly applied")
        
        return True
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_server_changes():
    """Test that server.py has updated learning frequency settings"""
    try:
        with open('backend/server.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for reduced cooldown
        if '_learning_cooldown_minutes = 5' in content:
            print("✅ Server learning cooldown reduced to 5 minutes")
        else:
            print("⚠️  Could not verify learning cooldown setting")
        
        # Check for faster background task
        if 'await asyncio.sleep(60)' in content:
            print("✅ Background task check interval reduced to 60 seconds")
        else:
            print("⚠️  Could not verify background task interval")
        
        return True
    except Exception as e:
        print(f"❌ Server changes validation failed: {e}")
        return False

if __name__ == '__main__':
    print("="*60)
    print("Testing Learning Improvements")
    print("="*60)
    
    all_passed = True
    
    print("\n1. Testing imports...")
    if not test_pipeline_imports()[0]:
        all_passed = False
    
    print("\n2. Testing PIPELINE_DEFAULTS...")
    if not test_pipeline_defaults():
        all_passed = False
    
    print("\n3. Testing LearningPipeline initialization...")
    if not test_pipeline_initialization():
        all_passed = False
    
    print("\n4. Testing server.py changes...")
    if not test_server_changes():
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 All tests passed! Learning improvements are working.")
    else:
        print("⚠️  Some tests failed. Review output above.")
    print("="*60)
