# Bug Fixes - November 9, 2025

## Fixed Errors

### 1. Tokenizer AttributeError: 'dict' object has no attribute 'to'

**Error:**
```
AttributeError: 'dict' object has no attribute 'to'
```

**Location:** `backend/server.py` line 1372

**Cause:** DummyTokenizer was returning a plain dict instead of an object with `.to()` method

**Fix:** Updated `DummyTokenizer.__call__()` to return a `TokenizerOutput` object that:
- Has a `.to(device)` method for device placement (returns self for chaining)
- Is subscriptable like a dict (`output['input_ids']`)
- Has an `input_ids` attribute for compatibility

Updated `DummyModel` to:
- Have a `device` attribute (set to "cpu")
- Return proper output structure with shape attribute

### 2. EWCTrainer TypeError: unexpected keyword argument 'num_items_in_batch'

**Error:**
```
IncrementalLearningOrchestrator._execute_training.<locals>.EWCTrainer.compute_loss() got an unexpected keyword argument 'num_items_in_batch'
```

**Location:** `scripts/learning_orchestrator.py` line 422

**Cause:** Newer versions of `transformers` library pass `num_items_in_batch` parameter to `compute_loss()`, but the custom EWCTrainer didn't accept it

**Fix:** Updated `EWCTrainer.compute_loss()` signature to accept optional `num_items_in_batch` parameter:

```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    # num_items_in_batch is passed by newer transformers versions but not used here
    ...
```

## Files Modified

1. `backend/server.py` - Fixed DummyModel and DummyTokenizer
2. `scripts/learning_orchestrator.py` - Fixed EWCTrainer.compute_loss signature

## Testing

Server starts successfully:
```
✅ Auto-learning background task started
✅ Application startup complete
✅ Uvicorn running on http://0.0.0.0:8001
```

Both errors resolved - Allie can now:
- Generate responses without tokenizer errors
- Run training episodes without compute_loss signature errors
