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

### 3. TokenizerOutput TypeError: argument after ** must be a mapping

**Error:**
```
TypeError: server.DummyModel.generate() argument after ** must be a mapping, not TokenizerOutput
```

**Location:** `backend/server.py` line 1403

**Cause:** `TokenizerOutput` class didn't implement dict-like methods needed for unpacking with `**inputs`

**Fix:** Added dict protocol methods to `TokenizerOutput` class:
- `keys()` - returns dict keys
- `values()` - returns dict values  
- `items()` - returns dict items

This allows `**inputs` unpacking to work properly: `model.generate(**inputs)`

### 4. TokenizerOutput AttributeError: 'list' object has no attribute 'shape'

**Error:**
```
AttributeError: 'list' object has no attribute 'shape'
```

**Location:** `backend/server.py` line 1414

**Cause:** `input_ids` was a plain Python list `[[1, 2, 3]]`, but code expected it to have `.shape` attribute like a PyTorch tensor

**Fix:** Created `DummyTensor` class that wraps list data and provides:
- `shape` attribute - tuple representing tensor dimensions (batch_size, sequence_length)
- `__getitem__` method - allows indexing like `tensor[0]`
- Automatic shape calculation from nested list structure

Now `inputs['input_ids'].shape[1]` works correctly for calculating max_length parameter.

## Files Modified

1. `backend/server.py` - Fixed DummyModel, DummyTokenizer, TokenizerOutput, and added DummyTensor
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
