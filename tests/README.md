# Trajectory Metrics Unit Tests

Comprehensive unit tests for trajectory metrics functionality.

## Test Structure

- `test_trajectory_utils.py` - Tests for trajectory utility functions
  - `stack_logits_history` - Shape validation, single/batch cases
  - `compute_trajectories` - Steps, fixation, ratio trajectory computation
  - `extract_logits_at_step` - Step extraction and bounds checking
  - `decode_logits_to_text` - Text decoding from logits

- `test_trajectory_adapters.py` - Tests for trajectory adapter functions
  - `LogitModelWrapper` - Wrapping logits to be callable as model
  - `compute_logit_metric_at_step` - Shape conversion and metric computation
  - `compute_text_metric_at_step` - Text metric computation

- `test_trajectory_metrics.py` - Tests for main trajectory_metrics function
  - Shape validation for logits_history and fixation_steps
  - Generated portion extraction from full sequence
  - Label extraction and alignment with logits
  - Batch template creation
  - Error handling

- `test_trajectory_config.py` - Tests for configuration validation
  - Config structure validation
  - Required fields checking
  - Default value handling
  - Hydra config integration

## Running Tests

### Install Dependencies

```bash
cd /workspaces/dllm/open-unlearning
pip install pytest pytest-cov
```

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test File

```bash
pytest tests/test_trajectory_utils.py -v
pytest tests/test_trajectory_adapters.py -v
pytest tests/test_trajectory_metrics.py -v
pytest tests/test_trajectory_config.py -v
```

### Run with Coverage

```bash
pytest tests/ --cov=src/evals/metrics --cov-report=html
```

### Run Specific Test Class

```bash
pytest tests/test_trajectory_utils.py::TestStackLogitsHistory -v
```

### Run Specific Test

```bash
pytest tests/test_trajectory_utils.py::TestStackLogitsHistory::test_single_sample_multiple_steps -v
```

## Test Coverage

The tests cover:

1. **Shape Validation**: All tensor shape operations are validated
2. **Boundary Cases**: Edge cases like empty lists, out-of-range indices
3. **Error Handling**: Proper error messages for invalid inputs
4. **Integration**: Mock-based integration tests for full workflow
5. **Config Validation**: Hydra config structure and overrides

## Key Test Scenarios

### Trajectory Utils
- ✅ Empty logits_history raises error
- ✅ Single sample vs batch handling
- ✅ Steps trajectory is copy of R
- ✅ Fixation trajectory lookback logic
- ✅ Ratio trajectory interpolation
- ✅ Shape mismatches raise errors
- ✅ Step extraction bounds checking

### Trajectory Adapters
- ✅ LogitModelWrapper callable interface
- ✅ 2D to 3D logits conversion
- ✅ Device placement for batch tensors
- ✅ Text metric batch creation

### Trajectory Metrics
- ✅ Generated portion extraction (prompt vs generated)
- ✅ Fixation steps extraction for generated region
- ✅ Label alignment with generated length
- ✅ Label padding when short
- ✅ Missing sampler error handling
- ✅ Empty metrics list error handling

### Config
- ✅ Required fields validation
- ✅ Default values
- ✅ CLI override support
- ✅ Hydra package directive

## Continuous Integration

These tests should be run in CI/CD pipelines to catch regressions early.
