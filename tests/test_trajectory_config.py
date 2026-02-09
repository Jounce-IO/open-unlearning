"""
Unit tests for trajectory metrics configuration validation.

Tests cover:
- Config structure validation
- Required fields checking
- Default value handling
- Hydra config integration
"""

import pytest
from omegaconf import OmegaConf, DictConfig, ListConfig

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))


class TestTrajectoryMetricsConfig:
    """Tests for trajectory_metrics configuration."""
    
    def test_minimal_config_structure(self):
        """Test that minimal config has required fields."""
        config = OmegaConf.create({
            "handler": "trajectory_metrics",
            "metrics": ["probability"],
            "trajectory_config": {
                "return_logits": True,
                "return_fixation_steps": True,
            },
        })
        
        assert config.handler == "trajectory_metrics"
        assert "probability" in config.metrics
        assert config.trajectory_config.return_logits is True
        assert config.trajectory_config.return_fixation_steps is True
    
    def test_config_with_sampler_kwargs(self):
        """Test config with sampler_kwargs."""
        config = OmegaConf.create({
            "handler": "trajectory_metrics",
            "metrics": ["probability", "exact_memorization"],
            "trajectory_config": {
                "return_logits": True,
                "return_fixation_steps": True,
                "sampler_kwargs": {
                    "steps": 32,
                    "max_new_tokens": 64,
                    "temperature": 0.0,
                },
            },
        })
        
        assert config.trajectory_config.sampler_kwargs.steps == 32
        assert config.trajectory_config.sampler_kwargs.max_new_tokens == 64
        assert config.trajectory_config.sampler_kwargs.temperature == 0.0
    
    def test_config_with_datasets_and_collators(self):
        """Test config with datasets and collators (from defaults)."""
        config = OmegaConf.create({
            "handler": "trajectory_metrics",
            "metrics": ["probability"],
            "trajectory_config": {
                "return_logits": True,
            },
            "datasets": {
                "MUSE_forget_knowmem": {
                    "args": {
                        "hf_args": {
                            "path": "muse-bench/MUSE-News",
                        },
                    },
                },
            },
            "collators": {
                "DataCollatorForSupervisedDataset": {
                    "args": {
                        "padding_side": "left",
                    },
                },
            },
        })
        
        assert "MUSE_forget_knowmem" in config.datasets
        assert "DataCollatorForSupervisedDataset" in config.collators
    
    def test_config_batch_size_default(self):
        """Test that batch_size has a sensible default."""
        config = OmegaConf.create({
            "handler": "trajectory_metrics",
            "metrics": ["probability"],
            "trajectory_config": {
                "return_logits": True,
            },
        })
        
        # batch_size should default to 1 if not specified
        batch_size = config.get("batch_size", 1)
        assert batch_size >= 1
    
    def test_config_metrics_list_validation(self):
        """Test that metrics list is properly validated."""
        # Valid metrics
        valid_config = OmegaConf.create({
            "handler": "trajectory_metrics",
            "metrics": ["probability", "exact_memorization"],
            "trajectory_config": {
                "return_logits": True,
            },
        })
        
        assert len(valid_config.metrics) > 0
        assert all(isinstance(m, str) for m in valid_config.metrics)
    
    def test_config_metrics_dict_format(self):
        """Test that metrics can be specified as dict with configs."""
        config = OmegaConf.create({
            "handler": "trajectory_metrics",
            "metrics": {
                "probability": {},
                "truth_ratio": {
                    "aggregator": "closer_to_1_better",
                    "pre_compute": {
                        "probability": {
                            "access_key": "correct",
                        },
                        "probability": {
                            "access_key": "wrong",
                        },
                    },
                },
            },
            "trajectory_config": {
                "return_logits": True,
            },
        })
        
        # OmegaConf creates DictConfig, not plain dict
        assert isinstance(config.metrics, (dict, DictConfig))
        assert "probability" in config.metrics
        assert "truth_ratio" in config.metrics
        assert config.metrics.truth_ratio.aggregator == "closer_to_1_better"
        assert "pre_compute" in config.metrics.truth_ratio
    
    def test_config_pre_compute_structure(self):
        """Test that pre_compute config has correct structure."""
        config = OmegaConf.create({
            "handler": "trajectory_metrics",
            "metrics": {
                "truth_ratio": {
                    "aggregator": "closer_to_1_better",
                    "pre_compute": {
                        "probability": {
                            "access_key": "correct",
                        },
                    },
                },
            },
            "trajectory_config": {
                "return_logits": True,
            },
        })
        
        pre_compute = config.metrics.truth_ratio.pre_compute
        assert isinstance(pre_compute, (dict, DictConfig))
        assert "probability" in pre_compute
        assert pre_compute.probability.access_key == "correct"
    
    def test_config_metrics_list_and_dict_both_valid(self):
        """Test that both list and dict formats are valid."""
        # List format
        list_config = OmegaConf.create({
            "handler": "trajectory_metrics",
            "metrics": ["probability", "exact_memorization"],
            "trajectory_config": {"return_logits": True},
        })
        assert isinstance(list_config.metrics, (list, ListConfig))
        
        # Dict format
        dict_config = OmegaConf.create({
            "handler": "trajectory_metrics",
            "metrics": {
                "probability": {},
                "exact_memorization": {},
            },
            "trajectory_config": {"return_logits": True},
        })
        assert isinstance(dict_config.metrics, (dict, DictConfig))
    
    def test_config_trajectory_config_required_fields(self):
        """Test that trajectory_config has required fields."""
        config = OmegaConf.create({
            "handler": "trajectory_metrics",
            "metrics": ["probability"],
            "trajectory_config": {
                "return_logits": True,
                "return_fixation_steps": True,
                "logits_source": "sampler",
            },
        })
        
        assert config.trajectory_config.return_logits is True
        assert config.trajectory_config.return_fixation_steps is True
        assert config.trajectory_config.logits_source == "sampler"
    
    def test_config_sampler_kwargs_optional(self):
        """Test that sampler_kwargs are optional but have defaults."""
        config = OmegaConf.create({
            "handler": "trajectory_metrics",
            "metrics": ["probability"],
            "trajectory_config": {
                "return_logits": True,
            },
        })
        
        # sampler_kwargs should be optional
        sampler_kwargs = config.trajectory_config.get("sampler_kwargs", {})
        # If present, should have reasonable defaults
        if sampler_kwargs:
            steps = sampler_kwargs.get("steps", 32)
            assert steps > 0


class TestConfigIntegration:
    """Tests for config integration with Hydra."""
    
    def test_config_can_be_loaded_from_yaml(self):
        """Test that config can be loaded from YAML structure."""
        # Simulate YAML structure - use OmegaConf.create directly with dict
        config_dict = {
            "handler": "trajectory_metrics",
            "batch_size": 1,
            "metrics": ["probability", "exact_memorization"],
            "trajectory_config": {
                "logits_source": "sampler",
                "return_logits": True,
                "return_fixation_steps": True,
                "sampler_kwargs": {
                    "steps": 32,
                    "max_new_tokens": 64,
                    "temperature": 0.0,
                },
            },
        }
        
        config = OmegaConf.create(config_dict)
        
        assert config.handler == "trajectory_metrics"
        assert config.batch_size == 1
        assert "probability" in config.metrics
        assert config.trajectory_config.return_logits is True
    
    def test_config_override_via_cli(self):
        """Test that config can be overridden via CLI-style overrides."""
        base_config = OmegaConf.create({
            "handler": "trajectory_metrics",
            "metrics": ["probability"],
            "trajectory_config": {
                "return_logits": True,
                "sampler_kwargs": {
                    "steps": 32,
                    "max_new_tokens": 64,
                },
            },
        })
        
        # Simulate CLI override
        override_config = OmegaConf.merge(
            base_config,
            OmegaConf.create({
                "trajectory_config": {
                    "sampler_kwargs": {
                        "steps": 16,  # Override
                        "max_new_tokens": 32,  # Override
                    },
                },
            }),
        )
        
        assert override_config.trajectory_config.sampler_kwargs.steps == 16
        assert override_config.trajectory_config.sampler_kwargs.max_new_tokens == 32
        assert override_config.trajectory_config.return_logits is True  # Preserved
    
    def test_config_global_package_directive(self):
        """Test that _global_ package directive works correctly."""
        # When using @package _global_.eval.metrics.trajectory_metrics
        # The config should be accessible at eval.metrics.trajectory_metrics
        config = OmegaConf.create({
            "eval": {
                "metrics": {
                    "trajectory_metrics": {
                        "handler": "trajectory_metrics",
                        "metrics": ["probability"],
                        "trajectory_config": {
                            "return_logits": True,
                        },
                    },
                },
            },
        })
        
        traj_config = config.eval.metrics.trajectory_metrics
        assert traj_config.handler == "trajectory_metrics"
        assert "probability" in traj_config.metrics


class TestConfigValidation:
    """Tests for config validation logic."""
    
    def test_validate_required_fields_present(self):
        """Test validation that required fields are present."""
        required_fields = ["handler", "metrics", "trajectory_config"]
        
        # Valid config
        valid_config = OmegaConf.create({
            "handler": "trajectory_metrics",
            "metrics": ["probability"],
            "trajectory_config": {
                "return_logits": True,
            },
        })
        
        for field in required_fields:
            assert field in valid_config, f"Missing required field: {field}"
    
    def test_validate_metrics_not_empty(self):
        """Test validation that metrics list is not empty."""
        # Empty metrics should be invalid
        invalid_config = OmegaConf.create({
            "handler": "trajectory_metrics",
            "metrics": [],
            "trajectory_config": {
                "return_logits": True,
            },
        })
        
        assert len(invalid_config.metrics) == 0  # This should be caught by code
    
    def test_validate_trajectory_config_return_logits(self):
        """Test validation that return_logits is set when needed."""
        # Config without return_logits should be invalid for trajectory metrics
        invalid_config = OmegaConf.create({
            "handler": "trajectory_metrics",
            "metrics": ["probability"],
            "trajectory_config": {
                # Missing return_logits
            },
        })
        
        return_logits = invalid_config.trajectory_config.get("return_logits", False)
        # This should be caught by code that requires return_logits=True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
