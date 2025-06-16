"""Shared pytest fixtures and configuration for all tests."""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Iterator, Dict, Any
from unittest.mock import Mock, MagicMock

import pytest
import torch
import numpy as np
import gymnasium as gym


@pytest.fixture
def temp_dir() -> Iterator[Path]:
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_env() -> Mock:
    """Create a mock Gymnasium environment."""
    env = Mock(spec=gym.Env)
    env.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
    env.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    env.reset.return_value = (np.zeros(4, dtype=np.float32), {})
    env.step.return_value = (np.zeros(4, dtype=np.float32), 0.0, False, False, {})
    return env


@pytest.fixture
def mock_discrete_env() -> Mock:
    """Create a mock Gymnasium environment with discrete action space."""
    env = Mock(spec=gym.Env)
    env.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
    env.action_space = gym.spaces.Discrete(4)
    env.reset.return_value = (np.zeros(4, dtype=np.float32), {})
    env.step.return_value = (np.zeros(4, dtype=np.float32), 0.0, False, False, {})
    return env


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Provide sample configuration for testing."""
    return {
        "env": "CartPole-v1",
        "mode": "train",
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "max_timesteps": 100000,
        "timesteps_per_batch": 2048,
        "n_updates_per_iteration": 10,
        "gamma": 0.99,
        "clip": 0.2,
        "seed": 42,
        "num_minibatches": 4,
        "hidden_dim": 64,
        "save_freq": 10,
        "hyperparameters": {
            "gamma": 0.99,
            "lam": 0.95,
            "clip": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        }
    }


@pytest.fixture
def mock_network() -> Mock:
    """Create a mock neural network."""
    network = MagicMock()
    network.parameters.return_value = [torch.zeros(1, requires_grad=True)]
    network.forward.return_value = (torch.zeros(1, 2), torch.zeros(1))
    return network


@pytest.fixture
def sample_observations() -> torch.Tensor:
    """Generate sample observations for testing."""
    return torch.randn(32, 4)


@pytest.fixture
def sample_actions() -> torch.Tensor:
    """Generate sample actions for testing."""
    return torch.randn(32, 2)


@pytest.fixture
def sample_rewards() -> torch.Tensor:
    """Generate sample rewards for testing."""
    return torch.randn(32)


@pytest.fixture
def sample_batch() -> Dict[str, torch.Tensor]:
    """Generate a sample batch of experience for testing."""
    batch_size = 32
    obs_dim = 4
    act_dim = 2
    
    return {
        "obs": torch.randn(batch_size, obs_dim),
        "acts": torch.randn(batch_size, act_dim),
        "log_probs": torch.randn(batch_size),
        "rewards": torch.randn(batch_size),
        "rewards_to_go": torch.randn(batch_size),
        "advantages": torch.randn(batch_size),
        "dones": torch.zeros(batch_size, dtype=torch.bool),
    }


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def disable_wandb(monkeypatch):
    """Disable wandb for tests."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")


@pytest.fixture
def gpu_available() -> bool:
    """Check if GPU is available for testing."""
    return torch.cuda.is_available()


@pytest.fixture
def mock_logger() -> Mock:
    """Create a mock logger for testing."""
    logger = Mock()
    logger.log.return_value = None
    logger.close.return_value = None
    return logger


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )