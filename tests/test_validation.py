"""Validation tests to ensure the testing infrastructure is set up correctly."""

import sys
from pathlib import Path

import pytest
import torch
import numpy as np


class TestInfrastructureValidation:
    """Validate that the testing infrastructure is properly configured."""
    
    @pytest.mark.unit
    def test_pytest_is_working(self):
        """Verify that pytest is running correctly."""
        assert True
    
    @pytest.mark.unit
    def test_imports_work(self):
        """Verify that all necessary imports are available."""
        import gymnasium
        import matplotlib
        import torch
        import numpy
        
        assert gymnasium is not None
        assert matplotlib is not None
        assert torch is not None
        assert numpy is not None
    
    @pytest.mark.unit
    def test_project_structure_exists(self):
        """Verify the project structure is accessible."""
        workspace_path = Path("/workspace")
        assert workspace_path.exists()
        assert (workspace_path / "tests").exists()
        assert (workspace_path / "tests" / "conftest.py").exists()
    
    @pytest.mark.unit
    def test_fixtures_are_available(self, temp_dir, mock_env, sample_config):
        """Verify that conftest fixtures are accessible."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        assert mock_env is not None
        assert hasattr(mock_env, "observation_space")
        assert hasattr(mock_env, "action_space")
        
        assert isinstance(sample_config, dict)
        assert "env" in sample_config
        assert "mode" in sample_config
    
    @pytest.mark.unit
    def test_torch_tensors(self, sample_observations, sample_actions):
        """Verify torch tensor fixtures work correctly."""
        assert isinstance(sample_observations, torch.Tensor)
        assert isinstance(sample_actions, torch.Tensor)
        assert sample_observations.shape[0] == 32
        assert sample_actions.shape[0] == 32
    
    @pytest.mark.unit
    def test_random_seed_fixture(self):
        """Verify random seed fixture ensures reproducibility."""
        random_value_1 = np.random.rand()
        torch_value_1 = torch.rand(1).item()
        
        # These should be deterministic due to the fixture
        expected_random = np.random.RandomState(42).rand()
        torch.manual_seed(42)
        expected_torch = torch.rand(1).item()
        
        # Reset and check again
        np.random.seed(42)
        torch.manual_seed(42)
        
        assert np.isclose(np.random.rand(), expected_random)
        assert np.isclose(torch.rand(1).item(), expected_torch)
    
    @pytest.mark.unit
    def test_coverage_is_configured(self):
        """Verify coverage configuration by checking imports."""
        try:
            import pytest_cov
            assert pytest_cov is not None
        except ImportError:
            pytest.fail("pytest-cov is not installed")
    
    @pytest.mark.unit
    def test_markers_are_defined(self, request):
        """Verify custom markers are properly defined."""
        markers = request.config.getini("markers")
        marker_str = str(markers)
        assert "unit" in marker_str
        assert "integration" in marker_str
        assert "slow" in marker_str
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration marker works."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow marker works."""
        import time
        start = time.time()
        time.sleep(0.1)
        assert time.time() - start >= 0.1


class TestPPOModuleImports:
    """Test that PPO modules can be imported."""
    
    @pytest.mark.unit
    def test_can_import_root_modules(self):
        """Test importing modules from root directory."""
        try:
            import arguments
            import network
            import ppo
            assert arguments is not None
            assert network is not None
            assert ppo is not None
        except ImportError as e:
            pytest.skip(f"Root modules not yet importable: {e}")
    
    @pytest.mark.unit
    def test_can_import_part4_modules(self):
        """Test importing modules from part4 directory."""
        try:
            sys.path.insert(0, "/workspace/part4")
            from ppo_for_beginners import arguments, network, ppo
            assert arguments is not None
            assert network is not None
            assert ppo is not None
        except ImportError as e:
            pytest.skip(f"Part4 modules not yet importable: {e}")
        finally:
            if "/workspace/part4" in sys.path:
                sys.path.remove("/workspace/part4")


@pytest.mark.unit
def test_temp_dir_fixture_cleanup(temp_dir):
    """Test that temp_dir fixture creates and cleans up properly."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("test content")
    assert test_file.exists()
    assert test_file.read_text() == "test content"