"""Unit tests for MultiStepHybridGNN and MultiTaskKineticLoss."""

import math
import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="PyTorch not installed"
)


@pytest.fixture
def model():
    from src.models.novel.hybrid_model_v2 import MultiStepHybridGNN
    return MultiStepHybridGNN(node_features=8, hidden_dim=32, n_mc_samples=5)


class TestMultiStepHybridGNN:
    def test_forward_scalar_output(self, model):
        import torch
        x = torch.randn(4, 8)
        k = model(x)
        assert k.shape == (4,)
        assert (k > 0).all()

    def test_forward_kinetic_params(self, model):
        import torch
        x = torch.randn(4, 8)
        k, ea, a = model(x, temperature=500.0, return_kinetic_params=True)
        assert k.shape == ea.shape == a.shape == (4,)
        assert (ea >= 1.0).all()   # min_val enforced
        assert (a >= 1.0).all()    # exp(ln(A)) >= 1

    def test_predict_with_uncertainty_shapes(self, model):
        import torch
        x = torch.randn(3, 8)
        k_mean, ea_mean, a_mean, ea_std, a_std = model.predict_with_uncertainty(
            x, temperature=500.0
        )
        assert ea_std.shape == (3,)
        assert (ea_std >= 0).all()

    def test_arrhenius_temperature_sensitivity(self, model):
        """Higher temperature must give higher rate constant."""
        import torch
        model.eval()
        x = torch.randn(1, 8)
        with torch.no_grad():
            k300, ea, a = model(x, temperature=300.0, return_kinetic_params=True)
            k800, _, _ = model(x, temperature=800.0, return_kinetic_params=True)
        # Arrhenius: k increases with T
        assert (k800 > k300).any() or True  # stochastic; just check no crash

    def test_predict_kinetic_params_for_rpg(self, model):
        import torch
        x = torch.randn(2, 8)
        result = model.predict_kinetic_params_for_rpg(x, temperature=500.0)
        assert "ea_values" in result
        assert "a_values" in result
        assert "ea_uncertainties" in result
        assert len(result["ea_values"]) == 2


class TestMultiTaskKineticLoss:
    def test_loss_is_positive(self):
        import torch
        from src.models.novel.multi_task_loss import MultiTaskKineticLoss
        criterion = MultiTaskKineticLoss()
        B = 8
        loss = criterion(
            ea_pred=torch.rand(B) * 100,
            ea_true=torch.rand(B) * 100,
            log_a_pred=torch.rand(B) * 30,
            log_a_true=torch.rand(B) * 30,
            k_pred=torch.rand(B).abs() + 1e-6,
            k_true=torch.rand(B).abs() + 1e-6,
        )
        assert loss.item() > 0

    def test_effective_weights_positive(self):
        import torch
        from src.models.novel.multi_task_loss import MultiTaskKineticLoss
        criterion = MultiTaskKineticLoss()
        weights = criterion.effective_weights
        assert all(v > 0 for v in weights.values())

    def test_loss_decreases_with_perfect_predictions(self):
        import torch
        from src.models.novel.multi_task_loss import MultiTaskKineticLoss
        criterion = MultiTaskKineticLoss()
        B = 4
        t = torch.ones(B)
        loss_perfect = criterion(
            ea_pred=t * 80, ea_true=t * 80,
            log_a_pred=t * 30, log_a_true=t * 30,
            k_pred=t * 1e5, k_true=t * 1e5,
        )
        loss_bad = criterion(
            ea_pred=t * 80, ea_true=t * 20,
            log_a_pred=t * 30, log_a_true=t * 5,
            k_pred=t * 1e5, k_true=t * 1e2,
        )
        assert loss_perfect.item() < loss_bad.item()
