import copy

import pytest
import torch

from surrogate_model.config import Config
from surrogate_model.models.blocks import TargetMultifidelityMFDNN, count_parameter_groups
from surrogate_model.models.checkpoint_utils import build_model_from_checkpoint


def config(names=("freq",)):
    cfg = Config()
    cfg.model_target_names = list(names)
    cfg.low_hidden = [8, 8]
    cfg.adapt_hidden = [8]
    return cfg


def inputs():
    return (torch.randn(4, 1), torch.randn(4, 1),
            torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long))


@pytest.mark.parametrize("names", [("freq",), ("damping",), ("magnitude",), ("sin_theta", "cos_theta")])
def test_slot_gradient_and_prediction_isolation(names):
    model = TargetMultifidelityMFDNN(config(names))
    x = inputs()
    before = model(*x)
    before[1][names[0]][:, 1].sum().backward()
    for group in (model.low, model.lin, model.adaptation):
        for slot in (0, 2):
            assert all(p.grad is None or torch.count_nonzero(p.grad) == 0
                       for p in group.slots[slot].parameters())
        assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0
                   for p in group.slots[1].parameters())
    with torch.no_grad():
        for group in (model.low, model.lin, model.adaptation):
            for p in group.slots[1].parameters():
                p.add_(torch.randn_like(p) * 0.2)
    after = model(*x)
    for old, new in zip(before, after):
        for name in names:
            assert new[name].shape == (4, 3)
            torch.testing.assert_close(old[name][:, [0, 2]], new[name][:, [0, 2]], rtol=0, atol=0)
        if len(names) == 2:
            norm = new['sin_theta'].square() + new['cos_theta'].square()
            torch.testing.assert_close(norm, torch.ones_like(norm), atol=1e-5, rtol=1e-5)
    assert count_parameter_groups(model)['total_parameters'] > 0


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("modulation", [False, True])
def test_checkpoint_roundtrip_old_and_new(independent, modulation):
    cfg = config()
    cfg.independent_slots = independent
    cfg.use_modulation = modulation
    model = TargetMultifidelityMFDNN(cfg).eval()
    payload = copy.deepcopy(vars(cfg))
    payload.update(model_type='TargetMultifidelityMFDNN', target_names=['freq'],
                   model_state_dict=model.state_dict())
    if not independent:
        payload.pop('independent_slots')
    restored = build_model_from_checkpoint(payload)
    x = inputs()
    for old, new in zip(model(*x), restored(*x)):
        torch.testing.assert_close(old['freq'], new['freq'], rtol=0, atol=0)
