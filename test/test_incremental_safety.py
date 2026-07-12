from copy import deepcopy

import torch

from surrogate_model.cli.train_incremental import (
    choose_safe_checkpoint,
    kd_loss_fn,
    restore_safe_or_initial,
)
from surrogate_model.config import Config
from surrogate_model.data.build_replay_data import select_global_guard_groups
from surrogate_model.data.io import OMEGA_N
from surrogate_model.models.blocks import MFDNN, freeze_base_train_lora_only, load_base_weights_into_lora_model


def _outputs(freq, damping):
    omega_phys = torch.as_tensor(freq) * (2.0 * torch.pi)
    damping = torch.as_tensor(damping)
    alpha = -damping * omega_phys / torch.sqrt((1.0 - damping.pow(2)).clamp_min(1e-8))
    shape = alpha.shape
    return {"alpha": alpha, "omega": omega_phys / float(OMEGA_N), "rho": torch.zeros(shape),
            "sin_theta": torch.zeros(shape), "cos_theta": torch.ones(shape)}


def test_kd_is_standardized_across_scales():
    teacher = _outputs([[10.0]], [[0.1]])
    student = _outputs([[20.0]], [[0.2]])
    loss = kd_loss_fn(student, teacher, torch.ones(1, 1), torch.tensor([10.0, 0.1]), ["freq", "damping"], 1.0)
    assert torch.isclose(loss, torch.tensor(2.0), atol=1e-4)


def test_kd_mask_excludes_invalid_slots():
    teacher = _outputs([[10.0, 10.0]], [[0.1, 0.1]])
    student = _outputs([[10.0, 1e6]], [[0.1, 0.9]])
    loss = kd_loss_fn(student, teacher, torch.tensor([[1.0, 0.0]]), torch.ones(2), ["freq", "damping"], 1.0)
    assert loss.item() < 1e-8


def test_plain_checkpoint_to_lora_preserves_output_and_only_lora_trainable():
    cfg = Config(K_max=2)
    cfg.use_lora = False
    teacher = MFDNN(cfg).eval()
    lora_cfg = deepcopy(cfg)
    lora_cfg.use_lora = True
    student = MFDNN(lora_cfg).eval()
    load_base_weights_into_lora_model(student, teacher.state_dict())
    inputs = (torch.randn(4, 1), torch.randn(4, 1), torch.randint(0, 6, (4,)), torch.randint(0, 34, (4,)))
    with torch.no_grad():
        old, new = teacher(*inputs), student(*inputs)
    for key in ("alpha", "omega", "rho", "sin_theta", "cos_theta"):
        assert torch.allclose(old[key], new[key], atol=1e-6, rtol=1e-6)
    freeze_base_train_lora_only(student)
    names = [name for name, parameter in student.named_parameters() if parameter.requires_grad]
    assert names and all(".A." in name or ".B." in name for name in names)


def test_guard_constraint_rejects_lower_new_loss_when_guard_regresses():
    assert not choose_safe_checkpoint(0.5, 1.06, 1.0, 0.05, 0.8)


def test_no_safe_checkpoint_restores_epoch_zero():
    model = torch.nn.Linear(1, 1, bias=False)
    initial = deepcopy(model.state_dict())
    with torch.no_grad():
        model.weight.fill_(99.0)
    assert not restore_safe_or_initial(model, None, initial)
    assert torch.equal(model.weight, initial["weight"])


def test_guard_selection_is_reproducible_and_cross_excitation():
    keys = [(float(h), float(m), exc) for exc in range(3) for h in (0, 5, 10) for m in (0.5, 1.0, 1.5)]
    targets = [("new", 5.0, 1.0, "exc", 0)]
    first = select_global_guard_groups(keys, targets, max_groups=15, min_groups_per_excitation=2, seed=7)
    second = select_global_guard_groups(keys, targets, max_groups=15, min_groups_per_excitation=2, seed=7)
    assert first == second
    assert {key[2] for key in first} == {0, 1, 2}
    assert (5.0, 1.0, 0) not in first
