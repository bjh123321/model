import pytest

from surrogate_model.data.OE import ResidueMode


def test_residue_mode_remains_slotted():
    mode = ResidueMode(
        alpha=-1.0,
        omega=2.0,
        residue_abs=3.0,
        residue_phase=4.0,
    )

    assert not hasattr(mode, "__dict__")
    with pytest.raises(AttributeError):
        mode.extra_attribute = 5.0
