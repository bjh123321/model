from pathlib import Path
from unittest.mock import patch

from surrogate_model.cli import train


def test_load_torch_artifact_falls_back_for_torch_113():
    calls = []
    expected = {"mean": [0.0], "std": [1.0]}

    def fake_load(path, **kwargs):
        calls.append((path, dict(kwargs)))
        if kwargs.get("weights_only") is True:
            raise TypeError("'weights_only' is an invalid keyword argument for Unpickler()")
        return expected

    with patch.object(train.torch, "load", side_effect=fake_load):
        result = train._load_torch_artifact(Path("Normalizer_Input_Sim.pth"))

    assert result is expected
    assert calls == [
        (
            Path("Normalizer_Input_Sim.pth"),
            {"map_location": "cpu", "weights_only": True},
        ),
        (Path("Normalizer_Input_Sim.pth"), {"map_location": "cpu"}),
    ]


def test_load_torch_artifact_uses_weights_only_when_supported():
    calls = []
    expected = {"model_state_dict": {}}

    def fake_load(path, **kwargs):
        calls.append((path, dict(kwargs)))
        return expected

    with patch.object(train.torch, "load", side_effect=fake_load):
        result = train._load_torch_artifact(Path("model.pth"))

    assert result is expected
    assert calls == [
        (Path("model.pth"), {"map_location": "cpu", "weights_only": True})
    ]


if __name__ == "__main__":
    test_load_torch_artifact_falls_back_for_torch_113()
    test_load_torch_artifact_uses_weights_only_when_supported()
    print("torch.load compatibility: PASS")
