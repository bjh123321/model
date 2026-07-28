from types import SimpleNamespace

from surrogate_model.config import Config
from surrogate_model.cli.train import (
    _configure_validation_for_height_selection,
    _filter_combinations,
    _filter_rows_by_heights,
    _height_selection_matches,
    _resolve_component_selection,
    _resolve_component_value_mapping,
    _resolve_height_selection,
    _resolve_id_selection,
)
from surrogate_model.utils.excitation_channel_models import CHANNEL_NAMES, EXCITATION_NAMES


def main():
    assert Config().use_val_split is False

    channel_names = {index: name for index, name in enumerate(CHANNEL_NAMES)}
    assert _resolve_id_selection(["FXD_DC"], EXCITATION_NAMES, "excitation") == {1}
    assert _resolve_id_selection(["fxd_fdc,1"], EXCITATION_NAMES, "excitation") == {0, 1}
    assert _resolve_id_selection(["LVT1;9"], channel_names, "channel") == {8, 9}
    assert _resolve_id_selection(None, EXCITATION_NAMES, "excitation") is None
    assert _resolve_component_selection(None) is None
    assert _resolve_component_selection(["damping"]) == {"damping"}
    assert _resolve_component_selection(
        ["frequency,damping", "residue-phase"]
    ) == {"frequency", "damping", "residue_phase"}
    assert _resolve_component_value_mapping(
        ["frequency=training,residue-phase=validation"],
        label="checkpoint",
        value_type=str,
        choices=("training", "validation"),
    ) == {"frequency": "training", "residue_phase": "validation"}
    assert _resolve_component_value_mapping(
        ["damping=0.75", "frequency=0.2"],
        label="simulation weight",
        value_type=float,
        minimum=0.0,
    ) == {"damping": 0.75, "frequency": 0.2}

    combinations = [(0, 8), (0, 9), (1, 8), (2, 20)]
    assert _filter_combinations(combinations, {0, 1}, {8}) == [(0, 8), (1, 8)]
    assert _filter_combinations(combinations, {2}, None) == [(2, 20)]
    assert _filter_combinations(combinations, None, {9, 20}) == [(0, 9), (2, 20)]

    assert _resolve_height_selection(None) is None
    assert _resolve_height_selection(["5.5, 3.0", "8.23;3"]) == [3.0, 5.5, 8.23]
    rows = [SimpleNamespace(H=3.0), SimpleNamespace(H=5.5), SimpleNamespace(H=8.23)]
    assert [row.H for row in _filter_rows_by_heights(rows, [5.5000001], 1e-6)] == [5.5]

    cfg = SimpleNamespace(
        use_val_split=True,
        val_height_tol=1e-6,
        val_split_strategy="height_mach_points",
        val_height_mach_points=[(3.0, 0.98), (5.5, 1.05)],
        val_heights=[3.0, 8.23],
    )
    _configure_validation_for_height_selection(cfg, [3.0], [SimpleNamespace(H=3.0)])
    assert cfg.val_split_strategy == "height_mach_points"
    assert cfg.val_height_mach_points == [(3.0, 0.98)]
    cfg = SimpleNamespace(
        use_val_split=True,
        val_height_tol=1e-6,
        val_split_strategy="height_mach_points",
        val_height_mach_points=[(3.0, 0.98)],
        val_heights=[3.0, 8.23],
    )
    _configure_validation_for_height_selection(cfg, [5.5], [SimpleNamespace(H=5.5)])
    assert cfg.val_split_strategy == "height_max_mach"
    assert cfg.val_heights == [5.5]

    all_height_config = {"dataset": {"data_selection": {"heights": None, "height_tolerance": 1e-6}}}
    selected_config = {"dataset": {"data_selection": {"heights": [3.0], "height_tolerance": 1e-6}}}
    assert _height_selection_matches(all_height_config, None, 1e-6)[0]
    assert _height_selection_matches(selected_config, [3.0], 1e-6)[0]
    assert not _height_selection_matches(selected_config, [5.5], 1e-6)[0]
    assert not _height_selection_matches(selected_config, None, 1e-6)[0]

    try:
        _resolve_id_selection(["NOT_AN_EXCITATION"], EXCITATION_NAMES, "excitation")
    except ValueError as exc:
        assert "Unknown excitation selector" in str(exc)
    else:
        raise AssertionError("Invalid excitation selector was accepted")
    try:
        _resolve_component_selection(["not_a_component"])
    except ValueError as exc:
        assert "Unknown component selector" in str(exc)
    else:
        raise AssertionError("Invalid component selector was accepted")
    for invalid_height in ("not-a-height", "nan", "inf"):
        try:
            _resolve_height_selection([invalid_height])
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid height selector was accepted: {invalid_height}")
    print("training excitation/channel/height selectors: PASS")


if __name__ == "__main__":
    main()
