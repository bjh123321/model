from surrogate_model.cli.train import _filter_combinations, _resolve_id_selection
from surrogate_model.utils.excitation_channel_models import CHANNEL_NAMES, EXCITATION_NAMES


def main():
    channel_names = {index: name for index, name in enumerate(CHANNEL_NAMES)}
    assert _resolve_id_selection(["FXD_DC"], EXCITATION_NAMES, "excitation") == {1}
    assert _resolve_id_selection(["fxd_fdc,1"], EXCITATION_NAMES, "excitation") == {0, 1}
    assert _resolve_id_selection(["LVT1;9"], channel_names, "channel") == {8, 9}
    assert _resolve_id_selection(None, EXCITATION_NAMES, "excitation") is None

    combinations = [(0, 8), (0, 9), (1, 8), (2, 20)]
    assert _filter_combinations(combinations, {0, 1}, {8}) == [(0, 8), (1, 8)]
    assert _filter_combinations(combinations, {2}, None) == [(2, 20)]
    assert _filter_combinations(combinations, None, {9, 20}) == [(0, 9), (2, 20)]

    try:
        _resolve_id_selection(["NOT_AN_EXCITATION"], EXCITATION_NAMES, "excitation")
    except ValueError as exc:
        assert "Unknown excitation selector" in str(exc)
    else:
        raise AssertionError("Invalid excitation selector was accepted")
    print("training excitation/channel selectors: PASS")


if __name__ == "__main__":
    main()
