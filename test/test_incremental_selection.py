from pathlib import Path
from tempfile import TemporaryDirectory

from surrogate_model.cli.train import (
    _filter_combinations,
    _resolve_component_selection,
    _resolve_id_selection,
)
from surrogate_model.cli.train_incremental import (
    _parser,
    _replace_selected_component_directories,
    _run_incremental_component_tasks,
)
from surrogate_model.utils.excitation_channel_models import (
    CHANNEL_NAMES,
    EXCITATION_NAMES,
)


def main():
    args = _parser().parse_args(
        [
            "--component",
            "damping,residue_phase",
            "--excitation",
            "PW_FDC",
            "--channel",
            "LHT1,23",
            "--parallel_workers",
            "2",
        ]
    )
    components = _resolve_component_selection(args.component)
    excitations = _resolve_id_selection(
        args.excitation, EXCITATION_NAMES, "excitation"
    )
    channels = _resolve_id_selection(
        args.channel,
        {index: name for index, name in enumerate(CHANNEL_NAMES)},
        "channel",
    )
    assert components == {"damping", "residue_phase"}
    assert excitations == {2}
    assert channels == {20, 23}
    assert args.parallel_workers == 2
    assert _filter_combinations(
        [(1, 8), (2, 20), (2, 23), (3, 20)], excitations, channels
    ) == [(2, 20), (2, 23)]

    with TemporaryDirectory(prefix="surrogate_incremental_selection_") as temp_dir:
        root = Path(temp_dir)
        final_dir = root / "PW_FDC" / "LHT1"
        staging_dir = root / "PW_FDC" / ".LHT1.training"
        for component, content in (
            ("frequency", "old-frequency"),
            ("damping", "old-damping"),
            ("residue_magnitude", "old-magnitude"),
            ("residue_phase", "old-phase"),
        ):
            path = final_dir / component
            path.mkdir(parents=True, exist_ok=True)
            (path / "model.pth").write_text(content, encoding="utf-8")
        for component, content in (
            ("damping", "new-damping"),
            ("residue_phase", "new-phase"),
        ):
            path = staging_dir / component
            path.mkdir(parents=True, exist_ok=True)
            (path / "model.pth").write_text(content, encoding="utf-8")

        _replace_selected_component_directories(
            staging_dir, final_dir, ("damping", "residue_phase")
        )

        assert (final_dir / "frequency" / "model.pth").read_text(
            encoding="utf-8"
        ) == "old-frequency"
        assert (final_dir / "residue_magnitude" / "model.pth").read_text(
            encoding="utf-8"
        ) == "old-magnitude"
        assert (final_dir / "damping" / "model.pth").read_text(
            encoding="utf-8"
        ) == "new-damping"
        assert (final_dir / "residue_phase" / "model.pth").read_text(
            encoding="utf-8"
        ) == "new-phase"
        assert not (staging_dir / "damping").exists()
        assert not (final_dir.parent / ".LHT1.component_backup").exists()

    parallel_results = sorted(
        _run_incremental_component_tasks([-3, 2], 2, worker=abs)
    )
    assert parallel_results == [2, 3]
    try:
        list(_run_incremental_component_tasks([], 0, worker=abs))
    except ValueError:
        pass
    else:
        raise AssertionError("parallel_workers=0 was accepted")

    print("incremental selection, replacement, and process pool: PASS")


if __name__ == "__main__":
    main()
