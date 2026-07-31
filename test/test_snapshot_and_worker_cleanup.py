from pathlib import Path
from tempfile import TemporaryDirectory

from surrogate_model.cli.train import _terminate_completed_windows_workers
from surrogate_model.utils.model_artifacts import build_dataset_snapshot


class _FakeProcess:
    def __init__(self, alive):
        self._alive = alive
        self.terminated = False

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.terminated = True
        self._alive = False


class _FakeExecutor:
    def __init__(self, processes):
        self._processes = {
            index: process for index, process in enumerate(processes)
        }


def test_sim_snapshot_uses_fz_prefix_and_real_default_stays_sf():
    rows = [[3.0, 0.29, 1.0, 8.0]]
    with TemporaryDirectory(prefix="surrogate_snapshot_prefix_") as temp_dir:
        root = Path(temp_dir)
        sim_files = build_dataset_snapshot(
            rows,
            root / "sim_dataset_snapshot",
            dataset_prefix="FZ",
        )
        real_files = build_dataset_snapshot(
            rows,
            root / "dataset_snapshot",
        )

        assert next(iter(sim_files.values())).name == (
            "FZ_XGD_FXD_DC_H0300_M029.txt"
        )
        assert next(iter(real_files.values())).name == (
            "SF_XGD_FXD_DC_H0300_M029.txt"
        )


def test_completed_windows_workers_are_terminated():
    running = _FakeProcess(alive=True)
    stopped = _FakeProcess(alive=False)
    executor = _FakeExecutor([running, stopped])

    count = _terminate_completed_windows_workers(
        executor, platform_name="win32"
    )

    assert count == 1
    assert running.terminated
    assert not stopped.terminated


def test_non_windows_workers_are_untouched():
    running = _FakeProcess(alive=True)
    executor = _FakeExecutor([running])

    count = _terminate_completed_windows_workers(
        executor, platform_name="linux"
    )

    assert count == 0
    assert not running.terminated


if __name__ == "__main__":
    test_sim_snapshot_uses_fz_prefix_and_real_default_stays_sf()
    test_completed_windows_workers_are_terminated()
    test_non_windows_workers_are_untouched()
    print("snapshot prefix and Windows worker cleanup: PASS")
