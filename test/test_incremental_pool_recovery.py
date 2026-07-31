from concurrent.futures.process import BrokenProcessPool
from unittest.mock import patch

from surrogate_model.cli import train_incremental


class _FakeFuture:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    def result(self):
        if self._error is not None:
            raise self._error
        return self._result


class _FakeExecutor:
    def __init__(self, *args, **kwargs):
        self._submit_count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def submit(self, worker, task):
        self._submit_count += 1
        if self._submit_count == 1:
            return _FakeFuture(
                result={
                    "ok": True,
                    "excitation_id": task["excitation_id"],
                    "channel_id": task["channel_id"],
                    "component": task["component"],
                }
            )
        return _FakeFuture(error=BrokenProcessPool("worker stopped"))


def test_broken_pool_tasks_are_retried_sequentially():
    tasks = [
        {
            "excitation_id": 1,
            "channel_id": 15,
            "component": component,
        }
        for component in ("frequency", "damping", "residue_phase")
    ]
    retried = []

    def worker(task):
        retried.append(task["component"])
        return {
            "ok": True,
            "excitation_id": task["excitation_id"],
            "channel_id": task["channel_id"],
            "component": task["component"],
        }

    with patch.object(
        train_incremental, "ProcessPoolExecutor", _FakeExecutor
    ), patch.object(
        train_incremental, "as_completed", side_effect=lambda futures: futures
    ), patch.object(
        train_incremental,
        "_terminate_completed_windows_workers",
        return_value=0,
    ):
        results = list(
            train_incremental._run_incremental_component_tasks(
                tasks, 2, worker=worker
            )
        )

    assert [result["component"] for result in results] == [
        "frequency",
        "damping",
        "residue_phase",
    ], results
    assert retried == ["damping", "residue_phase"]
    assert all(result["ok"] for result in results)


if __name__ == "__main__":
    test_broken_pool_tasks_are_retried_sequentially()
    print("incremental broken process-pool recovery: PASS")
