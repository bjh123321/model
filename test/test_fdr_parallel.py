from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from surrogate_model.cli.train import _resume_component_ready, _train_component_worker
from surrogate_model.config import Config
from surrogate_model.data.io import ParsedRow


def _rows():
    parsed = []
    for index, (height, mach) in enumerate(((3.0, 0.8), (3.0, 0.9), (5.5, 1.0), (8.23, 1.2))):
        freq = np.asarray([8 + index, 16 + index, 24 + index], dtype=np.float32)
        damping = np.asarray([0.05, 0.08, 0.10], dtype=np.float32)
        omega = 2.0 * np.pi * freq
        alpha = -damping * omega / np.sqrt(1.0 - damping * damping)
        magnitude = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        rho = np.log(magnitude)
        sin_theta = np.asarray([0.0, 0.6, 1.0], dtype=np.float32)
        cos_theta = np.asarray([1.0, 0.8, 0.0], dtype=np.float32)
        parsed.append(
            ParsedRow(
                height,
                mach,
                1,
                8,
                3,
                alpha.astype(np.float32),
                omega.astype(np.float32),
                rho,
                sin_theta,
                cos_theta,
                np.ones(3, dtype=np.float32),
                freq,
                damping,
                damping * 100.0,
                magnitude * cos_theta,
                magnitude * sin_theta,
            )
        )
    return parsed


def _config():
    cfg = Config()
    cfg.epochs_stage1 = 1
    cfg.epochs_stage2 = 1
    cfg.early_stop_patience_s1 = 0
    cfg.checkpoint_selection = "training"
    cfg.use_val_split = False
    cfg.num_workers = 0
    cfg.batch_real = 2
    cfg.low_hidden = [8, 8]
    cfg.embed_excitation_dim = 2
    cfg.embed_channel_dim = 2
    cfg.embed_excitation_num = 6
    cfg.embed_channel_num = 34
    cfg.use_modulation = False
    cfg.explicit_train_group_keys = [(300, 80, 1), (300, 90, 1), (550, 100, 1), (823, 120, 1)]
    cfg.explicit_validation_group_keys = []
    cfg.require_validation = False
    cfg.validation_metric_scope = "all"
    return cfg


def main():
    with TemporaryDirectory(prefix="surrogate_fdr_parallel_") as temp_dir:
        version_dir = Path(temp_dir) / "FXD_DC" / "LVT1"
        args = SimpleNamespace(
            device="cpu",
            kfold=0,
            debug_overfit_n=0,
            debug_overfit_offset=0,
            parallel_workers=2,
        )
        metadata = {
            "manifest": {"dataset_fingerprint": "parallel-smoke"},
            "training_mode": "base_multifidelity_training",
            "sim_dataset_fingerprint": "parallel-smoke-sim",
            "fidelity_mode": "multifidelity",
            "prediction_output": "high",
            "excitation_id": 1,
            "excitation_name": "FXD_DC",
            "channel_id": 8,
        }
        tasks = []
        for component in ("frequency", "damping"):
            tasks.append(
                {
                    "cfg": _config(),
                    "sim_parsed": _rows(),
                    "real_parsed": _rows(),
                    "model_version": "parallel-smoke",
                    "version_dir": str(version_dir),
                    "args": args,
                    "dataset_metadata": metadata,
                    "excitation_id": 1,
                    "channel_id": 8,
                    "component": component,
                }
            )
        with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as executor:
            results = list(executor.map(_train_component_worker, tasks))
        assert all(result["ok"] for result in results), results
        for component in ("frequency", "damping"):
            ready, reason = _resume_component_ready(
                version_dir,
                1,
                8,
                component,
                required_profile_version=5,
                expected_fidelity_mode="multifidelity",
            )
            assert ready, reason
        ready, _reason = _resume_component_ready(version_dir, 1, 8, "residue_phase")
        assert not ready
    print("parallel component training and component-level resume: PASS")


if __name__ == "__main__":
    main()
