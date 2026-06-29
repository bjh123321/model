from __future__ import annotations


def main():
    raise SystemExit(
        "online_ridge is disabled on user_for_win7_fixed. "
        "This branch trains fixed slots [freq, damping, residue_re, residue_im] "
        "and does not expose the old modal-existence u-space required by ridge correction. "
        "Use python -m surrogate_model.cli.train_incremental for fixed-slot fine tuning."
    )


if __name__ == "__main__":
    main()
