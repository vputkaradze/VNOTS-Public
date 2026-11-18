"""Run metadata and validated dataset loading."""
import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from .config import EXPERIMENTS


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_metadata(path, **fields):
    versions = {}
    for name in ("numpy", "scipy", "tensorflow", "keras", "h5py", "matplotlib"):
        try:
            versions[name] = version(name)
        except Exception:
            versions[name] = "unavailable"
    code = Path(__file__).parent
    sources = {str(p.relative_to(code)): sha256(p) for p in sorted(code.rglob("*.py"))}
    payload = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(),
        versions=versions,
        source_sha256=sources,
        **fields,
    )
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def data_path(args):
    return (
        Path(args.data).resolve()
        if args.data
        else Path(args.run_dir) / EXPERIMENTS[args.system].data_file
    )


def load_data(system, path):
    """Return row-major begin/end arrays, preserving the original on-disk formats."""
    if system == "piston":
        # Original piston files are NumPy object arrays; use trusted files only.
        raw = np.load(path, allow_pickle=True)
        if raw.ndim != 3 or raw.shape[0] != 2 or raw.shape[2] != 8:
            raise ValueError("Piston data must have shape (2, pairs, 8).")
        begin, end = np.asarray(raw[0], dtype=float), np.asarray(raw[1], dtype=float)
    else:
        with np.load(path, allow_pickle=False) as raw:
            begin, end = raw["Begin_pts"].T, raw["End_pts"].T
        if begin.ndim != 2 or begin.shape[1] != 9:
            raise ValueError("SO3 Begin_pts must have shape (9, pairs) on disk.")
    if begin.shape != end.shape or len(begin) < 2:
        raise ValueError(
            "Begin/end arrays must have matching shapes and at least two pairs."
        )
    if not np.isfinite(begin).all() or not np.isfinite(end).all():
        raise ValueError("Dataset contains nonfinite values.")
    if not np.allclose(end[:, -1] - begin[:, -1], 0.01, rtol=1e-8, atol=1e-12):
        raise ValueError("All data pairs must have h=0.01 for these experiments.")
    temps = slice(2, 4) if system == "piston" else slice(7, 8)
    if np.any(begin[:, temps] <= 0) or np.any(end[:, temps] <= 0):
        raise ValueError("Temperatures must be positive.")
    if system == "piston" and (
        np.any(abs(begin[:, 0]) >= 2) or np.any(abs(end[:, 0]) >= 2)
    ):
        raise ValueError("Piston position must lie strictly between -2 and 2.")
    return begin, end


def inspect_data(args):
    path = data_path(args)
    begin, end = load_data(args.system, path)
    starts = int(np.count_nonzero(np.isclose(begin[:, -1], 0, atol=1e-12)))
    print(
        f"File: {path}\nTraining pairs: {len(begin):,}\nTrajectories starting at t=0: {starts:,}"
    )
    print(
        f"Row-major begin/end shape: {begin.shape}\nTime step: 0.01\nSHA256: {sha256(path)}"
    )
    expected = EXPERIMENTS[args.system].trajectories * 20
    if len(begin) != expected:
        print(
            f"Note: default release size is {expected:,} pairs; this file contains {len(begin):,}."
        )
