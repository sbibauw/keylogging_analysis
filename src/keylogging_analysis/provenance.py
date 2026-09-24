"""What produced an output file: engine version and commit, config, inputs, cleaning counts."""
import datetime as dt
import hashlib
import json
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def provenance_path(out_csv: Path) -> Path:
    out_csv = Path(out_csv)
    return out_csv.with_name(out_csv.stem + ".provenance.json")


def _pkg_version() -> str:
    try:
        return version("keylogging-analysis")
    except PackageNotFoundError:
        from . import __version__
        return __version__


def _direct_url_git_info() -> dict | None:
    """Git info from importlib.metadata's direct_url.json, for an install
    from a git URL (e.g. ``uv tool run --from git+...@v0.1.0``), where
    REPO_ROOT has no ``.git`` checkout to inspect directly.

    Returns None on anything missing or malformed -- no distribution, no
    direct_url.json, undecodable or invalid JSON, a non-object, or no vcs_info/commit_id -- so the caller
    falls back to "unknown" the same way a plain pip install would.
    """
    try:
        dist = distribution("keylogging-analysis")
        text = dist.read_text("direct_url.json")
    except (PackageNotFoundError, UnicodeDecodeError):
        return None
    if text is None:
        return None
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    vcs_info = data.get("vcs_info")
    if not isinstance(vcs_info, dict) or "commit_id" not in vcs_info:
        return None
    info = {"commit": vcs_info["commit_id"], "source": "direct_url", "dirty": None}
    if "requested_revision" in vcs_info:
        info["requested_revision"] = vcs_info["requested_revision"]
    return info


def _git_state() -> dict:
    if (REPO_ROOT / ".git").exists():
        try:
            commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                    capture_output=True, text=True, check=True).stdout.strip()
            status = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain",
                                     "--untracked-files=no"],
                                    capture_output=True, text=True, check=True).stdout
            return {"commit": commit, "dirty": bool(status.strip()), "source": "checkout"}
        except (OSError, subprocess.CalledProcessError):
            return {"commit": None, "dirty": None, "source": "checkout"}
    direct_url = _direct_url_git_info()
    if direct_url is not None:
        return direct_url
    return {"commit": None, "dirty": None, "source": None}


def build_provenance(*, adapter, inputs, config, report, n_rows_out, output_path=None,
                     argv=None) -> dict:
    prov = {
        "engine": "keylogging_analysis",
        "version": _pkg_version(),
        "git": _git_state(),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "versions": {"pyarrow": version("pyarrow")},
        "adapter": adapter,
        "inputs": [{"path": str(p), "sha256": sha256_file(p), "bytes": Path(p).stat().st_size}
                   for p in inputs],
        "config": config.to_dict(),
        "cleaning": report.to_dict(),
        "rows_out": int(n_rows_out),
        "argv": list(argv) if argv is not None else None,
    }
    if output_path is not None:
        # Computed after the CSV is written, over the file as it actually
        # landed on disk -- lets a downstream reader verify the CSV wasn't
        # altered in transit without trusting rows_out alone.
        prov["output"] = {"sha256": sha256_file(output_path)}
    return prov


def write_provenance(prov: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(prov, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                          encoding="utf-8")
