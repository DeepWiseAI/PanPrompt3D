"""Check public-code hygiene and, when present, checkpoint checksums."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "dist",
    "data",
    "datasets",
    "work_dir",
    "outputs",
    "results",
    "logs",
    ".venv",
    "venv",
}
SENSITIVE = [
    re.compile(r"/(?:[m]nt|[U]sers|[h]ome)/"),
    re.compile(r"\b(?:10|192|172)\.(?:\d{1,3}\.){2}\d{1,3}\b"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
]


def public_files():
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if path.is_symlink():
            raise ValueError(f"Symlink is not permitted in release: {rel}")
        if not path.is_file() or any(part in EXCLUDED_DIRS for part in rel.parts):
            continue
        if path.suffix in {".pth", ".pt", ".ckpt", ".pyc"}:
            continue
        if rel.parts[0] == "checkpoints" and path.name not in {
            "README.md",
            "manifest.json",
        }:
            continue
        yield path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(require_weights=False):
    files = list(public_files())
    for path in files:
        text = path.read_text(encoding="utf-8")
        if any(pattern.search(text) for pattern in SENSITIVE):
            raise ValueError(
                f"Sensitive address or credential in {path.relative_to(ROOT)}"
            )
        if path.stat().st_size > 50_000_000:
            raise ValueError("Oversized ordinary-Git file")
    manifest_path = ROOT / "checkpoints/manifest.json"
    checked = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest["files"]:
            path = ROOT / "checkpoints" / entry["filename"]
            if not path.exists():
                if require_weights:
                    raise ValueError(f'Missing weight: {entry["filename"]}')
                continue
            if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
                raise ValueError(f"Weight checksum mismatch: {path.name}")
            checked.append(path.name)
    elif require_weights:
        raise ValueError("Checkpoint manifest missing")
    return {
        "code_files_checked": len(files),
        "private_address_scan": "passed",
        "weight_files_verified": checked,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-weights", action="store_true")
    parser.add_argument(
        "--archive", help="Write a code-only ZIP; existing output is never overwritten"
    )
    args = parser.parse_args()
    report = verify(args.require_weights)
    if args.archive:
        target = Path(args.archive)
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in public_files():
                archive.write(path, "PanPrompt3D/" + str(path.relative_to(ROOT)))
        report["code_archive_created"] = True
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
