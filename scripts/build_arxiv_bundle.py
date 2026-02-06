#!/usr/bin/env python3
"""Create a timestamped arXiv source bundle from paper/."""

from __future__ import annotations

import argparse
import shutil
import tarfile
from datetime import UTC, datetime
from pathlib import Path

LATEX_BUILD_ARTIFACT_SUFFIXES = {
    ".aux",
    ".bbl",
    ".bcf",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".pdf",
    ".run.xml",
    ".synctex.gz",
    ".toc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path("paper"),
        help="Directory containing main.tex and paper assets.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("outputs/arxiv"),
        help="Root directory for timestamped arXiv bundle outputs.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compile the staged source with latexmk before archiving.",
    )
    return parser.parse_args()


def should_skip(path: Path) -> bool:
    if path.name.startswith("."):
        return True
    if path.suffix in LATEX_BUILD_ARTIFACT_SUFFIXES:
        return True
    return False


def copy_tree_clean(src_root: Path, dst_root: Path) -> None:
    for path in sorted(src_root.rglob("*")):
        rel = path.relative_to(src_root)
        if should_skip(path):
            continue
        target = dst_root / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def check_compile(staging_dir: Path) -> None:
    import subprocess

    cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    subprocess.run(cmd, cwd=staging_dir, check=True)


def build_archive(staging_dir: Path, tar_path: Path) -> None:
    with tarfile.open(tar_path, "w:gz") as tar:
        for path in sorted(staging_dir.rglob("*")):
            if path.is_dir():
                continue
            arcname = path.relative_to(staging_dir)
            tar.add(path, arcname=str(arcname))


def main() -> None:
    args = parse_args()
    paper_dir = args.paper_dir.resolve()
    if not (paper_dir / "main.tex").exists():
        raise FileNotFoundError(f"Expected main.tex under {paper_dir}")

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%SZ")
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    staging_dir = out_root / f"source_{stamp}"
    staging_dir.mkdir(parents=True, exist_ok=True)
    copy_tree_clean(paper_dir, staging_dir)

    if args.check:
        check_compile(staging_dir)

    tar_path = out_root / f"arxiv_source_{stamp}.tar.gz"
    build_archive(staging_dir, tar_path)

    print(f"staging_dir={staging_dir}")
    print(f"tarball={tar_path}")


if __name__ == "__main__":
    main()
