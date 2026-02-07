#!/usr/bin/env python3
"""Build an arXiv-ready source bundle under arxiv/ and arxiv_source.tar.gz."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

SKIP_SUFFIXES = {
    ".aux",
    ".bcf",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
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
        help="Paper directory containing main.tex and main.fls.",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path("arxiv"),
        help="Output source directory for arXiv upload.",
    )
    parser.add_argument(
        "--tar-path",
        type=Path,
        default=Path("arxiv_source.tar.gz"),
        help="Output tar.gz path for arXiv upload.",
    )
    parser.add_argument(
        "--check-compile",
        action="store_true",
        help="Compile staged arXiv sources with latexmk.",
    )
    return parser.parse_args()


def _resolve_input(raw_path: str, paper_dir: Path) -> Path | None:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = paper_dir / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError:
        return None
    return resolved


def collect_dependency_files(paper_dir: Path) -> list[Path]:
    fls_path = paper_dir / "main.fls"
    if not fls_path.exists():
        raise FileNotFoundError(
            f"Missing {fls_path}. Run `make clean-paper && make paper` first."
        )

    paper_root = paper_dir.resolve()
    deps: set[Path] = set()

    for line in fls_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("INPUT "):
            continue
        raw = line[len("INPUT ") :].strip()
        resolved = _resolve_input(raw, paper_root)
        if resolved is None:
            continue
        try:
            rel = resolved.relative_to(paper_root)
        except ValueError:
            continue
        if rel.name.startswith("."):
            continue
        if rel.suffix in SKIP_SUFFIXES:
            continue
        if resolved.is_file():
            deps.add(rel)

    # Ensure core bibliography artifacts are available in the staged bundle.
    if (paper_dir / "references.bib").exists():
        deps.add(Path("references.bib"))
    if (paper_dir / "main.bbl").exists():
        deps.add(Path("main.bbl"))

    deps.add(Path("main.tex"))
    return sorted(deps)


def stage_bundle(paper_dir: Path, bundle_dir: Path, files: list[Path]) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    for rel in files:
        src = paper_dir / rel
        dst = bundle_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def write_manifest(bundle_dir: Path, files: list[Path]) -> None:
    manifest = bundle_dir / "ARXIV_MANIFEST.txt"
    lines = [
        "ArXiv source bundle manifest",
        "",
        "Included files (relative to arxiv/):",
        *[str(p) for p in files],
        "",
        "Notes:",
        "- Bundle built from LaTeX recorder dependencies (main.fls).",
        "- Build artifacts are excluded except main.bbl when present.",
    ]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_bundle(bundle_dir: Path) -> None:
    minted_pkg = re.compile(r"\\usepackage(?:\[[^\]]*\])?\{minted\}")
    for path in bundle_dir.rglob("*"):
        if path.is_file() and " " in path.name:
            raise ValueError(f"Filename contains spaces: {path}")

    for tex in bundle_dir.rglob("*.tex"):
        text = tex.read_text(encoding="utf-8", errors="ignore")
        if "/Users/" in text or "C:\\" in text:
            raise ValueError(f"Absolute path token found in {tex}")
        if minted_pkg.search(text) or "\\inputminted" in text:
            raise ValueError(f"minted package usage found in {tex}")
        if "-shell-escape" in text or "\\write18" in text:
            raise ValueError(f"Potential arXiv-incompatible dependency found in {tex}")


def compile_check(bundle_dir: Path) -> None:
    subprocess.run(
        ["latexmk", "-C"],
        cwd=bundle_dir,
        check=True,
    )
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        cwd=bundle_dir,
        check=True,
    )


def prune_build_artifacts(bundle_dir: Path) -> None:
    for path in bundle_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix in SKIP_SUFFIXES:
            path.unlink()
            continue
        if path.name == "main.pdf":
            path.unlink()


def build_tar(bundle_dir: Path, tar_path: Path, files: list[Path]) -> None:
    with tarfile.open(tar_path, "w:gz") as tar:
        for rel in files:
            path = bundle_dir / rel
            if path.exists() and path.is_file():
                tar.add(path, arcname=str(rel))
        manifest = bundle_dir / "ARXIV_MANIFEST.txt"
        if manifest.exists():
            tar.add(manifest, arcname="ARXIV_MANIFEST.txt")


def main() -> None:
    args = parse_args()
    paper_dir = args.paper_dir.resolve()
    bundle_dir = args.bundle_dir.resolve()
    tar_path = args.tar_path.resolve()

    if not (paper_dir / "main.tex").exists():
        raise FileNotFoundError(f"Expected main.tex under {paper_dir}")

    deps = collect_dependency_files(paper_dir)
    stage_bundle(paper_dir, bundle_dir, deps)
    write_manifest(bundle_dir, deps)
    validate_bundle(bundle_dir)

    if args.check_compile:
        compile_check(bundle_dir)
        prune_build_artifacts(bundle_dir)

    build_tar(bundle_dir, tar_path, deps)

    print(f"bundle_dir={bundle_dir}")
    print(f"tarball={tar_path}")
    print(f"file_count={len(deps)}")


if __name__ == "__main__":
    main()
