"""Put the Hugging Face repo together in build/repo/ (nothing is uploaded):
README.md, LICENSE, NOTICE, pyproject.toml, articulino_onnx/, config.json, fp32/ fp16/ int8/*.onnx
(the tools stay in the code repository).

  python tools/assemble.py
"""
import hashlib
import json
import os
import shutil
from pathlib import Path

ONNX = Path(__file__).resolve().parents[1]
BUILD = ONNX / "build"
GRAPHS = ("encoder", "voice", "text", "dit_step", "decoder")


def place(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)                      # same file system: no second copy of the weights
    except OSError:
        shutil.copy2(src, dst)


def main():
    repo = BUILD / "repo"
    if repo.exists():
        shutil.rmtree(repo)
    for f in ("README.md", "LICENSE", "NOTICE", "pyproject.toml"):
        place(ONNX / f, repo / f)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(ONNX / "articulino_onnx", repo / "articulino_onnx", ignore=ignore)
    place(BUILD / "config.json", repo / "config.json")
    sums = {}
    for prec in ("fp32", "fp16", "int8"):
        for g in GRAPHS:
            src = BUILD / prec / f"{g}.onnx"
            place(src, repo / prec / f"{g}.onnx")
            sums[f"{prec}/{g}.onnx"] = {"bytes": src.stat().st_size,
                                        "sha256": hashlib.sha256(src.read_bytes()).hexdigest()}
    (BUILD / "repo-files.json").write_text(json.dumps(sums, indent=1))
    total = sum(p.stat().st_size for p in repo.rglob("*") if p.is_file())
    print(f"{repo}: {total / 1e6:.0f} MB")
    for prec in ("fp32", "fp16", "int8"):
        print(prec, f"{sum(v['bytes'] for k, v in sums.items() if k.startswith(prec)) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
