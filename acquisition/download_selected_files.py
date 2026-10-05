#!/usr/bin/env python3
"""Download the small pilot selection from Drive without touching source files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    try:
        import gdown
    except ImportError as exc:
        raise SystemExit("Instale gdown ou baixe os arquivos manualmente para data/raw/") from exc

    config = json.loads(args.config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    existing = 0
    print(f"[download] starting: {len(config['files'])} file(s) -> {args.output_dir}")
    for item in config["files"]:
        destination = args.output_dir / item.get("local_filename", f"{item['id']}.h5")
        if destination.exists():
            existing += 1
            print(f"[download] exists: {destination}")
            continue
        print(f"[download] downloading: {item['title']} -> {destination}")
        result = gdown.download(id=item["id"], output=str(destination), quiet=False)
        if not result:
            print(f"[download] failed: {item['title']}")
            raise RuntimeError(f"Falha ao baixar {item['id']}; baixe-o manualmente para {destination}")
        downloaded += 1
        print(f"[download] complete: {destination}")
    print(f"[download] success: downloaded={downloaded}, existing={existing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
