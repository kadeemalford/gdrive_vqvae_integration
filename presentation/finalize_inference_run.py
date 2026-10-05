#!/usr/bin/env python3
"""Collect one inference run into an immutable, self-describing result directory."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--env", type=Path, required=True)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.work_dir / "reconstructions", args.run_dir / "reconstructions")
    shutil.copytree(args.work_dir / "metrics", args.run_dir / "metrics", dirs_exist_ok=True)
    shutil.copytree(args.work_dir / "figures", args.run_dir / "figures")
    shutil.copy2(args.config, args.run_dir / "config.json")
    metadata = {"created_at": datetime.now(timezone.utc).isoformat(), "config_path": str(args.config.resolve()), "checkpoint_path": str(args.checkpoint.resolve()), "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), "env_path": str(args.env.resolve()), "env_contents": args.env.read_text() if args.env.exists() else ""}
    (args.run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[analyze] success: finalized {args.run_dir}")


if __name__ == "__main__":
    main()
