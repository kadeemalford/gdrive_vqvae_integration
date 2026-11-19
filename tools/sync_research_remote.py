"""Preserve divergent remote files, then synchronize the local research snapshot."""
import argparse
import hashlib
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import paramiko

REMOTE = "/home/murilo/source/das_febus/gdrive_vqvae_integration"


def connect():
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect("aumo-gpu-001", username="murilo", password=os.environ["SSH_PASSWORD"], timeout=30)
    return client


def command(client, text):
    stdin, stdout, stderr = client.exec_command(text)
    output, errors = stdout.read().decode(), stderr.read().decode()
    status = stdout.channel.recv_exit_status()
    stdin.close()
    stdout.close()
    stderr.close()
    if status:
        raise RuntimeError(f"Remote exit {status}: {errors}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-results", type=Path, action="append", default=[])
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    paths = set(subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], text=True).splitlines())
    paths = sorted(p for p in paths if Path(p).is_file() and
                   (p.startswith(("finetuning/", "compression/", "tests/", "tools/", "progress_reports/", "config/"))
                    or p in ("README.md", ".gitignore", "Makefile") or p.startswith("requirements")))
    for directory in args.include_results:
        paths.extend(str(p) for p in directory.rglob("*") if p.is_file())
    client = connect()
    backup = f"{REMOTE}/results/remote_setup/pre_p90_sync_{stamp}"
    command(client, f"mkdir -p {shlex.quote(backup)}")
    status = command(client, f"git -C {shlex.quote(REMOTE)} status --short")
    sftp = client.open_sftp()
    records = []
    for path in sorted(set(paths)):
        target = f"{REMOTE}/{path}"
        content = Path(path).read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        try:
            with sftp.open(target, "rb") as stream:
                old = stream.read()
        except FileNotFoundError:
            old = None
        if old == content:
            continue
        command(client, f"mkdir -p {shlex.quote(str(Path(target).parent))}")
        if old is not None:
            saved = f"{backup}/{path}"
            command(client, f"mkdir -p {shlex.quote(str(Path(saved).parent))}")
            with sftp.open(saved, "wb") as stream:
                stream.write(old)
        with sftp.open(target + ".p90_upload", "wb") as stream:
            stream.write(content)
        sftp.posix_rename(target + ".p90_upload", target)
        with sftp.open(target, "rb") as stream:
            if hashlib.sha256(stream.read()).hexdigest() != digest:
                raise AssertionError(f"Transfer hash mismatch: {path}")
        records.append({"path": path, "sha256": digest, "previous_sha256":
                        hashlib.sha256(old).hexdigest() if old is not None else None})
    audit = {"timestamp": stamp, "local_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
        "remote_head": command(client, f"git -C {shlex.quote(REMOTE)} rev-parse HEAD").strip(),
        "mode": "verified working-tree snapshot; no reset, clean, commit or removal",
        "previous_status": status, "backup": backup, "updated": records}
    local = Path(f"results/remote_setup/p90_sync_{stamp}.json")
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps(audit, indent=2) + "\n")
    sftp.put(str(local), f"{backup}/sync.json")
    sftp.close()
    client.close()
    print(f"Verified {len(records)} updated files; backup: {backup}")


if __name__ == "__main__":
    main()
