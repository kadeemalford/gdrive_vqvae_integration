#!/usr/bin/env python3
"""Download raw FEBUS files listed in the conservative split manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def load_credentials(token_path: Path, client_secret: Path | None) -> Credentials:
    value = Credentials.from_authorized_user_file(token_path, SCOPES) if token_path.exists() else None
    if value and value.expired and value.refresh_token:
        value.refresh(Request())
    if not value or not value.valid:
        if not client_secret:
            raise RuntimeError("OAuth token unavailable; provide --client-secret to authorize read access")
        value = InstalledAppFlow.from_client_secrets_file(client_secret, SCOPES).run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(value.to_json())
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/febus_folds.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/finetune/raw"))
    parser.add_argument("--token-path", type=Path, default=Path.home() / ".config/gdrive-vqvae/token.json")
    parser.add_argument("--client-secret", type=Path)
    args = parser.parse_args()
    with args.manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    service = build("drive", "v3", credentials=load_credentials(args.token_path, args.client_secret))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    print(f"[finetune-download] starting: {len(rows)} raw HDF5 file(s) -> {args.output_dir}")
    for row in rows:
        path = args.output_dir / f"{row['drive_id']}.h5"
        if path.exists():
            continue
        with path.open("wb") as stream:
            downloader = MediaIoBaseDownload(stream, service.files().get_media(fileId=row["drive_id"]))
            done = False
            while not done:
                _, done = downloader.next_chunk()
        downloaded += 1
        print(f"[finetune-download] complete: {row['family']}/{row['configuration']} {row['title']}")
    print(f"[finetune-download] success: downloaded={downloaded}, existing={len(rows) - downloaded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
