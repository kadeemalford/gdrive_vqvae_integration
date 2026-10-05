#!/usr/bin/env python3
"""Download selected Drive files through an authenticated OAuth session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def load_credentials(client_secret: Path, token_path: Path) -> Credentials:
    credentials = None
    if token_path.exists():
        credentials = Credentials.from_authorized_user_file(token_path, SCOPES)
    if credentials and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
    if not credentials or not credentials.valid:
        flow = InstalledAppFlow.from_client_secrets_file(client_secret, SCOPES)
        credentials = flow.run_local_server(port=0)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(credentials.to_json())
    return credentials


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--client-secret", type=Path, required=True)
    parser.add_argument(
        "--token-path",
        type=Path,
        default=Path.home() / ".config/gdrive-vqvae/token.json",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    credentials = load_credentials(args.client_secret, args.token_path)
    service = build("drive", "v3", credentials=credentials)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    existing = 0
    print(f"[download-oauth] starting: {len(config['files'])} file(s) -> {args.output_dir}")

    for item in config["files"]:
        destination = args.output_dir / item.get("local_filename", f"{item['id']}.h5")
        if destination.exists():
            existing += 1
            print(f"[download-oauth] exists: {destination}")
            continue
        request = service.files().get_media(fileId=item["id"])
        print(f"[download-oauth] downloading: {item['title']} -> {destination}")
        with destination.open("wb") as stream:
            downloader = MediaIoBaseDownload(stream, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    print(f"  {status.progress() * 100:.1f}%")
        downloaded += 1
        print(f"[download-oauth] complete: {destination}")
    print(f"[download-oauth] success: downloaded={downloaded}, existing={existing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
