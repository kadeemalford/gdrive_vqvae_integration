#!/usr/bin/env python3
"""Build a read-only manifest of raw FEBUS HDF5 files grouped by configuration."""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
FOLDER_MIME = "application/vnd.google-apps.folder"


def credentials(token_path: Path, client_secret: Path | None) -> Credentials:
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


def children(service, folder_id: str):
    page_token = None
    while True:
        response = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken,files(id,name,mimeType,size)",
            pageSize=1000,
            pageToken=page_token,
        ).execute()
        yield from response.get("files", [])
        page_token = response.get("nextPageToken")
        if not page_token:
            return


def raw_hdf5(name: str) -> bool:
    lower = name.lower()
    return name.startswith("Copy of ") and lower.endswith((".h5", ".hdf5")) and "_output" not in lower


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/manifests/febus_drive_inventory.csv"))
    parser.add_argument("--token-path", type=Path, default=Path.home() / ".config/gdrive-vqvae/token.json")
    parser.add_argument("--client-secret", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    families = config["families"]
    service = build("drive", "v3", credentials=credentials(args.token_path, args.client_secret))
    queue = deque([(str(config["drive_root_id"]), [])])
    rows: list[dict[str, object]] = []
    while queue:
        folder_id, path = queue.popleft()
        for item in children(service, folder_id):
            name = item["name"]
            if item["mimeType"] == FOLDER_MIME:
                queue.append((item["id"], [*path, name]))
                continue
            if not path or path[0] not in families or not raw_hdf5(name):
                continue
            if len(path) < 2:
                continue
            family = families[path[0]]
            configuration = path[1]
            rows.append({
                "drive_id": item["id"], "title": name, "file_size": int(item.get("size", 0)),
                "family": family, "configuration": configuration,
                "group_id": f"{family}/{configuration}", "drive_path": "/".join([*path, name]),
            })
    rows.sort(key=lambda row: (str(row["family"]), str(row["configuration"]), str(row["title"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["drive_id", "title", "file_size", "family", "configuration", "group_id", "drive_path"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[finetune-inventory] success: {len(rows)} raw HDF5 file(s), {len({row['group_id'] for row in rows})} configuration group(s) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
