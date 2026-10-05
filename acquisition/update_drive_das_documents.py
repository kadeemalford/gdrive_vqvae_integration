#!/usr/bin/env python3
"""Update the DAS README and fix NaN-safe event selection in SR_processing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


SCOPES = ["https://www.googleapis.com/auth/drive"]
README_ID = "1Eh_ovULjX_EbTAVzqoyLY2VnMjLlHMXT"
SR_PROCESSING_ID = "1czq4103__BL_23gI964aJ8A5NRe_oVEO"


def load_client_config(client_secret: Path | None) -> dict[str, dict[str, object]]:
    if client_secret:
        return json.loads(client_secret.read_text())
    readonly_token = Path.home() / ".config/gdrive-vqvae/token.json"
    if not readonly_token.exists():
        raise RuntimeError("Informe --client-secret ou restaure o token OAuth de leitura")
    token = json.loads(readonly_token.read_text())
    return {
        "installed": {
            "client_id": token["client_id"],
            "client_secret": token["client_secret"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }


def load_credentials(client_secret: Path | None, token_path: Path) -> Credentials:
    credentials = None
    if token_path.exists():
        credentials = Credentials.from_authorized_user_file(token_path, SCOPES)
    if credentials and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
    if not credentials or not credentials.valid:
        flow = InstalledAppFlow.from_client_config(load_client_config(client_secret), SCOPES)
        credentials = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(credentials.to_json())
    return credentials


def patch_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text())
    replacements = 0
    for cell in notebook["cells"]:
        source = cell.get("source")
        if not isinstance(source, list):
            continue
        updated: list[str] = []
        for line in source:
            count = line.count("np.argmax(abs_all)")
            replacements += count
            updated.append(line.replace("np.argmax(abs_all)", "np.nanargmax(abs_all)"))
        cell["source"] = updated
    if replacements != 1:
        raise RuntimeError(f"Expected one event-peak selection, found {replacements}")
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")


def upload(service, file_id: str, path: Path, mime_type: str) -> None:
    service.files().update(
        fileId=file_id,
        media_body=MediaFileUpload(str(path), mimetype=mime_type, resumable=True),
    ).execute()
    print(f"updated: {path.name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-secret", type=Path)
    parser.add_argument("--readme", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument(
        "--token-path",
        type=Path,
        default=Path.home() / ".config/gdrive-vqvae/write_token.json",
    )
    args = parser.parse_args()
    credentials = load_credentials(args.client_secret, args.token_path)
    service = build("drive", "v3", credentials=credentials)
    patch_notebook(args.notebook)
    upload(service, README_ID, args.readme, "text/markdown")
    upload(service, SR_PROCESSING_ID, args.notebook, "application/x-ipynb+json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
