"""Renumber canonical post-89 documents without changing historical record IDs."""
import argparse
import hashlib
import io
import json
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def renumber(root, apply=False):
    root = Path(root)
    directory = root / "progress_reports"
    reports = sorted(
        (p for p in directory.glob("*.md")
         if p.name.split("-")[0].isdigit() and int(p.name.split("-")[0]) > 89),
        key=lambda p: int(p.name.split("-")[0]))
    numbers = [int(p.name.split("-")[0]) for p in reports]
    if len(set(numbers)) != len(numbers):
        raise ValueError("Duplicate report numbers; preserve and inspect")
    mapping = {p.name: f"{number}-{p.name.split('-', 1)[1]}"
               for number, p in enumerate(reports, 90)}
    if all(old == new for old, new in mapping.items()):
        return {"changed": 0, "mapping": mapping}
    if any((directory / new).exists() and new not in mapping for new in mapping.values()):
        raise FileExistsError("Destination exists outside the source set")
    paths = [root / "README.md", *directory.glob("*.md"), directory / "consolidation_map.json"]
    originals = {str(p.relative_to(root)): p.read_bytes() for p in paths if p.exists()}
    pattern = re.compile("|".join(re.escape(old) for old in mapping))
    generated = {}
    for relative, content in originals.items():
        text = content.decode()
        # Simultaneous substitution prevents collisions with newly assigned numbers.
        updated = pattern.sub(lambda m: mapping[m.group(0)], text)
        if relative == "progress_reports/README.md":
            updated = updated.replace(
                "preservando os números iniciais e as referências aos registros históricos por âncoras. Lacunas na numeração são intencionais.",
                "com etapas atuais renumeradas em sequência a partir do 90. Os números dos registros históricos e suas âncoras foram preservados.")
            for old, new in mapping.items():
                number = new.split("-")[0]
                updated = re.sub(r"(\| \[)\d+( — [^\n]*?\]\(" + re.escape(new) + r"\))",
                                 lambda m: m.group(1) + number + m.group(2), updated)
            updated = updated.replace("## Rastreabilidade\n", "## Rastreabilidade\n\n"
                "- [Mapa da renumeração dos documentos](numbering_map.json). "
                "Referências históricas a protocolos não são números atuais de etapa.\n")
        destination = "progress_reports/" + mapping.get(Path(relative).name, Path(relative).name) if relative.startswith("progress_reports/") else relative
        if updated != text or destination != relative:
            generated[destination] = updated.encode()
    if not apply:
        return {"changed": sum(a != b for a, b in mapping.items()), "mapping": mapping}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = directory / "archive" / f"sequential_numbering_{stamp}"
    archive.mkdir(parents=True, exist_ok=False)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as backup:
        for relative, content in originals.items():
            entry = tarfile.TarInfo(relative)
            entry.size = len(content)
            backup.addfile(entry, io.BytesIO(content))
    data = buffer.getvalue()
    (archive / "originals.tar.gz").write_bytes(data)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as backup:
        for relative, content in originals.items():
            if backup.extractfile(relative).read() != content:
                raise AssertionError("Backup differs from original")
    for relative, content in originals.items():
        if (root / relative).read_bytes() != content:
            raise RuntimeError("Concurrent document change; original preserved")
    for destination, content in generated.items():
        (root / destination).write_bytes(content)
    for old in mapping:
        if old not in mapping.values():
            (directory / old).unlink()
    (directory / "numbering_map.json").write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n")
    manifest = {
        "purpose": "Canonical document numbering only; historical record IDs and scientific artifacts unchanged",
        "mapping": mapping, "archive_sha256": hashlib.sha256(data).hexdigest(),
        "original_sha256": {p: hashlib.sha256(b).hexdigest() for p, b in originals.items()},
        "updated_sha256": {p: hashlib.sha256(b).hexdigest() for p, b in generated.items()}}
    (archive / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return {"changed": sum(a != b for a, b in mapping.items()), "archive": str(archive), "mapping": mapping}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(renumber(args.root, args.apply), indent=2, ensure_ascii=False))
