"""Merge post-89 research documents by stage, preserving original bytes in an archive."""
import argparse
import gzip
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

STAGES = [
    ((90,), "90-FIDELITY_PER_BIT_NEXT_RESEARCH_REVIEW.md", "Revisão de fidelidade por bit", "Direção e hipóteses da sequência; não é resultado de codec."),
    ((91,), "91-PROSPECTIVE_PREPARATION_AND_SPATIAL_OPPORTUNITY.md", "Preparação prospectiva e oportunidade espacial", "Preparação e contrato de métricas/taxa, com diagnóstico inicial."),
    ((92, 93, 94, 95), "92-LEARNED_LIFTING_STUDY.md", "H1 — lifting, coder e alocação por subbanda", "Quatro braços concluídos; economia lossless do coder não promoveu H1 sobre o VQ."),
    ((96, 97, 98), "96-NEURAL_ANCHOR_INNOVATION_STUDY.md", "H2 — âncoras neurais e inovação", "Preflight congelado e treino conjunto separados; ganho contra o controle, não contra o VQ maduro."),
    ((99, 100), "99-ENCODER_OBSERVATION_STUDY.md", "Observação completa no encoder", "Ganho pequeno; gate de extensão reprovado, sem promoção."),
    ((101, 102), "101-FROZEN_DEVELOPMENT_VALIDATION.md", "Validação de desenvolvimento congelada", "Cobertura integral dos cinco modelos; sem reseleção ou confirmação independente."),
    ((103, 104), "103-FULL_TILE_BOTTLENECK_DIAGNOSTIC.md", "Diagnóstico de gargalos full-tile", "Descritores de erro, índices e covariância; não são limites informacionais."),
    ((105, 106, 107, 108), "105-DENSE_PRODUCT_AND_RATE_MATCH_STUDY.md", "Produto VQ denso e curva de taxa congelada", "Economia de bytes com mais erro; a grade medida não demonstrou dominância RD."),
    ((109, 110, 111, 112), "109-PHYSICAL_METADATA_AND_PROVENANCE.md", "Metadados físicos, clocks e proveniência", "Cabeçalhos e relógios distinguem recortes, mas não certificam independência de aquisições."),
    ((113, 114), "113-CLOCK_ALIGNED_MULTICORE_OPPORTUNITY.md", "Oportunidade multicore nominalmente alinhada", "Probe linear não passou o gate; contexto VQ decodificado não foi lançado."),
    ((115, 116), "115-LOCAL_INTERNAL_NORMALIZATION_STUDY.md", "Normalização interna local", "Via residual ativa, sem avanço prático; sem extensão."),
    ((117,), "117-RESEARCH_SEQUENCE_CONSOLIDATION.md", "Consolidação dos budgets e decisões", "245.760 updates de otimização até 116, mantendo coortes separadas."),
    ((118, 119, 120, 121), "118-DIRECT_DISCRETE_INDEX_STUDY.md", "Decisão discreta de índices e auditoria ampliada", "Redução escalar com regressões físicas; sem promoção ou distilação."),
]
ARCHIVE_DIRECTORY = Path("progress_reports/archive/post_89_consolidation_20261115")
POST_89_START = "A nova direção está no [relatório 90]"
POST_89_END = "\n## Dados e objetivo de fidelidade"


def digest(content):
    return hashlib.sha256(content).hexdigest()


def heading_slug(text):
    text = re.sub(r"[^\w\s-]", "", text.lower(), flags=re.UNICODE)
    return re.sub(r"\s", "-", text)


def rewrite_references(text, mapping):
    for old, info in mapping.items():
        if old == info["target"]:
            continue
        pattern = re.escape(old) + r"(?:#([^\s)`>\"']+))?"
        def replacement(match):
            if info["anchor"]:
                return info["target"] + "#" + info["anchor"] + ("-" + match.group(1) if match.group(1) else "")
            return info["target"] + ("#" + match.group(1) if match.group(1) else "")
        text = re.sub(pattern, replacement, text)
    return text


def historical_part(number, name, text, mapping):
    lines = text.splitlines()
    title = lines[0].removeprefix("# ").strip()
    output = [f'<a id="report-{number}"></a>', "", f"## Registro histórico {number} — {title}", "",
              f"Fonte original preservada no arquivo: `{name}`.", "",
              f'<a id="report-{number}-{heading_slug(title)}"></a>']
    fenced = False
    slugs = {}
    for line in lines[1:]:
        if re.match(r"^\s*(`{3,}|~{3,})", line):
            fenced = not fenced
        heading = re.match(r"^(#{1,6})\s+(.+)$", line) if not fenced else None
        if heading:
            slug = heading_slug(heading.group(2))
            occurrence = slugs.get(slug, 0)
            slugs[slug] = occurrence + 1
            suffix = f"-{occurrence}" if occurrence else ""
            output.append(f'<a id="report-{number}-{slug}{suffix}"></a>')
            output.append("")
            line = "#" * min(6, len(heading.group(1)) + 2) + " " + heading.group(2)
        if not fenced:
            line = re.sub(r"\]\(#([^)]+)\)", rf"](#report-{number}-\1)", line)
        output.append(rewrite_references(line, mapping))
    return "\n".join(output).rstrip() + "\n"


def make_archive(originals):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, content in sorted(originals.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(content), 0o644, 0
            archive.addfile(info, io.BytesIO(content))
    return gzip.compress(buffer.getvalue(), mtime=0)


def consolidate(root, apply=False):
    root = Path(root)
    directory = root / "progress_reports"
    reports = {}
    for path in directory.glob("*.md"):
        prefix = path.name.split("-")[0]
        if prefix.isdigit() and int(prefix) > 89:
            number = int(prefix)
            if number in reports:
                raise FileExistsError(f"Ambiguous report number {number}; preserve both documents")
            reports[number] = path
    if set(reports) != set(range(90, 122)):
        raise ValueError("Expected exactly reports90..121; inspect newer/missing reports before consolidation")
    mapping = {}
    for numbers, target, _, _ in STAGES:
        for number in numbers:
            mapping[reports[number].name] = {"number": number, "target": target,
                "anchor": f"report-{number}" if len(numbers) > 1 else None}
    originals = {str(path.relative_to(root)): path.read_bytes() for path in reports.values()}
    for relative in ("README.md", "progress_reports/FEBUS_GENERALIST_ROADMAP.md"):
        originals[relative] = (root / relative).read_bytes()
    protected = {path: digest(path.read_bytes()) for path in directory.glob("*.md")
                 if path.name.split("-")[0].isdigit() and int(path.name.split("-")[0]) <= 89}
    generated = {}
    for numbers, target, title, summary in STAGES:
        if len(numbers) == 1:
            content = rewrite_references(reports[numbers[0]].read_text(), mapping)
        else:
            content = f"# {title}\n\n"
            content += f"**Etapa consolidada dos registros {', '.join(map(str, numbers))}.** {summary}\n\n"
            content += ("Protocolos, resultados e auditorias foram reunidos sem alterar resultados, budgets ou gates. "
                        "As partes históricas seguem em ordem: estados de lançamento/execução descrevem aquele momento, "
                        "não tarefas ainda pendentes. Consulte as últimas partes para o encerramento.\n\n")
            content += "## Registros desta etapa\n\n"
            content += "\n".join(f"- [Registro {number}](#report-{number}) — "
                                 + reports[number].read_text().splitlines()[0].removeprefix("# ")
                                 for number in numbers) + "\n\n"
            content += "\n---\n\n".join(historical_part(number, reports[number].name, reports[number].read_text(), mapping)
                                           for number in numbers)
        generated[f"progress_reports/{target}"] = content
    # Keep unrelated roadmap material intact; update only references.
    generated["progress_reports/FEBUS_GENERALIST_ROADMAP.md"] = rewrite_references(
        originals["progress_reports/FEBUS_GENERALIST_ROADMAP.md"].decode(), mapping)
    readme = originals["README.md"].decode()
    start, end = readme.index(POST_89_START), readme.index(POST_89_END)
    overview = (
        "A sequência posterior ao 89 está organizada [por etapa](progress_reports/README.md), "
        "com protocolos, resultados e auditorias no mesmo documento.\n\n"
        "- [Revisão de direção](progress_reports/90-FIDELITY_PER_BIT_NEXT_RESEARCH_REVIEW.md) e "
        "[preparação prospectiva](progress_reports/91-PROSPECTIVE_PREPARATION_AND_SPATIAL_OPPORTUNITY.md).\n"
        "- [Lifting/coder/alocação](progress_reports/92-LEARNED_LIFTING_STUDY.md) e "
        "[âncoras/inovação H2](progress_reports/96-NEURAL_ANCHOR_INNOVATION_STUDY.md).\n"
        "- [Observação do encoder](progress_reports/99-ENCODER_OBSERVATION_STUDY.md) e "
        "[validação congelada de desenvolvimento](progress_reports/101-FROZEN_DEVELOPMENT_VALIDATION.md).\n"
        "- [Diagnóstico full-tile](progress_reports/103-FULL_TILE_BOTTLENECK_DIAGNOSTIC.md) e "
        "[produto VQ/curva de taxa](progress_reports/105-DENSE_PRODUCT_AND_RATE_MATCH_STUDY.md).\n"
        "- [Metadados e proveniência](progress_reports/109-PHYSICAL_METADATA_AND_PROVENANCE.md), "
        "[multicore](progress_reports/113-CLOCK_ALIGNED_MULTICORE_OPPORTUNITY.md) e "
        "[normalização interna](progress_reports/115-LOCAL_INTERNAL_NORMALIZATION_STUDY.md).\n"
        "- [Budgets e decisões](progress_reports/117-RESEARCH_SEQUENCE_CONSOLIDATION.md) e "
        "[índices discretos](progress_reports/118-DIRECT_DISCRETE_INDEX_STUDY.md).\n\n"
        "**Estado consolidado:** o VQ maduro permanece a referência prática. Ganhos contra controles "
        "ou em médias normalizadas não aprovaram as guardas físicas; não houve promoção, "
        "distilação ou P3. A validação consumida não é confirmação independente, "
        "e teste/EDAS continuam reservados.\n")
    generated["README.md"] = rewrite_references(readme[:start] + overview + readme[end:], mapping)
    index = ("# Progress reports — índice por etapa\n\n"
             "Os relatórios 1–89 não foram alterados. Os 32 documentos de 90–121 foram "
             "reunidos em **13 documentos**, preservando os números iniciais e as referências "
             "aos registros históricos por âncoras. Lacunas na numeração são intencionais.\n\n"
             "| Documento | Registros anteriores | Etapa e conclusão |\n"
             "|---|---|---|\n")
    for numbers, target, title, summary in STAGES:
        index += f"| [{numbers[0]} — {title}]({target}) | {', '.join(map(str, numbers))} | {summary} |\n"
    index += ("\n## Rastreabilidade\n\n"
              "- Dentro de cada etapa, protocolo e resultado continuam explicitamente separados.\n"
              "- Os números em protocolos, artefatos e relatórios antigos não foram renumerados.\n"
              "- [Mapa completo de caminhos antigos e novos](consolidation_map.json).\n"
              "- [Originais byte a byte, incluindo README/roadmap anteriores](archive/post_89_consolidation_20261115/originals.tar.gz).\n"
              "- [Manifesto SHA-256 da consolidação](archive/post_89_consolidation_20261115/manifest.json).\n\n"
              "Resultados, checkpoints, fontes congelados e datasets não foram movidos ou reescritos. "
              "Para próximas etapas, atualizar o mesmo documento com protocolo e resultado, "
              "em vez de criar um progress report por atualização operacional.\n")
    generated["progress_reports/README.md"] = index
    removed = [path for path in reports.values() if path.name not in {stage[1] for stage in STAGES}]
    new_paths = [root / path for path in generated if root / path not in reports.values()
                 and path not in ("README.md", "progress_reports/FEBUS_GENERALIST_ROADMAP.md")]
    if (any(path.exists() for path in new_paths) or (root / ARCHIVE_DIRECTORY).exists()
            or (directory / "consolidation_map.json").exists()):
        raise FileExistsError("Consolidated output/archive already exists; preserve it and inspect before rerunning")
    plan = {"before": len(reports), "after": len(STAGES), "removed_numbered_documents": len(reports) - len(STAGES),
            "mapping": mapping}
    if not apply:
        return plan
    archive = make_archive(originals)
    archive_directory = root / ARCHIVE_DIRECTORY
    archive_directory.mkdir(parents=True, exist_ok=False)
    (archive_directory / "originals.tar.gz").write_bytes(archive)
    manifest = {"operation": "Documentation-only stage consolidation; no scientific result changes",
                "before": 32, "after": 13, "archive_sha256": digest(archive),
                "original_sha256": {name: digest(content) for name, content in originals.items()},
                "protected_report_sha256": {str(path.relative_to(root)): value for path, value in protected.items()},
                "mapping": mapping}
    # Verify the backup before replacing any report.
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as backup:
        for name, content in originals.items():
            if backup.extractfile(name).read() != content:
                raise ValueError("Original archive verification failed")
    for relative, content in generated.items():
        (root / relative).write_text(content)
    for path in removed:
        # Only the explicit, archived sources are retired; never broad directory removal.
        if digest(path.read_bytes()) != manifest["original_sha256"][str(path.relative_to(root))]:
            raise ValueError("Report changed during consolidation; preserved instead of deleting")
        path.unlink()
    for path, expected in protected.items():
        if digest(path.read_bytes()) != expected:
            raise AssertionError("A report <=89 changed")
    manifest["generated_sha256"] = {path: digest(content.encode()) for path, content in generated.items()}
    (directory / "consolidation_map.json").write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n")
    (archive_directory / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    result = consolidate(args.root, args.apply)
    print(json.dumps({"before": result["before"], "after": result["after"],
                      "reduced_by": result["removed_numbered_documents"], "applied": args.apply}, indent=2))


if __name__ == "__main__":
    main()
