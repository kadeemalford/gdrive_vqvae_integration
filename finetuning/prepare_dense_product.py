"""Fit the registered dense product initialization on fixed training files only."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from finetuning.audit_prospective_preparation import sha256, write_json
from finetuning.fit_lifting_allocation import fitting_rows
from finetuning.model_factory import build_model
from finetuning.robust_escape import encode_robust
from finetuning.run_alphabet_factorization import lloyd
from finetuning.run_spatial_opportunity import load, select


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path("vendor/smartgrids").resolve()))
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    corpus = Path("data/finetune/spatial_band_v1_fold0_20261024")
    source = Path("results/remote_setup/dense_transfer_fold0_weights_p93.pt")
    fitting = fitting_rows(select(corpus))
    if any(row["split"] != "train" for row in fitting):
        raise ValueError("Product preparation is train-only")
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {row["sample_path"]: sha256(corpus / "train" / Path(row["sample_path"]).name) for row in fitting}
    write_json(args.output / "protocol.json", {
        "kind": "dense-product-initialization-v1", "source_sha256": sha256(source),
        "fitting": [row["sample_path"] for row in fitting], "input_sha256": hashes,
        "seed": 2117, "per_window_vector_cap": 256, "lloyd_iterations": 12,
        "basis": "Full128 PCA alternating directions between64D halves, no whitening/truncation",
        "sources": {str(p): sha256(p) for p in [Path(__file__), Path("finetuning/run_alphabet_factorization.py"),
                   Path("finetuning/dense_product_vq.py"), Path("finetuning/factorized_vq.py")]},
        "source_parameter_updates": 0})
    saved = torch.load(source, map_location="cpu", weights_only=False)
    model = build_model({**saved["config"], "exact_eval_quantization": True}, saved, "cuda").eval().requires_grad_(False)
    generator = np.random.default_rng(2117)
    pool = []
    with torch.no_grad():
        for row in fitting:
            path = corpus / "train" / Path(row["sample_path"]).name
            if sha256(path) != hashes[row["sample_path"]]:
                raise ValueError("Preparation input changed")
            packet = encode_robust(load(corpus, row))
            h, w = packet.values.shape
            z = model.encoder(F.pad(torch.from_numpy(packet.values)[None, None].cuda(),
                                    (0, (-w) % 16, 0, (-h) % 16)))
            flat = z.permute(0, 2, 3, 1).reshape(-1, 128).cpu()
            indices = generator.choice(len(flat), min(256, len(flat)), replace=False)
            pool.append(flat[torch.from_numpy(indices)])
    values = torch.cat(pool).double()
    center = values.mean(0)
    centered = values - center
    _, basis = torch.linalg.eigh(centered.T @ centered / len(values))
    basis = basis.flip(1)
    for column in range(128):
        if basis[basis[:, column].abs().argmax(), column] < 0:
            basis[:, column] *= -1
    basis = basis[:, list(range(0, 128, 2)) + list(range(1, 128, 2))].float()
    center = center.float()
    rotated = (values.float() - center) @ basis
    assets = {"left": lloyd(rotated[:, :64], 32, seed=2117),
              "right": lloyd(rotated[:, 64:], 64, seed=2117), "basis": basis, "center": center}
    torch.save(assets, args.output / "assets.pt")
    write_json(args.output / "results.json", {"vectors": len(values), "windows": len(fitting),
        "source_parameter_updates": 0, "lloyd_iterations_per_book": 12,
        "assets_sha256": sha256(args.output / "assets.pt")})
    (args.output / "COMPLETE").touch()
    print(json.dumps({"prepared": True, "vectors": len(values)}), flush=True)


if __name__ == "__main__":
    main()
