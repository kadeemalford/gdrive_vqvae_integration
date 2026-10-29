#!/usr/bin/env python3
"""Maximum-size optimizer and coded-replay smoke before the controlled study."""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/home/murilo-barroso/source/smartgrids")
from finetuning.model_factory import new_model
from finetuning.latent_geometry import latent_positions
from compression.encode_decode_samples import encode_and_decode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the maximum-size smoke")
    rows = []
    torch.set_num_threads(4)
    for geometry in ("spatial-4x", "spatial4-temporal4"):
        for architecture in ("native-v1", "anisotropic-multiscale-v1"):
            torch.manual_seed(2031)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            config = dict(base_channels=64, embedding_dim=128, num_embeddings=2048,
                          downsample_levels=4, commitment_cost=.25,
                          architecture=architecture, latent_geometry=geometry, exact_eval_quantization=True,
                          activation_checkpointing=True)
            model = new_model(config, "cuda")
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            x = torch.randn(1,1,512,1000,device="cuda")
            torch.cuda.synchronize()
            start = time.monotonic()
            y, vq, indices, _ = model(x)
            loss = .5*(x-y).square().mean()+.5*(x-y).abs().mean()+vq
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            assert indices.numel()==latent_positions(model,512,1000)
            assert y.shape==x.shape and torch.isfinite(loss)
            del y, vq, indices, loss
            optimizer.zero_grad(set_to_none=True)
            model.eval()
            with torch.no_grad():
                expected = model(x)[0]
                actual, codes = encode_and_decode(model,x)
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            torch.cuda.synchronize()
            rows.append({"architecture": architecture, "geometry": geometry,
                         "shape": [512,1000], "indices": codes.numel(),
                         "parameters": sum(p.numel() for p in model.parameters()),
                         "elapsed_seconds": time.monotonic()-start,
                         "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                         "coded_replay_passed": True})
            print(rows[-1],flush=True)
            del model, optimizer, x, expected, actual, codes
    (args.output_dir/"summary.json").write_text(json.dumps(rows,indent=2)+"\n")
    (args.output_dir/"COMPLETE").write_text("All maximum-size gradient, AdamW and coded-only replay checks passed.\n")


if __name__=="__main__":
    main()
