#!/usr/bin/env python3
"""Train-only comparison of soft proxy, hard symbol cost, and actual frame bytes."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.model_factory import build_model
from finetuning.representation_study import digest, load_packet, read_manifest, tensor_packet, write_json
from finetuning.run_index_search import FrozenIndexStream
from finetuning.vq_encoder_rd import squared_distances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--calibration-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', action='append', required=True, help='label=checkpoint')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda', help='Inference device; CPU avoids interfering with occupied GPUs')
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo_root.resolve()))
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train = {r['sample_path']: r for r in read_manifest(args.samples_dir)['train']}
    calibration = json.loads((args.calibration_dir / 'calibration.json').read_text())
    rows = [train[name] for name in calibration['train_table_windows']]
    counts = np.load(args.calibration_dir / 'counts.npy', allow_pickle=False)
    stream = FrozenIndexStream(counts, calibration['checkpoint_sha256'])
    costs = -torch.log2(torch.from_numpy(stream.probabilities).to(args.device))
    results = {}
    for specification in args.checkpoint:
        label, name = specification.split('=', 1)
        path = Path(name)
        saved = torch.load(path, map_location='cpu', weights_only=False)
        config = {**saved['config'], 'exact_eval_quantization': True}
        model = build_model(config, saved, args.device)
        del saved
        model.eval().requires_grad_(False)
        total_cells = total_frames = 0
        hard_bits = soft_bits = 0.
        with torch.no_grad():
            for row in rows:
                _, packet = load_packet(row)
                tensor, _ = tensor_packet(packet, args.device)
                h, w = packet.values.shape
                z = model.encoder(F.pad(tensor, (0, (-w) % 16, 0, (-h) % 16)))
                distances = squared_distances(z, model.quantizer.embedding.weight)
                indices = distances.argmin(1)
                soft = (torch.softmax(-distances / calibration['temperature'], dim=1) * costs).sum()
                hard = costs[indices].sum()
                frame = stream.encode(indices, (h, w), z.shape[-2:], packet.side_stream)
                total_cells += tensor.numel(); total_frames += len(frame)
                hard_bits += float(hard); soft_bits += float(soft)
        results[label] = {'checkpoint_sha256': digest(path), 'windows': len(rows), 'cells': total_cells,
            'hard_marginal_bpc': hard_bits / total_cells, 'soft_proxy_bpc': soft_bits / total_cells,
            'frame_bytes': total_frames, 'frame_bpc': total_frames * 8 / total_cells}
        del model
        if torch.device(args.device).type == 'cuda':
            torch.cuda.empty_cache()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / 'results.json', {'checkpoints': results, 'source_sha256': digest(Path(__file__)),
        'calibration_sha256': digest(args.calibration_dir / 'calibration.json'), 'train_only': True,
        'device': args.device, 'parameter_updates': 0,
        'note': 'fixed original prior, identical training table windows; no validation selection or parameter updates'})
    (args.output_dir / 'COMPLETE').touch()
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
