#!/usr/bin/env python3
"""Frozen VQ/continuous diagnosis of selected alphabet models, never a codec claim."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.model_factory import build_model
from finetuning.paired_search_diagnostics import aggregate_metrics, finite_json
from finetuning.representation_study import digest, read_manifest, load_packet, score, write_json, distortion
from finetuning.run_index_search import nearest, vectors
from finetuning.run_spatial_output_study import macro_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--samples-dir', type=Path, required=True)
    p.add_argument('--repo-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available() or '4090' not in torch.cuda.get_device_name() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose one remote RTX 4090')
    sys.path.insert(0, str(args.repo_root.resolve()))
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root, out = args.experiment, args.output_dir
    assert (root / 'COMPLETE').exists()
    protocol = json.loads((root / 'protocol.json').read_text())
    assert digest(args.samples_dir / 'samples_manifest.csv') == protocol['manifest_sha256']
    splits = read_manifest(args.samples_dir)
    lookup = {r['sample_path']: r for rows in splits.values() for r in rows}
    selected = {'train_check': protocol['internal_train_check'], 'val': protocol['validation']}
    out.mkdir(parents=True, exist_ok=False)
    code = ['finetuning/diagnose_alphabet_gap.py', 'finetuning/model_factory.py', 'finetuning/latent_geometry.py']
    write_json(out / 'protocol.json', {'kind': 'selected_alphabet_frozen_quantization_gap_v1',
        'parameter_updates': 0, 'code_sha256': {f: digest(ROOT / f) for f in code},
        'reference_results_sha256': digest(root / 'results.json'), 'selection': selected,
        'metric_note': 'train checks contain constant tiles; normalized distortion is reported separately rather than interpreting huge relative RMSE there',
        'contrast': 'exact hard code vectors versus unquantized encoder vectors, same selected weights/input/decoder',
        'limitations': ['continuous representation is NOT a codec, bitrate-matched reference or proven optimum',
            'continuous vectors may be outside decoder training distribution; gap does not isolate all causal mechanisms',
            'same consumed development and train-file checks; no hyperparameter selection or new updates']})
    for f in code:
        dest = out / 'code_snapshot' / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dest)
    result = {'parameter_updates': 0, 'arms': {}}
    with torch.no_grad():
        for arm in protocol['arms']:
            folder = root / arm
            checkpoint = folder / 'selected.pt'
            descriptor = json.loads((folder / 'descriptor.json').read_text())
            assert digest(checkpoint) == descriptor['checkpoint_sha256']
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model = build_model(saved['config'], saved, 'cuda').eval().requires_grad_(False)
            del saved
            original = {r['sample_path']: r['metrics'] for r in json.loads((folder / 'records.json').read_text())}
            result['arms'][arm] = {'checkpoint_sha256': descriptor['checkpoint_sha256'], 'splits': {}}
            for split, paths in selected.items():
                metrics = {'hard': [], 'continuous': []}
                records = []
                distortions = {'hard': [], 'continuous': []}
                quant_error = latent_power = 0.
                for path in paths:
                    row = lookup[path]
                    assert row['split'] == ('train' if split == 'train_check' else 'val')
                    raw, packet = load_packet(row)
                    tensor = torch.from_numpy(packet.values)[None, None].cuda()
                    valid = torch.from_numpy(packet.valid)[None, None].cuda()
                    h, w = raw.shape
                    z = model.encoder(F.pad(tensor, (0, (-w) % 16, 0, (-h) % 16)))
                    indices = nearest(z, model.quantizer.embedding.weight)
                    q = vectors(indices, model.quantizer.embedding.weight, tuple(z.shape[-2:]))
                    quant_error += float((q.double() - z.double()).square().sum())
                    latent_power += float(z.double().square().sum())
                    item = {'sample_path': path}
                    for mode, latent in (('hard', q), ('continuous', z)):
                        predicted = model.decoder(latent)[..., :h, :w]
                        normalized_loss = float(distortion(predicted, tensor, valid))
                        output = predicted[0, 0].cpu().numpy()
                        metric = finite_json(score(row, raw, packet, output))
                        if split == 'val' and mode == 'hard':
                            assert metric == original[path], 'Frozen hard replay differs from pilot'
                        metrics[mode].append(metric)
                        distortions[mode].append((row['family'], row['configuration'], normalized_loss))
                        item[mode] = metric
                        item[mode + '_normalized_distortion'] = normalized_loss
                    records.append(item)
                summary = {}
                for mode, values in metrics.items():
                    macro, groups = aggregate_metrics(values)
                    summary[mode] = {'macro': macro, 'groups': groups,
                                     'macro_normalized_distortion': macro_loss(distortions[mode])}
                summary['relative_latent_quantization_mse'] = quant_error / max(latent_power, 1e-30)
                summary['continuous_rmse_gain_vs_hard'] = 1 - summary['continuous']['macro']['macro_relative_rmse'] / summary['hard']['macro']['macro_relative_rmse']
                result['arms'][arm]['splits'][split] = summary
                write_json(out / f'{arm}.{split}.records.json', records)
            assert digest(checkpoint) == descriptor['checkpoint_sha256']
            del model
            torch.cuda.empty_cache()
    write_json(out / 'results.json', finite_json(result))
    (out / 'COMPLETE').touch()
    print(json.dumps({a: {s: {'hard': v['hard']['macro']['macro_relative_rmse'],
        'continuous': v['continuous']['macro']['macro_relative_rmse'],
        'gap': v['continuous_rmse_gain_vs_hard'], 'latent_mse': v['relative_latent_quantization_mse']}
        for s, v in r['splits'].items()} for a, r in result['arms'].items()}), flush=True)


if __name__ == '__main__':
    main()
