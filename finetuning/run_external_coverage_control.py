#!/usr/bin/env python3
"""Frozen input-coverage ablation scored on exactly the same retained physical rows."""
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
from finetuning.affine_codec_study import decode_affine_frame
from finetuning.model_factory import build_model
from finetuning.paired_search_diagnostics import finite_json
from finetuning.representation_study import digest, write_json
from finetuning.robust_escape import encode_robust
from finetuning.run_external_smf_stress import pooled_score
from finetuning.run_index_search import FrozenIndexStream, nearest, vectors
from finetuning.reversible_normalization import decode
from finetuning.spatial_band_filter import contiguous_regions


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline-experiment', type=Path, required=True)
    p.add_argument('--samples-dir', type=Path, required=True)
    p.add_argument('--compact', type=Path, required=True)
    p.add_argument('--dense', type=Path, required=True)
    p.add_argument('--repo-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available() or '4060' not in torch.cuda.get_device_name():
        raise RuntimeError('Use the local RTX 4060')
    sys.path.insert(0, str(args.repo_root.resolve()))
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    baseline = json.loads((args.baseline_experiment / 'protocol.json').read_text())
    diagnostic_path = args.baseline_experiment / 'coverage_seam_diagnostic.json'
    diagnostic = json.loads(diagnostic_path.read_text())
    decisions = {r['drive_id']: r['fixed_policy_decision'] for r in diagnostic['files']}
    samples = json.loads((args.samples_dir / 'samples.json').read_text())
    hashes = {arm: digest(getattr(args, arm)) for arm in ('compact', 'dense')}
    assert hashes == baseline['source_checkpoint_sha256']
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    code = ['finetuning/run_external_coverage_control.py', 'finetuning/run_external_smf_stress.py',
            'finetuning/spatial_band_filter.py', 'finetuning/run_index_search.py']
    write_json(out / 'protocol.json', {'kind': 'external_development_common_coverage_control_v1',
        'parameter_updates': 0, 'gpu': torch.cuda.get_device_name(), 'source_checkpoint_sha256': hashes,
        'baseline_results_sha256': digest(args.baseline_experiment / 'results.json'),
        'fixed_decision_sha256': digest(diagnostic_path), 'code_sha256': {s: digest(ROOT / s) for s in code},
        'contrast': 'full-zone decoded output restricted to fixed retained rows versus re-encoding those same rows as separate contiguous regions',
        'target': 'exact same fully finite retained rows in both predictions; all finite target cells scored; escapes counted separately',
        'selection': 'unchanged historical amplitude heuristic decisions from prior diagnostic, based only on raw signal; no new thresholds or example selection',
        'rate_scope': 'frame bytes for retained-region experiment only; NO claim of full-zone compression, excluded rows not delivered by this ablation',
        'limitations': ['external DEVELOPMENT reuse, not independent confirmation',
            'joint normalization/context/padding input change; not isolated proof of any one mechanism',
            'heuristic based on the available one-second block, not the historical whole-file observation dose',
            'no claim that amplitude-rejected channels are physically invalid; primary full-zone results unchanged']})
    for name in code:
        dest = out / 'code_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    records = []
    with torch.no_grad():
        for arm in ('compact', 'dense'):
            saved = torch.load(getattr(args, arm), map_location='cpu', weights_only=False)
            model = build_model(saved['config'], saved, 'cuda').eval().requires_grad_(False)
            del saved
            stream = FrozenIndexStream(np.load(args.baseline_experiment / arm / 'counts.npy', allow_pickle=False), hashes[arm])
            folder = out / arm
            folder.mkdir()
            for file_index, sample in enumerate(samples):
                assert digest(args.samples_dir / sample['path']) == sample['sha256']
                with np.load(args.samples_dir / sample['path'], allow_pickle=False) as data:
                    raw = data['data']
                keep = np.isfinite(raw).all(axis=1)
                keep[decisions[sample['drive_id']]['removed_source_channel_indices']] = False
                assert keep.any()
                regions = [r[j:j + 512] for r in contiguous_regions(keep) for j in range(0, len(r), 512)]
                prediction = np.zeros(raw.shape, dtype=np.float32)
                size = escapes = 0
                for window in range(20):
                    for region, indices in enumerate(regions):
                        assert len(indices) == 1 or np.all(np.diff(indices) == 1)
                        part = raw[indices, window * 1000:(window + 1) * 1000].copy()
                        packet = encode_robust(part)
                        h, w = part.shape
                        target = torch.from_numpy(packet.values)[None, None].cuda()
                        z = model.encoder(F.pad(target, (0, (-w) % 16, 0, (-h) % 16)))
                        symbols = nearest(z, model.quantizer.embedding.weight)
                        latent = tuple(z.shape[-2:])
                        decoded = model.decoder(vectors(symbols, model.quantizer.embedding.weight, latent))[..., :h, :w][0, 0].cpu().numpy()
                        physical = decode(decoded, packet.side_stream)
                        frame = stream.encode(symbols, part.shape, latent, packet.side_stream)
                        np.testing.assert_array_equal(decode_affine_frame(frame, stream, model, 'cuda'), physical)
                        prediction[indices, window * 1000:(window + 1) * 1000] = physical
                        (folder / f'f{file_index:02d}_w{window:02d}_r{region}.frame').write_bytes(frame)
                        size += len(frame)
                        escapes += int((~packet.valid).sum())
                with np.load(args.baseline_experiment / arm / f'f{file_index:02d}.direct.reconstruction.npz', allow_pickle=False) as data:
                    old_prediction = data['data'][keep]
                target = raw[keep]
                all_target = np.ones(target.shape, dtype=bool)
                before, after = pooled_score(target, old_prediction, all_target), pooled_score(target, prediction[keep], all_target)
                records.append({'arm': arm, 'drive_id': sample['drive_id'], 'orientation': sample['orientation'],
                    'stimulus_label_hz': sample['stimulus_label_hz'], 'retained_rows': int(keep.sum()),
                    'source_rows': raw.shape[0], 'retained_cells': target.size, 'regions': [r.tolist() for r in regions],
                    'full_input_common_target': before, 'retained_input_common_target': after,
                    'rmse_gain_on_common_target': 1 - after['relative_rmse'] / before['relative_rmse'],
                    'retained_frames_bytes': size, 'finite_exact_escape_cells_in_retained_codec': escapes})
                np.savez_compressed(folder / f'f{file_index:02d}.retained_reconstruction.npz',
                                    data=prediction[keep], source_channel_indices=np.flatnonzero(keep))
            del model
            torch.cuda.empty_cache()
    result = {'parameter_updates': 0, 'records': records,
        'comparison_scope': 'only common retained target; not a full-zone compression or physical-validity claim'}
    assert all(digest(getattr(args, arm)) == hashes[arm] for arm in hashes)
    write_json(out / 'results.json', finite_json(result))
    (out / 'COMPLETE').touch()
    print(json.dumps([{'arm': r['arm'], 'orientation': r['orientation'], 'hz': r['stimulus_label_hz'],
        'rows': r['retained_rows'], 'before': r['full_input_common_target']['relative_rmse'],
        'after': r['retained_input_common_target']['relative_rmse'], 'gain': r['rmse_gain_on_common_target']} for r in records]), flush=True)


if __name__ == '__main__':
    main()
