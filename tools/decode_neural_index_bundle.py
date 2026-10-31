#!/usr/bin/env python3
"""Decode one received index-prior bundle window without original DAS access."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from compression.study_bundle import read_bundle
from compression.symbol_stream import SymbolCodec,HEADER
from compression.neural_symbol_stream import NeuralSymbolCodec
from compression.transition_symbol_stream import TransitionSymbolCodec
from finetuning.checkerboard_prior import CheckerboardPrior
from finetuning.model_factory import build_model
from finetuning.latent_geometry import latent_positions
from finetuning.reversible_normalization import decode


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--window',type=int,default=0)
    parser.add_argument('--waveform-checkpoint',type=Path,required=True)
    parser.add_argument('--repo-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Refusing to overwrite existing output')
    if args.window<0:raise ValueError('Window index must be nonnegative')
    torch.set_num_threads(1)
    reader=read_bundle(args.bundle);assets=next(reader)
    sequence=json.loads(assets['sequence.json'])
    if args.window>=len(sequence):raise ValueError('Requested window outside metadata sequence')
    frame=None
    for i,value in enumerate(reader):
        if i==args.window:frame=value;break
    if frame is None:raise ValueError('Requested frame missing from transport')
    if frame[:4]==b'FSC1':
        huffman=None
        for name in ('compact.tables.json','dense.tables.json','huffman.tables.json'):
            if name in assets:
                candidate=SymbolCodec.from_bytes(assets[name])
                if candidate.identifier==HEADER.unpack_from(frame)[-2]:huffman=candidate;break
        if huffman is None:raise ValueError('No matching original Huffman table')
        indices,shape,latent,side=huffman.decode(frame)
    elif frame[:4]==b'FTR1':
        huffman=SymbolCodec.from_bytes(assets['huffman.tables.json'])
        codec=TransitionSymbolCodec.from_bytes(huffman,assets['transition.tables.json'])
        indices,shape,latent,side=codec.decode(frame)
    else:
        huffman=SymbolCodec.from_bytes(assets['huffman.tables.json'])
        descriptor=json.loads(assets['entropy_model.json'])
        with np.load(io.BytesIO(assets['marginal.npz']),allow_pickle=False) as packet:
            marginal=packet['probabilities']
        prior=None
        if frame[5]==1:
            prior_bytes=assets['prior_inference.pt']
            if hashlib.sha256(prior_bytes).hexdigest()!=descriptor['prior_sha256']:
                raise ValueError('Prior fingerprint differs')
            saved=torch.load(io.BytesIO(prior_bytes),map_location='cpu',weights_only=False)
            prior=CheckerboardPrior(marginal);prior.load_state_dict(saved['state_dict'],strict=True);prior.eval()
        codec=NeuralSymbolCodec(huffman,marginal,descriptor,prior)
        indices,shape,latent,side=codec.decode(frame)
    with args.waveform_checkpoint.open('rb') as stream:
        waveform_sha=hashlib.file_digest(stream,'sha256').hexdigest()
    if waveform_sha!=huffman.spec['checkpoint_sha256']:raise ValueError('Waveform model differs from transmitted identifier')
    sample_id=sequence[args.window]
    if not isinstance(sample_id,str):raise ValueError('Unsupported compound-window sequence')
    entries=list(csv.DictReader(io.StringIO(assets['validation_metadata.csv'].decode())))
    entry=next(r for r in entries if Path(r['sample_path']).stem==sample_id)
    if shape!=(int(entry['channels']),int(entry['time_samples'])):raise ValueError('Geometry metadata differs')
    channel_map=next(json.loads(line) for line in assets['channel_maps.jsonl'].decode().splitlines()
                     if json.loads(line)['channel_map_id']==entry['channel_map_id'])
    sys.path.insert(0,str(args.repo_root));torch.set_num_threads(4)
    saved=torch.load(args.waveform_checkpoint,map_location='cpu',weights_only=False)
    if saved.get('finetune',{}).get('normalization')!='robust-escape-v1':raise ValueError('Unsupported normalization')
    model=build_model({**saved['config'],'exact_eval_quantization':True},saved,'cpu').eval();del saved
    if indices.size!=latent_positions(model,*shape):raise ValueError('Latent shape incompatible with waveform model')
    with torch.no_grad():
        vectors=model.quantizer.embedding(torch.from_numpy(indices.astype(np.int64)))
        vectors=vectors.view(1,*latent,model.embedding_dim).permute(0,3,1,2).contiguous()
        normalized=model.decoder(vectors)[0,0,:shape[0],:shape[1]].numpy()
    restored=decode(normalized,side)
    metadata={'sample_id':sample_id,'waveform_sha256':waveform_sha,'shape':list(shape),
              'index_sha256':hashlib.sha256(indices.astype('<u2').tobytes()).hexdigest(),
              'source_window_required':False,'acquisition':entry,'channel_map':channel_map,
              'decoder_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as stream:np.savez_compressed(stream,data=restored,metadata_json=json.dumps(metadata))
    print(json.dumps({k:metadata[k] for k in ('sample_id','shape','index_sha256','source_window_required')},indent=2))


if __name__=='__main__':main()
