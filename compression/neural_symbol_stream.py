"""Queue range coding of anchors followed by conditionally predicted indices.

The prior is evaluated on CPU on both sides. Software/hardware probability
reproducibility remains a deployment constraint; decoded-index CRC fails closed.
"""
import hashlib
import json
import struct
import zlib

import constriction
import numpy as np

from compression.symbol_stream import HEADER,MAGIC as HUFFMAN_MAGIC
from finetuning.checkerboard_prior import anchor_mask

MAGIC=b'FNP1'
INDEX_CRC=struct.Struct('<I')


class NeuralSymbolCodec:
    def __init__(self,huffman,marginal,descriptor,prior=None):
        self.huffman=huffman
        self.marginal=np.asarray(marginal,dtype=np.float32)
        if self.marginal.shape!=(2048,) or not np.isfinite(self.marginal).all() or np.any(self.marginal<=0):
            raise ValueError('Invalid marginal probabilities')
        self.descriptor=descriptor
        if descriptor['huffman_table_sha256']!=huffman.identifier.hex():
            raise ValueError('Incompatible Huffman dictionary')
        if descriptor.get('marginal_sha256') not in (None,hashlib.sha256(self.marginal.tobytes()).hexdigest()):
            raise ValueError('Marginal probability fingerprint differs')
        self.serialized=json.dumps(descriptor,sort_keys=True,separators=(',',':')).encode()
        self.identifier=hashlib.sha256(self.serialized).digest()
        self.prior=prior
        self.fixed=constriction.stream.model.Categorical(self.marginal,perfect=False)
        self.family=constriction.stream.model.Categorical(perfect=False)

    def frame(self,method,old_method,shape,latent,side,payload):
        header=HEADER.pack(MAGIC,1,method,old_method,0,*shape,*latent,len(side),self.identifier,0)
        crc=zlib.crc32(header+side+payload)
        return header[:-4]+struct.pack('<I',crc)+side+payload

    def encode(self,original,force_method=None):
        indices,shape,latent,side=self.huffman.decode(original)
        fields=HEADER.unpack_from(original)
        old_payload=original[HEADER.size+len(side):]
        fallback=self.frame(0,fields[2],shape,latent,side,old_payload)
        values=indices.astype(np.int32)
        checksum=INDEX_CRC.pack(zlib.crc32(indices.astype('<u2').tobytes()))
        encoder=constriction.stream.queue.RangeEncoder();encoder.encode(values,self.fixed)
        marginal=self.frame(2,0,shape,latent,side,checksum+encoder.get_compressed().astype('<u4').tobytes())
        candidates=[fallback,marginal]
        neural=None
        if self.prior is not None:
            grid=values.reshape(latent)
            anchors=anchor_mask(*latent).numpy()
            encoder=constriction.stream.queue.RangeEncoder()
            encoder.encode(grid[anchors].copy(),self.fixed)
            probabilities=self.prior.conditional_probabilities(grid)
            if (~anchors).any():encoder.encode(grid[~anchors].copy(),self.family,probabilities)
            neural=self.frame(1,0,shape,latent,side,checksum+encoder.get_compressed().astype('<u4').tobytes())
            candidates.append(neural)
        if force_method is None:
            chosen=min(candidates,key=lambda f:(len(f),f[5]))
        else:
            matches=[f for f in candidates if f[5]==force_method]
            if not matches:raise ValueError('Requested coder is unavailable')
            chosen=matches[0]
        nonneural=min((fallback,marginal),key=lambda f:(len(f),f[5]))
        return chosen,nonneural,{'old_frame_bytes':len(original),'marginal_bytes':len(marginal),
                                'neural_bytes':None if neural is None else len(neural),'chosen_method':chosen[5]}

    def decode(self,frame):
        if len(frame)<HEADER.size:raise ValueError('Truncated neural frame')
        fields=list(HEADER.unpack_from(frame))
        magic,version,method,old_method,reserved,h,w,lh,lw,side_size,identifier,crc=fields
        if (magic!=MAGIC or version!=1 or method not in (0,1,2) or reserved or old_method not in (0,1,2)
                or identifier!=self.identifier or min(h,w,lh,lw)<1 or lh*lw>32768
                or side_size>len(frame)-HEADER.size or (method!=0 and old_method)):
            raise ValueError('Invalid neural frame header')
        fields[-1]=0
        if zlib.crc32(HEADER.pack(*fields)+frame[HEADER.size:])!=crc:raise ValueError('Neural frame CRC mismatch')
        side=frame[HEADER.size:HEADER.size+side_size];payload=frame[HEADER.size+side_size:]
        if method==0:
            old_crc=zlib.crc32(payload,zlib.crc32(side))
            old_header=HEADER.pack(HUFFMAN_MAGIC,1,old_method,0,0,h,w,lh,lw,len(side),self.huffman.identifier,old_crc)
            return self.huffman.decode(old_header+side+payload)
        if len(payload)<8 or (len(payload)-INDEX_CRC.size)%4:raise ValueError('Malformed range stream')
        expected=INDEX_CRC.unpack_from(payload)[0]
        words=np.frombuffer(payload[INDEX_CRC.size:],dtype='<u4').astype(np.uint32,copy=True)
        decoder=constriction.stream.queue.RangeDecoder(words)
        if method==2:
            indices=decoder.decode(self.fixed,lh*lw).astype(np.uint16)
        else:
            if self.prior is None:raise ValueError('Shared neural prior required')
            anchors=anchor_mask(lh,lw).numpy()
            grid=np.full((lh,lw),2048,dtype=np.int32)
            grid[anchors]=decoder.decode(self.fixed,int(anchors.sum()))
            probabilities=self.prior.conditional_probabilities(grid)
            if (~anchors).any():grid[~anchors]=decoder.decode(self.family,probabilities)
            indices=grid.reshape(-1).astype(np.uint16)
        if zlib.crc32(indices.astype('<u2').tobytes())!=expected:
            raise ValueError('Index CRC mismatch; probability implementation or stream differs')
        return indices,(h,w),(lh,lw),side
