"""Static repeat-flag/literal entropy layer for immutable neural indices."""
import hashlib
import json
import struct
import zlib

import constriction
import numpy as np

from compression.symbol_stream import HEADER,MAGIC as HUFFMAN_MAGIC

MAGIC=b'FTR1'
INDEX_CRC=struct.Struct('<I')


def flags_and_literals(grid,order):
    view=grid if order=='temporal' else grid.T
    repeated=np.zeros(view.shape,dtype=bool)
    repeated[:,1:]=view[:,1:]==view[:,:-1]
    return repeated[:,1:].reshape(-1).astype(np.int32),view[~repeated].astype(np.int32)


class TransitionSymbolCodec:
    def __init__(self,huffman,statistics):
        self.huffman=huffman;self.statistics=statistics
        self.spec={'kind':'static-repeat-literal-range-v1','huffman_table_sha256':huffman.identifier.hex(),
                   'statistics':statistics,'backend':'constriction-0.5.0-perfect-false'}
        self.serialized=json.dumps(self.spec,sort_keys=True,separators=(',',':')).encode()
        self.identifier=hashlib.sha256(self.serialized).digest()
        counts=np.asarray(statistics['marginal_counts'],dtype=np.float64)
        if counts.shape!=(2048,) or not np.isfinite(counts).all() or np.any(counts<0):raise ValueError('Invalid counts')
        self.marginal=constriction.stream.model.Categorical(counts+1,perfect=False)
        self.orders={}
        for order,values in statistics.get('orders',{}).items():
            if order not in ('temporal','spatial'):raise ValueError('Unknown repeat direction')
            repeats,transitions=int(values['repeats']),int(values['transitions'])
            if repeats<0 or repeats>transitions:raise ValueError('Invalid transition counts')
            literal=np.asarray(values['literal_counts'],dtype=np.float64)
            if literal.shape!=(2048,) or not np.isfinite(literal).all() or np.any(literal<0):raise ValueError('Invalid literal counts')
            flag=constriction.stream.model.Bernoulli((repeats+1)/(transitions+2),perfect=False)
            self.orders[order]=(flag,constriction.stream.model.Categorical(literal+1,perfect=False))

    @classmethod
    def from_bytes(cls,huffman,data):
        values=json.loads(data)
        result=cls(huffman,values['statistics'])
        if result.serialized!=data:raise ValueError('Transition specification differs from reference format')
        return result

    def frame(self,method,old_method,shape,latent,side,payload):
        header=HEADER.pack(MAGIC,1,method,old_method,0,*shape,*latent,len(side),self.identifier,0)
        return header[:-4]+struct.pack('<I',zlib.crc32(header+side+payload))+side+payload

    def encode(self,original,force_method=None):
        indices,shape,latent,side=self.huffman.decode(original)
        fields=HEADER.unpack_from(original);checksum=INDEX_CRC.pack(zlib.crc32(indices.astype('<u2').tobytes()))
        candidates=[self.frame(0,fields[2],shape,latent,side,original[HEADER.size+len(side):])]
        encoder=constriction.stream.queue.RangeEncoder();encoder.encode(indices.astype(np.int32),self.marginal)
        candidates.append(self.frame(3,0,shape,latent,side,checksum+encoder.get_compressed().astype('<u4').tobytes()))
        for order,method in (('temporal',1),('spatial',2)):
            if order not in self.orders:continue
            flag_model,literal_model=self.orders[order]
            flags,literals=flags_and_literals(indices.reshape(latent),order)
            encoder=constriction.stream.queue.RangeEncoder()
            if flags.size:encoder.encode(flags,flag_model)
            encoder.encode(literals,literal_model)
            candidates.append(self.frame(method,0,shape,latent,side,checksum+encoder.get_compressed().astype('<u4').tobytes()))
        if force_method is not None:
            found=[f for f in candidates if f[5]==force_method]
            if not found:raise ValueError('Requested repeat coder unavailable')
            return found[0]
        return min(candidates,key=lambda f:(len(f),f[5]))

    def decode(self,frame):
        if len(frame)<HEADER.size:raise ValueError('Truncated transition frame')
        fields=list(HEADER.unpack_from(frame))
        magic,version,method,old_method,reserved,h,w,lh,lw,side_size,identifier,crc=fields
        if (magic!=MAGIC or version!=1 or method not in (0,1,2,3) or old_method not in (0,1,2) or reserved
                or identifier!=self.identifier or min(h,w,lh,lw)<1 or lh*lw>32768
                or side_size>len(frame)-HEADER.size or (method!=0 and old_method)):
            raise ValueError('Invalid transition header')
        fields[-1]=0
        if zlib.crc32(HEADER.pack(*fields)+frame[HEADER.size:])!=crc:raise ValueError('Transition CRC mismatch')
        side=frame[HEADER.size:HEADER.size+side_size];payload=frame[HEADER.size+side_size:]
        if method==0:
            header=HEADER.pack(HUFFMAN_MAGIC,1,old_method,0,0,h,w,lh,lw,len(side),self.huffman.identifier,
                               zlib.crc32(payload,zlib.crc32(side)))
            return self.huffman.decode(header+side+payload)
        if len(payload)<8 or (len(payload)-4)%4:raise ValueError('Malformed transition range payload')
        checksum=INDEX_CRC.unpack_from(payload)[0]
        words=np.frombuffer(payload[4:],dtype='<u4').astype(np.uint32,copy=True)
        decoder=constriction.stream.queue.RangeDecoder(words)
        if method==3:indices=decoder.decode(self.marginal,lh*lw).astype(np.uint16)
        else:
            order='temporal' if method==1 else 'spatial'
            if order not in self.orders:raise ValueError('Required repeat model missing')
            flag_model,literal_model=self.orders[order]
            shape=(lh,lw) if order=='temporal' else (lw,lh)
            repeated=np.zeros(shape,dtype=bool)
            count=shape[0]*(shape[1]-1)
            if count:repeated[:,1:]=decoder.decode(flag_model,count).reshape(shape[0],shape[1]-1).astype(bool)
            literal_mask=~repeated.reshape(-1)
            values=np.zeros(lh*lw,dtype=np.int32)
            values[literal_mask]=decoder.decode(literal_model,int(literal_mask.sum()))
            positions=np.maximum.accumulate(np.where(literal_mask,np.arange(lh*lw),0))
            view=values[positions].reshape(shape)
            indices=(view if order=='temporal' else view.T).reshape(-1).astype(np.uint16)
        if zlib.crc32(indices.astype('<u2').tobytes())!=checksum:raise ValueError('Decoded transition indices differ')
        return indices,(h,w),(lh,lw),side
