import unittest
import numpy as np

try:
    import constriction
except ImportError:
    constriction=None


@unittest.skipIf(constriction is None,'Optional entropy backend unavailable')
class TransitionTests(unittest.TestCase):
    def test_directions_and_reset_boundaries(self):
        from compression.symbol_stream import SymbolCodec
        from compression.transition_symbol_stream import TransitionSymbolCodec,flags_and_literals
        counts=np.ones(2048,dtype=np.int64)
        huffman=SymbolCodec.from_counts(counts,np.tile(counts,(17,1)),'a'*64)
        stats={'marginal_counts':counts.tolist(),'orders':{name:{'repeats':7,'transitions':20,
               'literal_counts':counts.tolist()} for name in ('temporal','spatial')}}
        codec=TransitionSymbolCodec(huffman,stats)
        for grid in (np.array([[1]]),np.array([[1,1,2],[2,2,3]]),np.tile(np.arange(7),(4,1)),
                     np.random.default_rng(42).integers(0,2048,(5,7))):
            original,_=huffman.encode(grid.reshape(-1),(512,1000),grid.shape,b'side')
            for method in (0,1,2,3):
                frame=codec.encode(original,force_method=method)
                recovered,_,shape,side=codec.decode(frame)
                np.testing.assert_array_equal(recovered,grid.reshape(-1))
                self.assertEqual(shape,grid.shape);self.assertEqual(side,b'side')
            damaged=bytearray(codec.encode(original));damaged[-1]^=1
            with self.assertRaises(ValueError):codec.decode(bytes(damaged))

    def test_minimum_payload_and_control(self):
        from compression.symbol_stream import SymbolCodec
        from compression.transition_symbol_stream import TransitionSymbolCodec
        counts=np.ones(2048,dtype=np.int64)
        huffman=SymbolCodec.from_counts(counts,np.tile(counts,(17,1)),'b'*64)
        codec=TransitionSymbolCodec(huffman,{'marginal_counts':counts.tolist(),'orders':{}})
        original,_=huffman.encode(np.arange(128),(32,1000),(4,32),b'')
        self.assertLessEqual(len(codec.encode(original)),len(original))


if __name__=='__main__':unittest.main()
