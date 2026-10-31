import unittest
import numpy as np
import torch

from finetuning.checkerboard_prior import CheckerboardPrior,anchor_mask


class CheckerboardPriorTests(unittest.TestCase):
    def test_nonanchors_cannot_leak_into_predictions(self):
        model=CheckerboardPrior(np.ones(2048)).eval()
        torch.nn.init.normal_(model.head.weight,std=.01)
        x=torch.randint(0,2048,(1,5,7));other=x.clone()
        other[:,~anchor_mask(5,7)]=torch.randint(0,2048,(int((~anchor_mask(5,7)).sum()),))
        with torch.no_grad():torch.testing.assert_close(model(x),model(other),atol=0,rtol=0)

    def test_reference_cpu_probabilities_reproduce(self):
        model=CheckerboardPrior(np.ones(2048)).eval()
        x=np.arange(35).reshape(5,7).astype(np.int32)
        a=model.conditional_probabilities(x)
        np.testing.assert_array_equal(a,model.conditional_probabilities(x))
        np.testing.assert_allclose(a.sum(axis=1),1.,atol=1e-6)


try:
    import constriction
except ImportError:
    constriction=None


@unittest.skipIf(constriction is None,'Optional requirements-entropy.txt not installed')
class NeuralStreamTests(unittest.TestCase):
    def test_range_and_fallback_recover_identical_indices(self):
        from compression.neural_symbol_stream import NeuralSymbolCodec
        from compression.symbol_stream import SymbolCodec
        counts=np.ones(2048,dtype=np.int64);counts[7]=10000
        huffman=SymbolCodec.from_counts(counts,np.tile(counts,(17,1)),'a'*64)
        prior=CheckerboardPrior(counts/counts.sum()).eval()
        torch.nn.init.normal_(prior.head.weight,std=.01)
        descriptor={'huffman_table_sha256':huffman.identifier.hex(),'software':'test'}
        codec=NeuralSymbolCodec(huffman,prior.marginal.numpy(),descriptor,prior)
        for shape in ((1,1),(3,7),(4,8)):
            symbols=np.full(np.prod(shape),7,dtype=np.uint16)
            if np.prod(shape)>1:symbols[-1]=symbols[1]=2047
            original,_=huffman.encode(symbols,(10,1000),shape,b'normalization')
            chosen,nonneural,_=codec.encode(original)
            forced=[codec.encode(original,force_method=method)[0] for method in (0,1,2)]
            for frame in (chosen,nonneural,*forced):
                recovered,_,_,side=codec.decode(frame)
                np.testing.assert_array_equal(recovered,symbols);self.assertEqual(side,b'normalization')
            control=NeuralSymbolCodec(huffman,prior.marginal.numpy(),descriptor,None)
            np.testing.assert_array_equal(control.decode(nonneural)[0],symbols)
            with self.assertRaises(ValueError):control.decode(forced[1])
            damaged=bytearray(chosen);damaged[8]^=1
            with self.assertRaises(ValueError):codec.decode(bytes(damaged))


if __name__=='__main__':unittest.main()
