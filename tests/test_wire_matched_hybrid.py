import unittest
from finetuning.run_wire_matched_hybrid import allocate_heap
from finetuning.compare_refinement_modes import allocate


class HeapAllocationTests(unittest.TestCase):
    def test_heap_agrees_with_greedy_reference(self):
        rows=[]
        for i in range(8):
            def mode(size,error):return {'bytes':size,'wire_bytes':size,
                 'metrics':{'relative_rmse':error,'correlation':1-error/2,'ordinary_channel_p90_relative_rmse':error}}
            rows.append({'sample_id':str(i),'drive_id':'file','cells':1000,
                'base':mode(100,1.),'dense':mode(200+i,.5),'residual':mode(120+i,.8)})
        for factor in (1.,1.1,1.25,1.5,2.):
            a,_=allocate(rows,['dense','residual'],factor)
            b=allocate_heap(rows,{'file':int(800*factor)})
            self.assertEqual(a,b)

    def test_independent_file_caps(self):
        row={'sample_id':'x','drive_id':'file','cells':1000,
             'base':{'wire_bytes':100,'metrics':{'relative_rmse':1.,'correlation':.5,'ordinary_channel_p90_relative_rmse':1.}},
             'dense':{'wire_bytes':200,'metrics':{'relative_rmse':.5,'correlation':.8,'ordinary_channel_p90_relative_rmse':.5}},
             'residual':{'wire_bytes':120,'metrics':{'relative_rmse':.8,'correlation':.6,'ordinary_channel_p90_relative_rmse':.8}}}
        self.assertEqual(allocate_heap([row],{'file':100})['x'],'base')


if __name__=='__main__':unittest.main()
