import unittest
from finetuning.compare_refinement_modes import allocate


def row(name):
    def mode(cost,error,corr,tail):
        return {'bytes':cost,'metrics':{'relative_rmse':error,'correlation':corr,'ordinary_channel_p90_relative_rmse':tail}}
    return {'sample_id':name,'cells':1000,'base':mode(100,1.,.5,1.),
            'residual':mode(120,.8,.7,.8),'dense':mode(200,.5,.9,.5)}


class RefinementModeTests(unittest.TestCase):
    def test_nested_upgrade_and_byte_ceiling(self):
        records=[row('x'),row('y')]
        low,_=allocate(records,['dense','residual'],1.25)
        high,budget=allocate(records,['dense','residual'],2.,low)
        for value in records:
            self.assertLessEqual(value[high[value['sample_id']]]['metrics']['relative_rmse'],
                                 value[low[value['sample_id']]]['metrics']['relative_rmse'])
        self.assertLessEqual(budget['frame_bytes'],budget['ceiling_bytes'])

    def test_tail_regression_blocks_upgrade(self):
        records=[row('x')]
        records[0]['dense']['metrics']['ordinary_channel_p90_relative_rmse']=2.
        selected,_=allocate(records,['dense','residual'],2.)
        self.assertEqual(selected['x'],'residual')


if __name__=='__main__':unittest.main()
