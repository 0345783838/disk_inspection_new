"""Coverage planner respects indivisible target groups and train-only AI roots."""
import unittest
from training.plan_classifier_coverage_split import solve_split,coverage,cell

class CoverageSplitTests(unittest.TestCase):
    def test_coverage_holes_fixed_with_70_30_and_forced_root(self):
        rows=[];clusters={}
        for region in ['top','bottom']:
            for label in ['ng','no_disk','ok']:
                for group in range(10):
                    r=dict(id=f'{region}_{label}_{group}',origin='new_factory',camera=1,region=region,label=label,split_group=f'{region}_{group}')
                    rows.append(r);clusters[r['id']]=cell(r)+'|visual0'
        assignments,info=solve_split(rows,clusters,{'bottom_0'},time_limit=10)
        self.assertEqual(assignments['bottom_0'],'train')
        result=coverage(rows,clusters,assignments)
        self.assertTrue(all(v['both'] for v in result.values()))
        self.assertEqual(result['all']['val'],18)
        self.assertFalse(info['used_predictions'])

if __name__=='__main__':unittest.main()
