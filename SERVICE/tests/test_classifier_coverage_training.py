"""Every training image is visited; balanced extras use only real train strata."""
import unittest
from collections import Counter
from training.train_classifier_coverage import coverage_weights,CompleteCoverageSampler
from training.train_classifier_canonical_holdout import BALANCED_WEIGHTS,LABELS

class CoverageTrainingTests(unittest.TestCase):
    def fixture(self):
        rows=[];visual={}
        for origin in BALANCED_WEIGHTS:
            for camera in ((1,2) if origin!='old_legacy_train' else (None,)):
                for label in LABELS:
                    for region in (['top','bottom'] if camera else ['unknown']):
                        for cluster in range(2):
                            for group in range(2):
                                for view in range(group+1):
                                    identifier=f'{origin}/{camera}/{label}/{region}/{cluster}/{group}/{view}'
                                    rows.append(dict(id=identifier,origin=origin,camera=camera,label=label,region=region,split='train',split_group=f'{origin}/{camera}/{region}/{group}'))
                                    visual[identifier]=f'{origin}/{camera}/{label}/{region}/cluster{cluster}'
        for origin,camera in [('old_ha',2),('new_factory',1),('new_factory',2)]:
            rows.append(dict(id=f'ai_{origin}_{camera}',origin=origin,camera=camera,label='ng',split='train',region='bottom',split_group='root',is_synthetic=True))
        return rows,visual

    def test_extra_budget_region_balance_and_full_epoch_visit(self):
        rows,visual=self.fixture();weights=coverage_weights(rows,visual)
        self.assertAlmostEqual(sum(weights),1.,places=10)
        self.assertAlmostEqual(sum(w for r,w in zip(rows,weights) if r.get('is_synthetic')),.027,places=10)
        for origin,budget in BALANCED_WEIGHTS.items():self.assertAlmostEqual(sum(w for r,w in zip(rows,weights) if r['origin']==origin),budget,places=10)
        for region in ['top','bottom']:
            mass=sum(w for r,w in zip(rows,weights) if r['origin']=='new_factory' and r['camera']==1 and r['label']=='ng' and not r.get('is_synthetic') and r['region']==region)
            self.assertAlmostEqual(mass,.4*.5*.3*.85*.5,places=10)
        sampler=CompleteCoverageSampler(weights,1030)
        first=list(sampler);second=list(sampler)
        self.assertEqual(len(first),2*len(rows));self.assertEqual(set(first),set(range(len(rows))))
        self.assertEqual(set(second),set(range(len(rows))))
        self.assertEqual(first,list(CompleteCoverageSampler(weights,1030)))
        self.assertNotEqual(first,second)

    def test_validation_and_unavailable_features_rejected(self):
        rows,visual=self.fixture();rows[0]['split']='val'
        with self.assertRaises(ValueError):coverage_weights(rows,visual)
        rows,visual=self.fixture();visual.pop(rows[0]['id'])
        with self.assertRaises(KeyError):coverage_weights(rows,visual)

if __name__=='__main__':unittest.main()
