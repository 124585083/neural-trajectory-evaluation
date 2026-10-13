"""Independent checks of complete C files, including geometric baseline."""
import unittest
import numpy as np
from trajectory_project.closeout_v1.null_endpoint import ROOT,SOURCE,SEEDS,EPOCHS,METRICS
from trajectory_project.condition_endpoint_fa_gpfa_v1.condition_data import load_condition_data
from trajectory_project.step3_io import read_json


@unittest.skipUnless((ROOT/'artifacts/C/mahler_FA50_OLS_weights.npz').is_file(),
    'Full external C artifacts are not materialized here; use the read-only run.py --replay route.')
class CDeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=load_condition_data(SOURCE)
        cls.splits=read_json(ROOT/'configs/condition_splits_100.json')

    def test_all_random_baseline_denominators_train_only_and_same_between_representations(self):
        for animal,p in self.data.items():
            with np.load(ROOT/'artifacts/C'/f'{animal}_endpoint_assignments.npz') as a:
                endpoints=a['endpoints'];alpha=a['alpha'];mask=a['common_mask'];epochs=a['epoch_masks']
            expected=np.zeros((1000,100,len(EPOCHS)))
            train_means=np.empty((100,1000))
            for i,split in enumerate(self.splits):
                valid_train=np.asarray(split['train_indices']);valid_train=valid_train[mask[valid_train].any(1)]
                train_means[i]=endpoints[:,valid_train].mean(1)
                test=np.zeros(len(mask),bool);test[split['test_indices']]=True
                for e in range(len(EPOCHS)):
                    active=epochs[e]&test[:,None]
                    c=np.flatnonzero(active.any(1))
                    squared_alpha=np.nansum(np.where(active,alpha*alpha,0.),axis=1)
                    # Analytic candidate cancellation: anchors cancel from
                    # F(c,mu)-F(c,b), leaving alpha*(mu-b).
                    expected[:,i,e]=np.sum(squared_alpha[c][None,:]*(train_means[i,:,None]-endpoints[:,c])**2,axis=1)
            previous=None
            for representation in ('FA50','GPFA50'):
                stem=f'{animal}_{representation}'
                with np.load(ROOT/'artifacts/C'/f'{stem}_OLS_weights.npz') as w:
                    np.testing.assert_allclose(w['mean_train_endpoints'],train_means,atol=1e-14,rtol=1e-14)
                    self.assertEqual(w['coefficients'].shape,(100,1000,50))
                    self.assertEqual(w['x_coefficients'].shape,(100,50))
                with np.load(ROOT/'results/C'/f'{stem}_all_scores.npz') as s:
                    actual=s['random_scores'][...,METRICS.index('baseline_SSE')]
                    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-8)
                    if previous is not None:np.testing.assert_array_equal(actual,previous)
                    previous=actual

    def test_observed_geometric_baseline_uses_train_endpoints_only(self):
        for animal,p in self.data.items():
            with np.load(ROOT/'artifacts/C'/f'{animal}_endpoint_assignments.npz') as a:
                offset=a['offset'];alpha=a['alpha'];epochs=a['epoch_masks']
            endpoints=(p['geometry'].sort_values('condition_index').objective_end_y.to_numpy(float),p['mean_endpoint'])
            for representation in ('FA50','GPFA50'):
                with np.load(ROOT/'results/C'/f'{animal}_{representation}_all_scores.npz') as scores:
                    observed=scores['observed_scores']
                for i,split in enumerate(self.splits):
                    train=np.asarray(split['train_indices']);train=train[p['common_mask'][train].any(1)]
                    test=np.zeros(len(p['condition_ids']),bool);test[split['test_indices']]=True
                    for h,target in enumerate((p['objective_xy'],p['behavior_xy'])):
                        mu=endpoints[h][train].mean()
                        for e in range(len(EPOCHS)):
                            active=epochs[e]&test[:,None]
                            base=offset[active]+alpha[active]*mu
                            sse=np.sum((base-target[...,1][active])**2)
                            self.assertAlmostEqual(sse,observed[h,i,e,METRICS.index('baseline_SSE')],places=7)
                            neural_sse=observed[h,i,e,METRICS.index('SSE')]
                            expected_skill=1-neural_sse/sse if sse else np.nan
                            np.testing.assert_allclose(expected_skill,observed[h,i,e,METRICS.index('skill')],equal_nan=True)

    def test_example_predictions_are_heldout_only_and_x_cannot_change_with_q(self):
        for animal,p in self.data.items():
            for representation in ('FA50','GPFA50'):
                stem=f'{animal}_{representation}'
                with np.load(ROOT/'artifacts/C'/f'{stem}_fixed_examples.npz') as a:
                    pred=a['predictions'];supports=a['test_support']
                    self.assertEqual(pred.shape,(3,100,79,100))
                    for i,split in enumerate(self.splits):
                        self.assertFalse(supports[i,split['train_indices']].any())
                        self.assertFalse(np.isfinite(pred[:,i,split['train_indices']]).any())
                        np.testing.assert_array_equal(np.isfinite(pred[:,i]),np.broadcast_to(supports[i],pred[:,i].shape))
                with np.load(ROOT/'artifacts/C'/f'{stem}_OLS_weights.npz') as weights:
                    self.assertEqual(weights['x_coefficients'].shape,(100,50))
                    # There is exactly one shared x head per split: q cannot
                    # independently change x while y endpoints are reassigned.
                marker=read_json(ROOT/'results/C'/f'{stem}_completed.json')
                self.assertLess(marker['max_identical_x_prediction_error'],1e-8)
                self.assertEqual(marker['raw_preprocessing'],'fail')


if __name__=='__main__':unittest.main()
