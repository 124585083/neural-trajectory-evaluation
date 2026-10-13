"""Small mathematical checks; production uses the frozen real source splits."""
import unittest
import numpy as np
from sklearn.linear_model import LinearRegression
from trajectory_project.closeout_v1.null_endpoint import fit_batch, score_batch, make_assignments, _indices, METRICS


class EndpointNullTests(unittest.TestCase):
    def test_batched_OLS_is_independent_OLS(self):
        rng=np.random.default_rng(23)
        x=rng.normal(size=(200,12)); y=rng.normal(size=(200,8))
        x[:,11]=x[:,0]+x[:,1]  # same rank-deficient least-squares rule
        model=fit_batch(x,y)
        for j in range(8):
            solo=LinearRegression(fit_intercept=True).fit(x,y[:,j])
            np.testing.assert_allclose(model.predict(x)[:,j],solo.predict(x),atol=1e-10)

    def test_identical_y_and_x_targets_are_identical(self):
        rng=np.random.default_rng(123);x=rng.normal(size=(120,8)); y=rng.normal(size=120)
        model=fit_batch(x,np.column_stack([y,y,y]))
        np.testing.assert_allclose(model.coef_[0],model.coef_[1],atol=1e-12)
        np.testing.assert_allclose(model.predict(x)[:,1],model.predict(x)[:,2],atol=1e-12)

    def test_endpoint_marginal_missing_condition_and_fixed_seed(self):
        p={'common_mask':np.array([[1,1],[0,0],[1,1],[1,1]],bool),
           'mean_endpoint':np.array([-2.,np.nan,0.,3.]),'condition_ids':np.arange(4)}
        ix,maps,endpoints=make_assignments(p,77,repeats=100)
        a,b,c=make_assignments(p,77,repeats=100)
        np.testing.assert_array_equal(maps,b)
        np.testing.assert_array_equal(endpoints,c)
        self.assertTrue(np.isnan(endpoints[:,1]).all())
        np.testing.assert_array_equal(np.sort(endpoints[:,ix],axis=1),np.tile([-2.,0.,3.],(100,1)))

    def test_mask_retains_39_40_identities_and_does_not_fill_invalid(self):
        mask=np.ones((79,5),bool);mask[0]=False
        split={'train_indices':list(range(39)),'test_indices':list(range(39,79))}
        train,test=_indices(mask,split)
        self.assertEqual(len(np.unique(train[0])),38)
        self.assertEqual(len(np.unique(test[0])),40)
        self.assertFalse(set(train[0])&set(test[0]))

    def test_score_own_random_label_not_actual_target(self):
        target=np.arange(8,dtype=float)[:,None]
        prediction=target.copy();base=np.zeros_like(target);objective=-target[:,0]
        values=score_batch(target,prediction,base,objective)[0]
        self.assertAlmostEqual(values[METRICS.index('r')],1.)
        self.assertEqual(values[METRICS.index('RMSE')],0.)
        self.assertEqual(values[METRICS.index('skill')],1.)
        self.assertGreater(values[METRICS.index('objective_distance_RMSE')],0.)

    def test_zero_baseline_sse_and_constant_correlation_are_na(self):
        target=np.ones((6,2));prediction=np.zeros_like(target)
        values=score_batch(target,prediction,target,target[:,0])
        self.assertTrue(np.isnan(values[:,METRICS.index('r')]).all())
        self.assertTrue(np.isnan(values[:,METRICS.index('skill')]).all())
        np.testing.assert_array_equal(values[:,METRICS.index('RMSE')],[1,1])

    def test_correlation_and_RMSE_match_direct_scalar_formula(self):
        rng=np.random.default_rng(9);target=rng.normal(size=(91,4));prediction=rng.normal(size=(91,4));base=target*.2
        values=score_batch(target,prediction,base,target[:,0])
        for j in range(4):
            self.assertAlmostEqual(values[j,0],np.corrcoef(target[:,j],prediction[:,j])[0,1])
            self.assertAlmostEqual(values[j,1],np.sqrt(np.mean((target[:,j]-prediction[:,j])**2)))


if __name__=='__main__':unittest.main()
