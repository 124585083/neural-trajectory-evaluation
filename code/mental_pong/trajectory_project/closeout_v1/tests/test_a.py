import numpy as np
from trajectory_project.closeout_v1.replay_a import score


def test_self_and_cross_cannot_be_interchanged():
    objective=np.array([-3.,-1.,1.,3.])
    behavior=np.array([-2.,0.,2.,4.])
    assert score(objective,objective)['RMSE']==0
    assert score(behavior,behavior)['RMSE']==0
    assert score(behavior,objective)['RMSE']==1
    assert score(objective,behavior)['RMSE']==1
    assert score(behavior,objective)['bias']==-1
    assert score(objective,behavior)['bias']==1


def test_undefined_correlation_is_not_zero():
    assert np.isnan(score(np.ones(5),np.arange(5.))['r'])
    assert np.isnan(score(np.arange(2.),np.arange(2.))['r'])
    assert score(np.ones(5),np.arange(5.))['RMSE']>0
    assert np.isnan(score([],[])['RMSE'])


def test_centered_correlation_and_error_definition():
    y=np.array([-3.,-1.,0.,2.,6.])
    p=np.array([-2.,0.,-1.,3.,5.])
    result=score(y,p)
    np.testing.assert_allclose(result['r'],np.corrcoef(y,p)[0,1],atol=1e-15)
    np.testing.assert_allclose(result['RMSE'],np.linalg.norm(y-p)/np.sqrt(len(y)))
