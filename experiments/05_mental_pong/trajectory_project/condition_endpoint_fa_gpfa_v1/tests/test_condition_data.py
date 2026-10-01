import numpy as np
import pandas as pd
import pytest
from trajectory_project.condition_endpoint_fa_gpfa_v1.condition_data import mean_payload


def fixture():
    ids=np.array([10,20]);ci=np.array([0,0,0,1]);mask=np.array([[1,1,0],[1,1,0],[0,0,0],[0,0,0]],bool)
    obj=np.array([[[.1,1.],[.2,2.],[.3,3.]],[[.1,2.],[.2,3.],[.3,4.]]])
    beh=np.full((4,3,2),np.nan)
    beh[0,:2]=[[.1,2],[.2,4]];beh[1,:2]=[[.1,4],[.2,8]]
    return {'animal':'test','condition_ids':ids,'times_ms':np.array([50,100,150]),'condition_index':ci,
        'objective_xy':obj,'behavior_xy':beh,'common_mask':mask,'endpoint':np.array([4.,8.,np.nan,5.]),
        'trial_table':pd.DataFrame({'endpoint_valid':[True,True,False,True]}),
        'epoch_condition_masks':{e:np.ones((2,3),bool) for e in ('full','visible','hidden','bounce','no_bounce','post_bounce')},
        'geometry':pd.DataFrame()}


def test_mean_uses_exact_valid_trial_pool_and_preserves_unknown():
    a=mean_payload(fixture())
    np.testing.assert_array_equal(a['n_behavior_trials'],[2,0])
    assert a['source_trial_indices_by_condition']=={10:[0,1],20:[]}
    np.testing.assert_array_equal(a['behavior_xy'][0,:2,1],[3,6])
    assert a['mean_endpoint'][0]==6 and np.isnan(a['mean_endpoint'][1])
    assert np.isnan(a['behavior_xy'][1]).all()
    assert a['common_mask'].sum()==2


def test_changing_trial_membership_across_bins_rejected():
    p=fixture();p['common_mask'][1,0]=False
    with pytest.raises(ValueError,match='support varies'):mean_payload(p)


def test_shared_x_is_copied_exactly():
    p=fixture();a=mean_payload(p)
    np.testing.assert_array_equal(a['behavior_xy'][a['common_mask'],0],p['objective_xy'][a['common_mask'],0])
