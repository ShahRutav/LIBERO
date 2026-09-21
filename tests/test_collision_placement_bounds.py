"""Compiled-geometry placement logic without simulator imports or model compiles."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

PATH = Path(__file__).parents[1] / 'libero/libero/envs/regions/collision_bounds.py'
spec = importlib.util.spec_from_file_location('collision_bounds_pure', PATH)
cb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cb)
IDENTITY = [1., 0., 0., 0.]


def fake_sim(kind=6, size=(.1,.2,.3), *, local=(.02,.03,.04), geom_rotation=None,
             old_position=(.5,-.4,.9), old_quaternion=IDENTITY, child=True):
    R = cb._rotation(old_quaternion)
    G = np.eye(3) if geom_rotation is None else geom_rotation
    pos = np.asarray(old_position)
    model = NS(ngeom=2, geom_type=np.array([kind,6]), geom_size=np.array([size,(9,9,9)]),
               geom_dataid=np.array([0,-1]), mesh_vertadr=np.array([0]), mesh_vertnum=np.array([3]),
               mesh_vert=np.array([[-.03,-.04,-.05],[.02,.06,.08],[.07,.01,.04]]),
               geom_contype=np.array([1,0]), geom_conaffinity=np.array([1,0]),
               geom_bodyid=np.array([2 if child else 1,1]), body_parentid=np.array([0,0,1]))
    data = NS(xpos=np.array([[0,0,0],pos,pos]), xmat=np.tile(R.reshape(1,9),(3,1)),
              geom_xpos=np.array([pos+R@np.asarray(local),pos]),
              geom_xmat=np.array([(R@G).reshape(9),R.reshape(9)]))
    return NS(model=NS(_model=model,body_name2id=lambda _:1),data=NS(_data=data))


@pytest.mark.parametrize('old_position,old_quaternion', [((0,0,0),IDENTITY), ((1,-2,3),[.5,.5,.5,.5])])
@pytest.mark.parametrize('new_quaternion', [IDENTITY,[np.sqrt(.5),0,np.sqrt(.5),0], [.5,.5,.5,.5]])
def test_rotated_box_child_is_independent_of_previous_root(old_position,old_quaternion,new_quaternion):
    G=cb._rotation([np.sqrt(.5),np.sqrt(.5),0,0])
    sim=fake_sim(old_position=old_position,old_quaternion=old_quaternion,geom_rotation=G)
    obj=NS(name='object',root_body='object_root')
    low,high,ids=cb.object_vertical_bounds(sim,obj,new_quaternion)
    R=cb._rotation(new_quaternion)
    corners=np.array([[x,y,z] for x in (-.1,.1) for y in (-.2,.2) for z in (-.3,.3)])
    expected=(corners@G.T+np.array([.02,.03,.04]))@R.T
    assert (low,high)==pytest.approx((expected[:,2].min(),expected[:,2].max()))
    assert ids==[0]  # visual 9m box must not affect support


def test_compiled_mesh_uses_geom_frame_once():
    G=cb._rotation([.5,.5,.5,.5]); q=[np.sqrt(.5),0,np.sqrt(.5),0]
    sim=fake_sim(7,geom_rotation=G,old_quaternion=[.5,.5,.5,.5])
    # Compiled vertices already carry scale; irrelevant original mesh fields
    # must not be reapplied to these vertices.
    sim.model._model.mesh_scale=np.array([[100,100,100]])
    sim.model._model.mesh_pos=np.array([[100,100,100]])
    actual=cb.object_vertical_bounds(sim,NS(name='mesh',root_body='root'),q)
    expected=(sim.model._model.mesh_vert@G.T+np.array([.02,.03,.04]))@cb._rotation(q).T
    assert actual[:2]==pytest.approx((expected[:,2].min(),expected[:,2].max()))


@pytest.mark.parametrize('kind,size,rotation,extent', [
    (2,(.2,0,0),np.eye(3),.2),
    (3,(.2,.3,0),np.eye(3),.5),
    (3,(.2,.3,0),cb._rotation([np.sqrt(.5),0,np.sqrt(.5),0]),.2),
    (4,(.2,.3,.4),np.eye(3),.4),
    (5,(.2,.3,0),np.eye(3),.3),
    (5,(.2,.3,0),cb._rotation([np.sqrt(.5),0,np.sqrt(.5),0]),.2),
])
def test_primitive_support(kind,size,rotation,extent):
    model=NS(geom_type=[kind],geom_size=[size])
    assert cb._geom_z_bounds(model,0,np.array([0.,0.,1.]),rotation)==pytest.approx((1-extent,1+extent))


def test_nested_supports_toposorted_preserve_xy_quaternion_rng_and_inputs(monkeypatch):
    objects={name:NS(name=name,root_body=name) for name in ('bottom','top')}
    placements={name:((.2,.3,9.),np.array(IDENTITY),obj) for name,obj in objects.items()}
    monkeypatch.setattr(cb,'object_vertical_bounds',lambda sim,obj,q:(-.02,.03,[0]) if obj.name=='bottom' else (-.01,.02,[1]))
    before=np.random.get_state()
    result,report=cb.correct_on_placements(None,placements,
        [('on','top','bottom'),('on','bottom','table_region')],
        {'table_region':{'target':'table'}},objects,[],.9)
    assert result['bottom'][0]==pytest.approx((.2,.3,.921))
    assert result['top'][0]==pytest.approx((.2,.3,.962))
    for name in objects:
        np.testing.assert_array_equal(result[name][1],placements[name][1])
        assert placements[name][0][2]==9
    after=np.random.get_state()
    assert before[0]==after[0] and before[2:]==after[2:]
    np.testing.assert_array_equal(before[1],after[1])
    assert [r['object'] for r in report['adjustments']]==['bottom','top']
    assert report['clearance_m']==.001
    json.dumps(report,allow_nan=False)


def test_site_and_in_are_explicitly_unchanged():
    obj=NS(name='bowl',root_body='bowl'); placement=((0,0,.9),IDENTITY,obj)
    for predicate in ('on','in'):
        result,report=cb.correct_on_placements(None,{'bowl':placement},[(predicate,'bowl','drawer_site')],
            {'drawer_site':{'target':'cabinet'}},['bowl'],['cabinet'],.9)
        assert result['bowl'] is placement
        assert not report['adjustments'] and len(report['ignored_relations'])==1


def test_cycles_missing_support_and_unsupported_geom_fail(monkeypatch):
    monkeypatch.setattr(cb,'object_vertical_bounds',lambda *a:(0,.1,[0]))
    p={name:((0,0,0),IDENTITY,NS(name=name)) for name in ('a','b')}
    with pytest.raises(ValueError,match='Cyclic'):
        cb.correct_on_placements(None,p,[('on','a','b'),('on','b','a')],{},p,[],0)
    with pytest.raises(ValueError,match='Missing sampled'):
        cb.correct_on_placements(None,{'a':p['a']},[('on','a','b')],{},p,[],0)
    with pytest.raises(ValueError,match='Unsupported On'):
        cb.correct_on_placements(None,p,[('on','a','unknown')],{},p,[],0)
    with pytest.raises(ValueError,match='Unsupported collision'):
        cb._geom_z_bounds(NS(geom_type=[1],geom_size=[[1,1,1]]),0,np.zeros(3),np.eye(3))


@pytest.mark.parametrize('mode,clearance',[('bad',.001),('collision_bounds',-1),('legacy',float('nan')),('legacy',True)])
def test_invalid_policy(mode,clearance):
    with pytest.raises(ValueError):cb.validate_height_policy(mode,clearance)


def test_mixed_site_child_of_corrected_support_fails():
    with pytest.raises(ValueError,match='Site/In child'):
        cb.correct_on_placements(None,{},[('on','base','table_region'),('in','child','base_site')],
            {'table_region':{'target':'table'},'base_site':{'target':'base'}},['base','child'],[],.9)


def test_explicit_pair_activates_zero_mask_collision_geom():
    sim=fake_sim()
    model=sim.model._model
    model.geom_contype[0]=model.geom_conaffinity[0]=0
    model.pair_geom1=np.array([0]); model.pair_geom2=np.array([0])
    low,high,ids=cb.object_vertical_bounds(sim,NS(name='object',root_body='root'),IDENTITY)
    assert ids==[0] and (low,high)==pytest.approx((-.26,.34))


def test_workspace_top_uses_world_rotation_and_ignores_visuals():
    sim = fake_sim(old_position=(0, 0, .41), old_quaternion=[.5, .5, .5, .5])
    sim.data._data.xquat = np.tile([.5, .5, .5, .5], (3, 1))
    top, geoms = cb.workspace_collision_top(sim, 'table')
    # Rotation maps local y onto world z: center .03 + half-width .2.
    assert top == pytest.approx(.41 + .03 + .2)
    assert geoms == [0]


def test_workspace_geometry_replaces_nominal_height(monkeypatch):
    obj = NS(name='box', root_body='box')
    monkeypatch.setattr(cb, 'workspace_collision_top', lambda *a: (.43676, [10]))
    monkeypatch.setattr(cb, 'object_vertical_bounds', lambda *a: (-.01, .02, [20]))
    corrected, report = cb.correct_on_placements(
        None, {'box': ((.1, .2, .48), IDENTITY, obj)}, [('on', 'box', 'region')],
        {'region': {'target': 'living_room_table'}}, ['box'], [], .41,
        clearance=.004, workspace_body='living_room_table')
    assert corrected['box'][0] == pytest.approx((.1, .2, .45076))
    assert report['workspace']['nominal_z'] == .41
    assert report['workspace']['collision_top_z'] == .43676
