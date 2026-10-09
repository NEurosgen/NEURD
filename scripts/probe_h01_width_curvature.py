"""Bounded diagnostic of smoothed shaft-width second derivatives in reviewed H01.

No segmentation changes or learned thresholds. Width is the equivalent radius
sqrt(section area/pi) along a local shaft frame, not the second derivative of
the skeleton coordinates. Missing contours remain missing; derivatives never
bridge gaps. Entire original parent surfaces supply sections. Measurements are
local approximations and may be distorted by bending of the shaft or branches.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.ndimage import gaussian_filter1d
import trimesh

os.environ.setdefault('MPLCONFIGDIR', '/tmp/neurd-spine-refinement-mpl')
MAIN = Path('/home/eugen/Desktop/CodeWork/Projects/NEURD_modif/NEURD')
STEP = 100.0
SMOOTHING = (100.0, 200.0)


def load_refinement():
    path = Path(__file__).parents[1]/'neurd/spine_refinement.py'
    spec = importlib.util.spec_from_file_location('_curvature_probe_refinement', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def masked_derivatives(radius, smoothing_nm):
    """Physical-units derivatives on finite contiguous runs, excluding margins."""
    radius = np.asarray(radius, float)
    smooth = np.full(len(radius), np.nan)
    second = np.full(len(radius), np.nan)
    normalized = np.full(len(radius), np.nan)
    finite = np.flatnonzero(np.isfinite(radius))
    chunks = np.split(finite, np.flatnonzero(np.diff(finite)>1)+1)
    margin = int(np.ceil(3*smoothing_nm/STEP))+2
    for ids in chunks:
        if len(ids) < 2*margin+3:
            continue
        value = gaussian_filter1d(radius[ids], smoothing_nm/STEP, mode='reflect', truncate=3)
        first = np.gradient(value, STEP, edge_order=2)
        derivative = np.gradient(first, STEP, edge_order=2)
        inner = ids[margin:-margin]
        smooth[inner] = value[margin:-margin]
        second[inner] = derivative[margin:-margin]
        normalized[inner] = (-value*derivative/(1+first**2)**1.5)[margin:-margin]
    return smooth, second, normalized


def analytic_checks():
    s = np.arange(-3000, 3001, STEP)
    _, d2, strength = masked_derivatives(np.full(len(s), 500.0), 100)
    assert np.nanmax(abs(d2)) < 1e-12 and np.nanmax(abs(strength)) < 1e-9
    _, d2, _ = masked_derivatives(500+s*.04, 100)
    assert np.nanmax(abs(d2)) < 1e-12
    _, d2, strength = masked_derivatives(500+250*np.exp(-s*s/(2*300**2)), 100)
    assert d2[len(s)//2] < 0 and strength[len(s)//2] > 0
    broken = np.full(len(s), 500.0);broken[len(s)//2] = np.nan
    _, second, _ = masked_derivatives(broken, 100)
    assert np.isnan(second[len(s)//2-3:len(s)//2+4]).all()


def native(values):
    return [float(v) if np.isfinite(v) else None for v in values]


def profile(sr, parent, context, groups, raw_id, part_id):
    ids = next(g for r,p,g in groups if r==raw_id and p==part_id)
    boundary, _ = sr._surface_boundary(parent, ids)
    rim, weights, _ = boundary
    origin = np.average(rim, axis=0, weights=weights)
    _, distance = context.axis.project(rim)
    axis_distance = float(sr._quantile(distance, weights, .9))
    if axis_distance <= context.config.trim_maximum_axis_distance_nm:
        anchor = context.axis.project([origin])[0][0]
        basis = sr._frame(context.axis.tangent(anchor));direction = 'skeleton'
    else:
        anchor, basis = context.mesh_frame(origin);direction = 'mesh'
    along = (parent.triangles_center[ids]-anchor) @ basis[:,2]
    low = np.floor((along.min()-1500)/STEP)*STEP
    high = np.ceil((along.max()+1500)/STEP)*STEP
    s = np.arange(low, high+STEP/2, STEP)
    radius = np.hypot(3000, max(abs(low),abs(high)))
    nearby = np.array(context.face_tree.query_ball_point(anchor,radius+context.max_face_radius),int)
    context.unique_faces()
    nearby = nearby[context.canonical[nearby]==nearby]
    vertices, inverse = np.unique(parent.faces[nearby],return_inverse=True)
    points = (parent.vertices[vertices]-anchor) @ basis
    faces = inverse.reshape(-1,3)
    use = (np.linalg.norm(points[:,:2],axis=1)[faces] <= 3000).any(axis=1)
    faces = faces[use]
    support = ~context.excluded_support[nearby[use]]
    full_radius, shaft_radius, center_offset, contour_area = [], [], [], []
    for offset in s:
        polygons = sr._section_polygons(points,faces,np.zeros(3),np.eye(3),offset,support)
        if not polygons:
            full_radius.append(np.nan);shaft_radius.append(np.nan);center_offset.append(np.nan);contour_area.append(np.nan);continue
        polygon, area, supported = min(polygons,key=lambda p:max(0,sr._polygon_distance(np.zeros((1,2)),p[0])[0])+.001*np.linalg.norm(p[0].mean(axis=0)))
        # A constant local frame is not trusted after its axis leaves the shaft.
        outside = float(max(0,sr._polygon_distance(np.zeros((1,2)),polygon)[0]))
        if outside > 800:
            full_radius.append(np.nan);shaft_radius.append(np.nan);center_offset.append(outside);contour_area.append(np.nan);continue
        full_radius.append(np.sqrt(area/np.pi));center_offset.append(outside);contour_area.append(area)
        shaft = sr._shaft_contour((polygon,area,supported),context.config.trim_maximum_section_bridge_nm)
        shaft_radius.append(np.nan if shaft is None else np.sqrt(shaft[1]/np.pi))
    derivatives = {};metrics = {}
    near_candidate = (s>=along.min()-200)&(s<=along.max()+200)
    for name, values in [('parent',full_radius),('shaft_support',shaft_radius)]:
        for sigma in SMOOTHING:
            smooth, second, strength = masked_derivatives(values,sigma)
            key=f'{name}_sigma{int(sigma)}'
            derivatives[key]=dict(smoothed_radius_nm=native(smooth),second_derivative_per_nm=native(second),dimensionless_meridional_curvature=native(strength))
            active = near_candidate & np.isfinite(strength)
            metrics[key]=dict(valid_candidate_samples=int(active.sum()),
                              q95_abs_curvature=float(np.quantile(abs(strength[active]),.95)) if active.any() else None,
                              radius_range_nm=[float(np.nanmin(np.array(values)[near_candidate])),float(np.nanmax(np.array(values)[near_candidate]))] if np.isfinite(np.array(values)[near_candidate]).any() else None)
    return dict(status='measured',frame_direction=direction,axis_distance_q90_nm=axis_distance,
                candidate_interval_nm=[float(along.min()),float(along.max())],anchor_nm=anchor.tolist(),basis=basis.tolist(),
                along_nm=s.tolist(),parent_radius_nm=native(full_radius),shaft_support_radius_nm=native(shaft_radius),
                axis_outside_section_nm=native(center_offset),full_section_area_nm2=native(contour_area),
                derivatives=derivatives,metrics=metrics)


def plot_cases(out, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    by={r['case_id']:r for r in rows}
    ids=['C08','C12','C05','C19','C21','C03','C18','C09']
    fig,axes=plt.subplots(4,2,figsize=(13,13),layout='constrained')
    for ax,cid in zip(axes.flat,ids):
        row=by[cid];p=row['profile'];a=row['anatomy']
        ax.set_title(f'{cid} · human: {a}',fontsize=11)
        if p['status']!='measured':
            ax.text(.5,.5,p['reason'],transform=ax.transAxes,ha='center',wrap=True);ax.set_axis_off();continue
        x=np.array(p['along_nm'])/1000
        ax.plot(x,np.array([np.nan if v is None else v for v in p['parent_radius_nm']])/1000,color='#999',lw=1,label='Parent equivalent radius')
        smooth=p['derivatives']['parent_sigma100']['smoothed_radius_nm']
        ax.plot(x,np.array([np.nan if v is None else v for v in smooth])/1000,color='#007b83',lw=1.6,label='Smoothed radius (sigma 100 nm)')
        ax.axvspan(*np.array(p['candidate_interval_nm'])/1000,alpha=.12,color='#f9ab31')
        twin=ax.twinx();k=p['derivatives']['parent_sigma100']['dimensionless_meridional_curvature']
        twin.plot(x,k,color='#a84753',lw=1.4,label='Radius × meridional curvature')
        twin.set_ylabel('Dimensionless curvature',color='#a84753',fontsize=9)
        ax.set(xlabel='Distance along local shaft frame (µm)',ylabel='Equivalent radius (µm)');ax.grid(alpha=.15)
        ax.legend(fontsize=7,loc='upper left')
    fig.suptitle('H01 width curvature probe · existing meshes, no segmentation changes\nShaded interval: current candidate. Gaps remain unmeasured. Local fixed-axis sections.',fontsize=12)
    fig.savefig(out/'profile_comparison.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review',type=Path,default=MAIN/'artifacts/h01_em_review_20261009/human_reviews/h01_spine_answers_2026-10-09T06-43-08-933Z/validated_review.json')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args();review=json.loads(args.review.read_text())
    out=(args.out or args.review.parent/'width_curvature_probe').resolve();out.mkdir(parents=True,exist_ok=True)
    analytic_checks();sr,code_path=load_refinement();cache={};rows=[];sources={};started=perf_counter()
    manifest=json.loads((MAIN/'artifacts/h01_em_review_20261009/manifest.json').read_text());cases={c['id']:c for c in manifest['cases']}
    for r in review['cases']:
        c=cases[r['case_id']];key=c['key']
        if key not in cache:
            parent=trimesh.load(c['parent_source'],process=False)
            assert sha(c['parent_source'])==c['parent_sha256']
            branch=MAIN/'artifacts/spine_shaft_trim_20261009/child_guard_comparison/branches'/key
            result=json.loads((branch/'result.json').read_text())
            skeleton=np.load(result['selection']['skeleton'])
            with np.load(branch/'candidate_faces.npz') as z:
                groups=list(zip(z['raw_ids'],z['part_ids'],np.split(z['faces'],z['offsets'][1:-1])))
            raw=MAIN/'artifacts/spine_surface_classification_guarded_20261009/comparison/branches'/key/'raw_faces.npz'
            with np.load(raw) as z:initial=np.split(z['faces'],z['offsets'][1:-1])
            ctx=sr._Context(parent,skeleton,initial,sr.RefinementConfig(trim_shaft_connections=True))
            cache[key]=(parent,ctx,groups)
            sources[key]=dict(parent_sha256=c['parent_sha256'],candidate_masks_sha256=sha(branch/'candidate_faces.npz'),
                              initial_masks_sha256=sha(raw),skeleton_sha256=sha(result['selection']['skeleton']))
        try:p=profile(sr,*cache[key],c['raw_id'],c['part_id'])
        except ValueError as exc:p=dict(status='unavailable',reason=str(exc))
        row=dict(case_id=c['id'],key=key,raw_id=c['raw_id'],part_id=c['part_id'],target_sha256=c['target_sha256'],
                 anatomy=r['annotation']['anatomy'],quality=r['annotation']['quality'],confidence=r['annotation']['confidence'],profile=p)
        rows.append(row);print(c['id'],r['annotation']['anatomy'],p['status'],p.get('reason',''),flush=True)
    result=dict(schema='h01-width-curvature-probe/1',manifest_id=review['manifest_id'],review_sha256=sha(args.review),
                script_sha256=sha(__file__),refinement_sha256=sha(code_path),step_nm=STEP,smoothing_sigma_nm=list(SMOOTHING),
                width_definition='Equivalent radius sqrt(area/pi) in fixed local-axis sections',
                derivative_definition='Smoothed radius second derivative; -r*r_second/(1+r_first^2)^1.5',
                missing_data='No filling or derivative across missing sections; finite contiguous runs only',
                thresholds_trained=False,segmentation_changed=False,analytic_derivative_checks='passed',
                sources=sources,seconds=perf_counter()-started,cases=rows)
    (out/'profiles.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    plot_cases(out,rows)
    print('Probe saved:',out,flush=True)


if __name__=='__main__':main()
