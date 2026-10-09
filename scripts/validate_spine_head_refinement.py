#!/usr/bin/env python3
"""Compare bounded ragged-head completion against frozen shaft refinement.

All final filters run through calculate_spines_on_branch in both modes. Cached
extraction is allowed only for unchanged extraction/initial length functions,
matching source hashes and H01 parameters. Changed parts repeat the real length
filter. Writes JSON, original parent face masks and OFF meshes, never a report.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    temp=Path(path).with_suffix('.json.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def unpack(path):
    import numpy as np
    with np.load(path) as z:
        return list(zip(z['raw_ids'].tolist(),np.split(z['faces'],z['offsets'][1:-1])))


def pack(path,rows):
    import numpy as np
    ids=[r['faces'] for r in rows]
    np.savez_compressed(path,faces=np.concatenate(ids) if ids else np.empty(0,dtype=int),
                        offsets=np.r_[0,np.cumsum([len(g) for g in ids])],
                        raw_ids=[r['raw_id'] for r in rows],part_ids=[r['part_id'] for r in rows])


def function_hashes(path):
    tree=ast.parse(Path(path).read_text())
    return {node.name:hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
            for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}


def triangle_keys(mesh):
    import numpy as np
    triangles=np.asarray(mesh.triangles,dtype=np.float64)
    vertices=triangles.view(np.dtype((np.void,24))).reshape(-1,3).copy()
    vertices.sort(axis=1)
    return np.sort(vertices.copy().view(np.dtype((np.void,72))).reshape(-1))


def job(args,path):
    for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[variable]='1'
    os.environ.setdefault('MPLCONFIGDIR','/tmp/neurd-spine-refinement-mpl')
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(args.code_snapshot))
    import neurd
    import numpy as np
    import trimesh
    from neurd import parameters,spine_utils as spu,spine_refinement as sr
    assert Path(neurd.__file__).resolve().parent.parent==args.code_snapshot
    previous=json.loads(path.read_text());key=previous['key']
    dest=args.output/'branches'/key;dest.mkdir(parents=True,exist_ok=True)
    os.chdir(dest)
    parent=trimesh.load(previous['selection']['source'],process=False,force='mesh')
    skeleton=np.load(previous['selection']['skeleton'])
    assert sha(previous['selection']['source'])==previous['parent_sha256']
    assert sha(previous['selection']['skeleton'])==previous['skeleton_sha256']
    raw_pairs=unpack(path.parent/'raw_faces.npz')
    assert [r for r,_ in raw_pairs]==list(range(len(raw_pairs)))
    groups=[g for _,g in raw_pairs];meshes=[parent.submesh([g],append=True) for g in groups]
    prefix=json.loads((path.parent/'shared_prefix.json').read_text())
    length_ids=prefix['length_raw_ids'];length_meshes=[meshes[i] for i in length_ids]
    parameters.params.use('h01')
    assert parameters.params._data==previous['parameters']
    baseline_path=args.baseline/'branches'/key/'candidate_faces.npz'
    with np.load(baseline_path) as z:
        baseline_masks=np.split(z['faces'],z['offsets'][1:-1])
        old_refined={(int(raw),int(part)):mask for raw,part,mask in zip(z['raw_ids'],z['part_ids'],baseline_masks)}
    assert len(old_refined)==len(baseline_masks)
    frozen_previous = None
    if args.validated_baseline:
        reference_folder = args.validated_baseline/'branches'/key
        reference = json.loads((reference_folder/'result.json').read_text())
        assert reference['status']=='passed' and reference['source_sha256']==previous['parent_sha256']
        assert reference['skeleton_sha256']==previous['skeleton_sha256']
        with np.load(reference_folder/'previous_faces.npz') as z:
            masks=np.split(z['faces'],z['offsets'][1:-1].astype(int)) if len(z['faces']) else []
            verified={(int(raw),int(part)):mask for raw,part,mask in zip(z['raw_ids'],z['part_ids'],masks)}
        assert verified.keys()==old_refined.keys()
        assert all(np.array_equal(verified[key],mask) for key,mask in old_refined.items())
        baseline_result=json.loads((args.baseline/'branches'/key/'result.json').read_text())
        frozen_previous=[{**r,'faces':old_refined[(r['raw_id'],r['part_id'])]}
                         for r in baseline_result['candidate_outputs']]
    soma_path=Path(previous['selection']['source']).parents[2]/'soma/soma_mesh.off'
    soma=trimesh.load(soma_path,process=False,force='mesh').vertices if soma_path.exists() else None
    kwargs=dict(soma_verts=soma,soma_vertex_nullification=soma is not None,cgal_folder=dest/'cgal',print_flag=False)
    original_raw,original_length,original_refine=spu.get_spine_meshes_unfiltered_from_mesh,spu.filter_spine_meshes,sr.refine_spine_face_indices
    captured={};length_calls=[]

    def cached_raw(*a,**kw):
        return (meshes,groups) if kw.get('return_mesh_idx') else meshes

    def length(mesh_list,*a,**kw):
        if len(mesh_list)==len(meshes) and all(a is b for a,b in zip(mesh_list,meshes)):
            return length_meshes
        started=perf_counter();result=original_length(mesh_list,*a,**kw)
        length_calls.append(dict(input_faces=[len(m.faces) for m in mesh_list],
                                 passing_input_indices=[i for i,m in enumerate(mesh_list) if any(m is p for p in result)],
                                 seconds=perf_counter()-started))
        return result

    def refine(*a,**kw):
        result,d=original_refine(*a,**kw);captured.update(groups=result,diagnostics=d);return result,d

    spu.get_spine_meshes_unfiltered_from_mesh=cached_raw
    spu.filter_spine_meshes=length
    sr.refine_spine_face_indices=refine
    exports={};timings={};stages={}
    try:
        for mode,config in (('previous',{'trim_shaft_connections':True}),
                            ('candidate',{'trim_shaft_connections':True,'grow_ragged_heads':True})):
            if mode=='previous' and frozen_previous is not None:
                exports[mode]=frozen_previous
                timings[mode]=dict(cached=True,reference=str(args.validated_baseline/'branches'/key),
                                   recorded_post_seconds=reference['timings']['previous']['post_seconds'],
                                   recorded_geometry_seconds=reference['timings']['previous']['geometry_seconds'])
                pack(dest/(mode+'_faces.npz'),frozen_previous)
                continue
            started=perf_counter();stage={};np.random.seed(0)
            result,volumes=spu.calculate_spines_on_branch(SimpleNamespace(mesh=parent,skeleton=skeleton),
                    geometric_refinement=True,geometric_refinement_config=config,
                    geometric_refinement_diagnostics=stage,**kwargs)
            timings[mode]=dict(post_seconds=perf_counter()-started,geometry_seconds=stage['seconds'])
            rows=[None]*len(result)
            for record in stage['candidates']:
                for mapping in record['final_output_mapping']:
                    final_index=mapping['final_output_index']
                    if final_index is None: continue
                    index=mapping['output_index'];ids=captured['groups'][index]
                    assert np.array_equal(triangle_keys(result[final_index]),triangle_keys(parent.submesh([ids],append=True)))
                    rows[final_index]=dict(raw_id=record['raw_input_index'],part_id=mapping['part_index'],
                                          output_index=index,faces=ids,volume_nm3=float(volumes[final_index]))
            assert all(r is not None for r in rows),'Missing split-to-final mesh mapping'
            for i,r in enumerate(rows):r['final_index']=i
            if mode=='previous':
                assert len(rows)==len(old_refined)
                assert all(np.array_equal(old_refined[(r['raw_id'],r['part_id'])],r['faces']) for r in rows), 'Disabled head option did not reproduce the frozen baseline'
            else:
                folder=dest/'candidate';folder.mkdir(exist_ok=True)
                for i,(mesh,r) in enumerate(zip(result,rows)):
                    assert len(mesh.faces)==len(r['faces'])
                    assert np.isfinite(r['volume_nm3']) and r['volume_nm3']>0
                    mesh.export(folder/f'spine_{i:03d}.off')
                    r['mesh']=str(folder/f'spine_{i:03d}.off')
            pack(dest/(mode+'_faces.npz'),rows)
            exports[mode]=rows;stages[mode]=stage
    finally:
        spu.get_spine_meshes_unfiltered_from_mesh=original_raw
        spu.filter_spine_meshes=original_length
        sr.refine_spine_face_indices=original_refine
    decisions={r['raw_input_index']:r for r in stages['candidate']['candidates']}
    _,canonical=np.unique(np.sort(parent.faces,axis=1),axis=0,return_inverse=True)
    # The stage may grow original seeds, but subtraction/splitting may never
    # create overlapping child parts or introduce synthetic parent faces.
    changes=[]
    for raw_id in sorted({raw for raw,part in old_refined}|{r['raw_id'] for r in exports['candidate']}):
        after=[r for r in exports['candidate'] if r['raw_id']==raw_id]
        before=[(part,mask) for (raw,part),mask in old_refined.items() if raw==raw_id]
        physical=[np.unique(canonical[r['faces']]) for r in after]
        assert all(not np.intersect1d(a,b).size for i,a in enumerate(physical) for b in physical[i+1:])
        unchanged=(len(before)==len(after) and all(
            (raw_id,r['part_id']) in old_refined and np.array_equal(old_refined[(raw_id,r['part_id'])],r['faces'])
            for r in after))
        if not unchanged:
            changes.append(dict(raw_id=raw_id,previous_faces=[len(mask) for part,mask in before],
                                output_parts=len(after),output_faces=[len(r['faces']) for r in after],
                                final_indices=[r['final_index'] for r in after],decision=decisions.get(raw_id)))
    result=dict(status='passed',key=key,selection=previous['selection'],previous_comparison=str(args.baseline/'branches'/key),cached_extraction_comparison=str(path.parent),
                previous_spines=len(exports['previous']),candidate_spines=len(exports['candidate']),
                timings=timings,changed_part_length_calls=length_calls,refinement=stages['candidate'],changes=changes,
                candidate_outputs=[{k:v for k,v in r.items() if k!='faces'} for r in exports['candidate']],
                code_sha256={name:sha(args.code_snapshot/'neurd'/name) for name in ('spine_utils.py','spine_refinement.py','spine_volume.py')},
                source_sha256=previous['parent_sha256'],skeleton_sha256=previous['skeleton_sha256'])
    write_json(dest/'result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--code-snapshot',type=Path,required=True)
    parser.add_argument('--previous-code',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--validated-baseline',type=Path,help='Reuse a previously reproduced baseline, checking every parent mask')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--key',action='append')
    parser.add_argument('--job',type=Path)
    args=parser.parse_args()
    args.comparison,args.code_snapshot,args.previous_code,args.output=(p.resolve() for p in
                (args.comparison,args.code_snapshot,args.previous_code,args.output))
    args.baseline=args.baseline.resolve()
    if args.validated_baseline: args.validated_baseline=args.validated_baseline.resolve()
    before=function_hashes(args.previous_code/'neurd/spine_utils.py');after=function_hashes(args.code_snapshot/'neurd/spine_utils.py')
    changed=[name for name in before if before[name]!=after.get(name)]
    assert changed==['calculate_spines_on_branch'],changed
    assert sha(args.previous_code/'neurd/spine_volume.py')==sha(args.code_snapshot/'neurd/spine_volume.py')
    upstream=['parameters.py','_cgal_segmentation.py','_mesh_ops.py','submesh_ops.py']
    assert all(sha(args.previous_code/'neurd'/n)==sha(args.code_snapshot/'neurd'/n) for n in upstream)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.job:
        job(args,args.job);return
    paths=sorted((args.comparison/'branches').glob('*/result.json'))
    if args.key: paths=[p for p in paths if p.parent.name in args.key]
    manifest=dict(comparison=str(args.comparison),code_snapshot=str(args.code_snapshot),previous_code=str(args.previous_code),
                  script_sha256=sha(__file__),changed_functions=changed,parameters_unchanged=True,
                  frozen_prefix=True,baseline=str(args.baseline),config={'trim_shaft_connections':True,'grow_ragged_heads':True},branches=[p.parent.name for p in paths])
    manifest['validated_baseline']=str(args.validated_baseline) if args.validated_baseline else None
    write_json(args.output/'manifest.json',manifest)

    def execute(path):
        folder=args.output/'branches'/path.parent.name;folder.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,__file__,'--comparison',str(args.comparison),'--code-snapshot',str(args.code_snapshot),
                 '--previous-code',str(args.previous_code),'--baseline',str(args.baseline),'--output',str(args.output),'--job',str(path)]
        if args.validated_baseline: command+=['--validated-baseline',str(args.validated_baseline)]
        with (folder/'run.log').open('w') as log:
            process=subprocess.run(command,cwd=folder,stdout=log,stderr=subprocess.STDOUT)
        if process.returncode:
            return dict(status='failed',key=path.parent.name,log=str(folder/'run.log'))
        return json.loads((folder/'result.json').read_text())
    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(execute,paths):
            results.append(r);write_json(args.output/'results.json',results)
            print(json.dumps({k:r[k] for k in ('key','status','previous_spines','candidate_spines') if k in r}),flush=True)
    assert all(r['status']=='passed' for r in results),'Some branches failed; inspect their run.log'


if __name__=='__main__':
    main()
