#!/usr/bin/env python3
"""Matched scientific previews of final shaft-subtraction/splitting results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import trimesh

from preview_spine_refinement_changes import axis_samples,local_view,render_panel,unpack


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rows',type=int,default=4)
    parser.add_argument('--key',action='append')
    parser.add_argument('--raw-ids',type=int,nargs='+')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    index=[]
    for path in sorted((args.comparison/'branches').glob('*/result.json')):
        result=json.loads(path.read_text());key=result['key']
        if args.key and key not in args.key: continue
        rows=[r for r in result['changes'] if not args.raw_ids or r['raw_id'] in args.raw_ids]
        if not rows: continue
        previous=Path(result['previous_comparison'])
        before=unpack(previous/'refined_faces.npz');raw=unpack(previous/'raw_faces.npz')
        with np.load(path.parent/'candidate_faces.npz') as z:
            masks=np.split(z['faces'],z['offsets'][1:-1])
            after={}
            for raw_id,part,mask in zip(z['raw_ids'],z['part_ids'],masks):
                after.setdefault(int(raw_id),[]).append((int(part),mask))
        m=trimesh.load(result['selection']['source'],process=False,force='mesh')
        centers=m.triangles_center;tree=cKDTree(centers)
        axis,skel=axis_samples(np.load(result['selection']['skeleton']))
        for page,start in enumerate(range(0,len(rows),args.rows)):
            block=rows[start:start+args.rows]
            fig=plt.figure(figsize=(20,3*len(block)+.5),layout='constrained')
            fig.suptitle(key+' | blue/orange: final individual outputs; grey: branch',fontsize=11)
            for row,r in enumerate(block):
                raw_id=r['raw_id'];original=before.get(raw_id,raw[raw_id]);parts=[g for _,g in after.get(raw_id,[])]
                union=np.concatenate(parts) if parts else np.empty(0,dtype=int)
                local,nearby,ss,middle,span=local_view(m,centers,tree,axis,skel,original,union)
                cv=local[np.unique(m.faces[np.union1d(original,union)])];cspan=np.ptp(cv,axis=0).max()
                if axis.query(centers[original].mean(0))[0]>2000 and span>cspan+3:
                    middle=(cv.min(0)+cv.max(0))/2;span=max(2.4,cspan+2.4)
                    points=m.vertices[np.unique(m.faces[original])];origin=(points.min(0)+points.max(0))/2
                    radius=np.linalg.norm(points-origin,axis=1).max()+1500
                    nearby=np.asarray(tree.query_ball_point(origin,radius),dtype=int)
                    ss=ss[np.linalg.norm(ss.mean(1)-middle,axis=1)<span]
                triangles=local[m.faces]
                for col,(angle,enabled) in enumerate([((18,-65),False),((18,-65),True),((18,65),False),((18,65),True)]):
                    ax=fig.add_subplot(len(block),4,4*row+col+1,projection='3d')
                    kept=(parts[0] if parts else []) if enabled else (original if raw_id in before else [])
                    extra=np.concatenate(parts[1:]) if enabled and len(parts)>1 else []
                    label=('AFTER '+str([len(p) for p in parts])) if enabled else ('BEFORE '+str(len(original)) if raw_id in before else 'BEFORE: filtered out')
                    title=f'raw {raw_id} | {label}\nwidth {span:.2f} µm'
                    render_panel(ax,triangles,nearby,kept,extra,ss,middle,span,angle,title)
                index.append(dict(key=key,raw_id=raw_id,image=str(args.output/f'{key}_page_{page:02d}.png'),
                                  final_parts=len(parts),final_indices=r['final_indices']))
            fig.savefig(args.output/f'{key}_page_{page:02d}.png',dpi=110);plt.close(fig)
            print(json.dumps(dict(key=key,page=page,candidates=len(block))),flush=True)
    (args.output/'index.json').write_text(json.dumps(index,indent=2)+'\n')


if __name__=='__main__':
    main()
