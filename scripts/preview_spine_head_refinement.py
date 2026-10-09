#!/usr/bin/env python3
"""Matched PNG and offline interactive views of head completion and controls."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import trimesh

from preview_spine_refinement_changes import axis_samples, local_view, render_panel


HTML = r'''<!doctype html><html lang="ru"><meta charset="utf-8">
<title>H01: сравнение достраивания головок</title>
<style>body{font:16px system-ui;margin:20px;color:#18212b;background:#fafbfc}
select,button{font:inherit;padding:8px;margin-right:8px}h1{font-size:24px}
.panels{display:grid;grid-template-columns:1fr 1fr;gap:14px}.plot{height:65vh;min-height:440px}
.panel{background:white;border:1px solid #ccd4dd;border-radius:8px;padding:10px}
#meta{margin:12px 0;white-space:pre-line}a{color:#147ca3}
@media(max-width:900px){.panels{grid-template-columns:1fr}.plot{height:55vh}}
</style><h1>H01: достраивание головок</h1>
<p>Синий — прежнее выделение, оранжевый — добавленная поверхность, серый — ветвь.
Вращение и масштаб синхронизированы между видами. Контрольные примеры отмечены в списке.</p>
<select id="case" aria-label="Пример"></select><button id="prev">←</button><button id="next">→</button>
<button id="reset">Сбросить вид</button><div id="meta"></div>
<div class="panels"><div class="panel"><b>До</b><div class="plot" id="before"></div></div>
<div class="panel"><b>После</b><div class="plot" id="after"></div></div></div>
<p><a id="png" href="">Два ракурса в PNG</a></p>
<script src="plotly.min.js"></script><script src="index.js"></script>
<script>
window.HEAD_COMPARE_CASES={};let selected=0,syncing=false,requestNumber=0;
const before=document.getElementById('before'),after=document.getElementById('after');
const menu=document.getElementById('case');
for(const [i,c] of HEAD_COMPARE_INDEX.entries())menu.add(new Option(c.label,i));
const initialCamera={eye:{x:1.6,y:-1.7,z:.7},up:{x:0,y:0,z:1},projection:{type:'orthographic'}};
function load(c){return new Promise((resolve,reject)=>{
 if(HEAD_COMPARE_CASES[c.id])return resolve(HEAD_COMPARE_CASES[c.id]);
 const s=document.createElement('script');s.src=c.path;s.onload=()=>resolve(HEAD_COMPARE_CASES[c.id]);s.onerror=reject;document.head.append(s);
});}
function traces(c,isAfter){return [{...(isAfter?c.after_context:c.before_context),opacity:.17,color:'#bdc5cc',name:'Ветвь'},
 {...(isAfter?c.kept:c.original),color:'#137ca3',name:'Прежняя поверхность'},
 ...(isAfter?[{...c.added,color:'#f29432',name:'Добавленная поверхность'}]:[])];}
function layout(c){const ranges={};for(const [i,name] of ['xaxis','yaxis','zaxis'].entries())
 ranges[name]={range:[c.middle[i]-c.span/2,c.middle[i]+c.span/2],title:'мкм',showgrid:false,zeroline:false};
 return {margin:{l:0,r:0,t:10,b:0},showlegend:false,scene:{...ranges,aspectmode:'cube',camera:initialCamera}};}
async function selectCase(i){const request=++requestNumber;selected=(i+HEAD_COMPARE_INDEX.length)%HEAD_COMPARE_INDEX.length;menu.value=selected;
 const item=HEAD_COMPARE_INDEX[selected],c=await load(item);
 if(request!==requestNumber)return null;
 document.getElementById('meta').textContent=item.label+'\n'+c.description;
 document.getElementById('png').href=item.image;
 await Promise.all([Plotly.react(before,traces(c,false),layout(c),{responsive:true,displaylogo:false}),
                    Plotly.react(after,traces(c,true),layout(c),{responsive:true,displaylogo:false})]);
 if(request!==requestNumber)return null;
 for(const [a,b] of [[before,after],[after,before]]){a.removeAllListeners('plotly_relayout');a.on('plotly_relayout',async event=>{
  if(syncing||!event['scene.camera'])return;syncing=true;
  try{await Plotly.relayout(b,{'scene.camera':event['scene.camera']});}finally{syncing=false;}
 });}window.HEAD_COMPARE.current=c;return c;
}
window.HEAD_COMPARE={selectCase,current:null,pending:null};
function choose(i){HEAD_COMPARE.pending=selectCase(i);return HEAD_COMPARE.pending;}
menu.onchange=()=>choose(Number(menu.value));document.getElementById('prev').onclick=()=>choose(selected-1);
document.getElementById('next').onclick=()=>choose(selected+1);
document.getElementById('reset').onclick=()=>Promise.all([before,after].map(p=>Plotly.relayout(p,{'scene.camera':initialCamera})));
choose(0);
</script></html>'''


def unpack(path):
    with np.load(path) as z:
        masks = np.split(z['faces'], z['offsets'][1:-1].astype(int)) if len(z['faces']) else []
        result = {}
        for raw, part, mask in zip(z['raw_ids'], z['part_ids'], masks):
            result.setdefault(int(raw), []).append((int(part), mask))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plotly', type=Path, required=True)
    parser.add_argument('--partial', action='store_true', help='Preview completed branches while the comparison is running')
    parser.add_argument('--title', default='H01: достраивание головок')
    parser.add_argument('--include-cases', type=Path, help='Additional case IDs and notes, without human annotations')
    parser.add_argument('--only-listed-cases', action='store_true', help='Render review and include-cases entries only')
    args = parser.parse_args()
    rows = json.loads((args.comparison/'results.json').read_text())
    manifest = json.loads((args.comparison/'manifest.json').read_text())
    assert (args.partial or len(rows) == len(manifest['branches'])) and all(r['status']=='passed' for r in rows)
    review = json.loads(args.review.read_text())
    labels = {(r['key'], r['raw_id']): r for r in review['cases']}
    extras = json.loads(args.include_cases.read_text()) if args.include_cases else []
    extra_labels = {(r['key'], r['raw_id']): r for r in extras}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'cases').mkdir(exist_ok=True)
    index = []
    for row in rows:
        path = args.comparison/'branches'/row['key']
        before, after = unpack(path/'previous_faces.npz'), unpack(path/'candidate_faces.npz')
        raw_ids = sorted((set() if args.only_listed_cases else {r['raw_id'] for r in row['changes']}) | {
            r['raw_id'] for r in review['cases']+extras if r['key']==row['key']})
        if not raw_ids:
            continue
        parent = trimesh.load(row['selection']['source'], process=False, force='mesh')
        assert hashlib.sha256(Path(row['selection']['source']).read_bytes()).hexdigest()==row['source_sha256']
        centers = parent.triangles_center
        tree = cKDTree(centers)
        axis, skeleton = axis_samples(np.load(row['selection']['skeleton']))
        for raw in raw_ids:
            original = np.unique(np.concatenate([mask for part, mask in before.get(raw, [])])) if raw in before else np.empty(0, int)
            proposal = np.unique(np.concatenate([mask for part, mask in after.get(raw, [])])) if raw in after else np.empty(0, int)
            assert len(original), 'New raw output without a baseline requires a separate preview'
            local, nearby, segments, middle, span = local_view(parent, centers, tree, axis, skeleton, original, proposal)
            union = np.union1d(original, proposal)
            vertices = local[np.unique(parent.faces[union])]
            cspan = np.ptp(vertices, axis=0).max()
            if axis.query(centers[original].mean(0))[0]>2000 and span>cspan+3:
                middle=(vertices.min(0)+vertices.max(0))/2
                span=max(2.4,cspan+2.4)
                points=parent.vertices[np.unique(parent.faces[union])]
                origin=(points.min(0)+points.max(0))/2
                nearby=np.asarray(tree.query_ball_point(origin,np.linalg.norm(points-origin,axis=1).max()+1500),int)
                segments=segments[np.linalg.norm(segments.mean(1)-middle,axis=1)<span]
            kept, added = np.intersect1d(original, proposal), np.setdiff1d(proposal, original)
            nearby = np.union1d(nearby, union)
            removed = np.setdiff1d(original, proposal)
            human = labels.get((row['key'], raw))
            extra = extra_labels.get((row['key'], raw))
            changed = raw in {r['raw_id'] for r in row['changes']}
            control = bool(human and human['expectations']['accepted_geometry_control'])
            id_ = row['key']+f'_raw_{raw:03d}'
            case_label = human['case_id'] if human else extra['case_id'] if extra else None
            label = (case_label+' · ' if case_label else '')+row['key']+f' · {raw}'
            if control:
                label += ' · контроль'
            elif not changed:
                label += ' · без изменений'
            image = id_+'.png'
            fig = plt.figure(figsize=(17, 4.4), layout='constrained')
            fig.suptitle(label+' | blue: previous mask; orange: added surface', fontsize=10)
            for col, (angle, enabled) in enumerate([((18,-65),False),((18,-65),True),((18,65),False),((18,65),True)]):
                ax = fig.add_subplot(1,4,col+1,projection='3d')
                render_panel(ax,local[parent.faces],nearby,kept if enabled else original,added if enabled else [],
                             segments,middle,span,angle,('After' if enabled else 'Before')+f' | {len(proposal) if enabled else len(original)} faces\nwidth {span:.2f} µm')
            fig.savefig(args.output/image,dpi=130)
            plt.close(fig)

            def mesh(ids):
                faces = parent.faces[np.asarray(ids, int)]
                vertices, inverse = np.unique(faces, return_inverse=True)
                faces = inverse.reshape(-1,3)
                xyz = local[vertices]
                return dict(type='mesh3d',x=xyz[:,0].tolist(),y=xyz[:,1].tolist(),z=xyz[:,2].tolist(),
                            i=faces[:,0].tolist(),j=faces[:,1].tolist(),k=faces[:,2].tolist(),flatshading=False,hoverinfo='skip')
            payload = dict(id=id_,key=row['key'],raw_id=raw,original=mesh(original),kept=mesh(kept),added=mesh(added),
                           before_context=mesh(np.setdiff1d(nearby,original)),
                           after_context=mesh(np.setdiff1d(nearby,proposal)),
                           parent_roi_faces=len(nearby),middle=middle.tolist(),span=float(span),
                           description=f'{len(original)} → {len(proposal)} треугольников. Добавлено: {len(added)}; удалено: {len(removed)}.')
            if extra and extra.get('note'):
                payload['description']+='\n'+extra['note']
            if human:
                anatomy={'one_spine':'один шипик','multiple_spines':'несколько шипиков','shaft_bump':'бугорок / участок ствола','uncertain':'неопределённо'}
                quality={'looks_ok':'выделение приемлемое','shaft_included':'захвачен ствол','head_incomplete':'неполная головка','spines_merged':'шипики склеены','uncertain':'неопределённо'}
                annotation=human['annotation']
                payload['description']+='\nТвоя оценка исходного выделения: '+anatomy[annotation['anatomy']]
                if annotation['quality']:payload['description']+='; '+', '.join(quality[q] for q in annotation['quality'])
                if annotation['comment'].strip():payload['description']+='\n'+annotation['comment'].strip()
            data_path = 'cases/'+id_+'.js'
            (args.output/data_path).write_text('window.HEAD_COMPARE_CASES['+json.dumps(id_)+']='+json.dumps(payload,allow_nan=False)+';\n')
            index.append(dict(id=id_,key=row['key'],raw_id=raw,label=label,control=control,changed=changed,image=image,path=data_path,
                              before_faces=len(original),after_faces=len(proposal),added_faces=len(added),removed_faces=len(removed)))
            print(json.dumps(dict(key=row['key'],raw_id=raw,control=control)),flush=True)
    assert index
    shutil.copyfile(args.plotly,args.output/'plotly.min.js')
    page = HTML.replace('H01: сравнение достраивания головок', html.escape(args.title))
    page = page.replace('H01: достраивание головок', html.escape(args.title))
    (args.output/'index.html').write_text(page)
    (args.output/'index.js').write_text('window.HEAD_COMPARE_INDEX='+json.dumps(index,ensure_ascii=False)+';\n')
    (args.output/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    provenance=dict(comparison=str(args.comparison.resolve()),
        comparison_results_sha256=hashlib.sha256((args.comparison/'results.json').read_bytes()).hexdigest(),
        review=str(args.review.resolve()),review_sha256=hashlib.sha256(args.review.read_bytes()).hexdigest(),
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(Path(__file__).with_name('preview_spine_refinement_changes.py').read_bytes()).hexdigest(),
        partial=args.partial,cases=len(index),changed=sum(c['changed'] for c in index),
        title=args.title,only_listed_cases=args.only_listed_cases,
        include_cases_sha256=hashlib.sha256(args.include_cases.read_bytes()).hexdigest() if args.include_cases else None,
        original_parent_surface_present_in_both_panes=True)
    (args.output/'render_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__=='__main__':
    main()
