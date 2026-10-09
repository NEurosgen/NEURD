"""Verify real EM/mesh coordinates and offline interactions in headless Chrome."""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import requests
import trimesh
import websocket
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('/home/eugen/Desktop/CodeWork/Projects/NEURD_modif/NEURD/artifacts/h01_em_review_20261009'))
    parser.add_argument('--limit', type=int, default=24)
    args = parser.parse_args()
    out = args.out.resolve()
    manifest = json.loads((out/'manifest.json').read_text())
    checked_frames = 0
    for case in manifest['cases'][:args.limit]:
        folder = out/case['path']
        with np.load(folder/'em_volume.npz') as saved:
            assert saved['manifest_id'].item() == manifest['manifest_id']
            raw = saved['image_xyz']; low = saved['bbox_min_voxels']; res = saved['resolution_nm']
        text = (folder/'data.js').read_text()
        data = json.loads(text.split(']=',1)[1].rstrip(';\n'))
        window = data['display_range']
        target = trimesh.load(folder/'target_mesh.off',process=False)
        vertices = np.array(data['target_mesh']['vertices'])*1000+low*res
        assert np.allclose(vertices,target.vertices,atol=1e-7,rtol=0)
        assert np.array_equal(data['target_mesh']['faces'],target.faces)
        assert isinstance(data['source_c3_id'],str)
        assert all(isinstance(s['id'],str) for s in data['synapses'])
        assert data['target_sha256'] == hashlib.sha256((folder/'target_mesh.off').read_bytes()).hexdigest()
        triangle_z = target.triangles[:,:,2]
        for k, section in enumerate(data['sections']):
            png = np.array(Image.open(folder/'slices'/f'{k:03}.png'))
            expected = np.clip((raw[:,:,k].astype(np.float32)-window[0])*255/(window[1]-window[0]),0,255).astype(np.uint8).T[::-1]
            assert np.array_equal(png,expected), (case['id'],k)
            assert section['z_nm'] == (low[2]+k+.5)*res[2]
            # Check every contour point against the original, independent mesh
            # plane intersection: catches swapping x/y, wrong origin or scaling.
            mesh_section = target.section(plane_origin=[0,0,section['z_nm']],plane_normal=[0,0,1])
            exact = [] if mesh_section is None else [e.discrete(mesh_section.vertices) for e in mesh_section.entities]
            assert len(exact) == len(section['target'])
            crossed = np.any((triangle_z.min(axis=1)<section['z_nm']) & (triangle_z.max(axis=1)>section['z_nm']))
            assert not crossed or len(section['target']) > 0, (case['id'],k,'lost open contour')
            for actual,pixels in zip(exact,section['target']):
                rebuilt = np.array(pixels)*res[:2]+(low*res)[:2]
                assert np.allclose(rebuilt,actual[:,:2],atol=.0005,rtol=0)
            checked_frames += 1
    tmp = Path(tempfile.mkdtemp(prefix='h01-em-review-browser-'))
    downloads = tmp/'downloads';downloads.mkdir()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port = sock.getsockname()[1]
    log = (out/'browser.log').open('w')
    process = subprocess.Popen(['/usr/bin/google-chrome','--headless=new','--no-sandbox',
      '--disable-background-networking','--disable-extensions','--no-first-run','--no-default-browser-check',
      '--disable-dev-shm-usage','--use-angle=swiftshader','--enable-unsafe-swiftshader',
      '--remote-debugging-address=127.0.0.1',f'--remote-debugging-port={port}',
      '--remote-allow-origins=*',f'--user-data-dir={tmp/"profile"}',
      '--window-size=1800,1450','about:blank'],stdout=log,stderr=log)
    ws=None;events=[];serial=0
    session=requests.Session();session.trust_env=False
    try:
        for _ in range(100):
            try:
                tab=next(t for t in session.get(f'http://127.0.0.1:{port}/json',timeout=1).json() if t['type']=='page');break
            except (requests.RequestException,ValueError,StopIteration):time.sleep(.2)
        else:raise RuntimeError('Chrome target missing')
        ws=websocket.create_connection(tab['webSocketDebuggerUrl'],timeout=120,suppress_origin=True)
        def command(method,params=None):
            nonlocal serial
            serial+=1;number=serial;ws.send(json.dumps(dict(id=number,method=method,params=params or {})))
            while True:
                message=json.loads(ws.recv())
                if message.get('id')==number:
                    assert 'error' not in message,message
                    return message.get('result',{})
                events.append(message)
        def evaluate(expression):
            r=command('Runtime.evaluate',dict(expression=expression,returnByValue=True,awaitPromise=True))
            assert 'exceptionDetails' not in r,r
            return r['result'].get('value')
        def ready():
            for _ in range(120):
                if evaluate('!!window.H01_REVIEW'):break
                time.sleep(.2)
            else:raise RuntimeError('Viewer did not start')
            evaluate('H01_REVIEW.pendingCase.then(()=>true)')
            evaluate('H01_REVIEW.pendingFrame.then(()=>true)')
            evaluate('H01_REVIEW.pendingMesh.then(()=>true)')
            assert evaluate('!!H01_REVIEW.current&&!!document.getElementById("mesh-plot").data')
        command('Runtime.enable');command('Page.enable');command('Network.enable')
        # All frames/scripts/Plotly must work from file:// with the internet off.
        command('Network.emulateNetworkConditions',dict(offline=True,latency=0,downloadThroughput=-1,uploadThroughput=-1))
        command('Browser.setDownloadBehavior',dict(behavior='allow',downloadPath=str(downloads)))
        command('Page.navigate',dict(url=(out/'index.html').as_uri()));ready()
        for n in range(args.limit):
            evaluate(f'H01_REVIEW.selectCase({n}).then(()=>true)')
            assert evaluate('''(()=>{const c=H01_REVIEW.current,g=document.getElementById('mesh-plot');
              for(const [i,m] of [[0,c.parent_mesh],[1,c.target_mesh]]){
                const t=g.data[i];if(t.x.length!==m.vertices.length||t.i.length!==m.faces.length)return false;
                for(let j=0;j<m.vertices.length;j++)if(t.x[j]!==m.vertices[j][0]||t.y[j]!==m.vertices[j][1]||t.z[j]!==m.vertices[j][2])return false;
                for(let j=0;j<m.faces.length;j++)if(t.i[j]!==m.faces[j][0]||t.j[j]!==m.faces[j][1]||t.k[j]!==m.faces[j][2])return false;
              }return document.getElementById('slice').max==c.sections.length-1;})()'''),n
            evaluate('H01_REVIEW.requestFrame(0).then(()=>H01_REVIEW.pendingMesh).then(()=>true)')
            assert evaluate('H01_REVIEW.frame===0')
            evaluate('H01_REVIEW.requestFrame(H01_REVIEW.current.sections.length-1).then(()=>H01_REVIEW.pendingMesh).then(()=>true)')
            assert evaluate('''(()=>{const c=H01_REVIEW.current,z=(c.sections[H01_REVIEW.frame].z_nm-c.origin_nm[2])/1000;
                return document.getElementById('mesh-plot').data[2].z.every(v=>v===z);})()''')
        evaluate('H01_REVIEW.selectCase(0).then(()=>true)')
        evaluate("document.getElementById('reset-slice').click();H01_REVIEW.pendingFrame.then(()=>H01_REVIEW.pendingMesh).then(()=>true)")
        before=evaluate('H01_REVIEW.frame')
        evaluate('document.activeElement.blur()')
        command('Input.dispatchKeyEvent',dict(type='keyDown',key='ArrowRight',code='ArrowRight'))
        command('Input.dispatchKeyEvent',dict(type='keyUp',key='ArrowRight',code='ArrowRight'))
        evaluate('H01_REVIEW.pendingFrame.then(()=>true)');assert evaluate('H01_REVIEW.frame')==before+1
        evaluate(r'''document.getElementById('anatomy').value='uncertain';document.getElementById('confidence').value='low';
            document.getElementById('em-readability').value='difficult';document.getElementById('comment').value='Тест, «сомнение»\nстрока 2';
            document.querySelector('.quality[value="uncertain"]').click();H01_REVIEW.saveForm();true''')
        expected=evaluate('H01_REVIEW.answers.C01')
        evaluate("document.getElementById('mark-mode').click()")
        point=evaluate("(()=>{const r=document.getElementById('em-overlay').getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2};})()")
        command('Input.dispatchMouseEvent',dict(type='mousePressed',button='left',clickCount=1,**point))
        command('Input.dispatchMouseEvent',dict(type='mouseReleased',button='left',clickCount=1,**point))
        marks=evaluate('H01_REVIEW.answers.C01.marks');assert len(marks)==1
        expected_xyz=evaluate('''(()=>{const c=H01_REVIEW.current;return [c.origin_nm[0]+c.shape_xyz[0]*c.resolution_nm[0]/2,
          c.origin_nm[1]+c.shape_xyz[1]*c.resolution_nm[1]/2,c.sections[H01_REVIEW.frame].z_nm];})()''')
        # Input.dispatchMouseEvent rounds subpixel screen coordinates; allow
        # less than one native XY voxel, while z must match the plane exactly.
        assert np.allclose(marks[0]['xyz_nm'][:2],expected_xyz[:2],atol=8,rtol=0), (marks,expected_xyz)
        assert marks[0]['xyz_nm'][2] == expected_xyz[2]
        evaluate("document.getElementById('show-parent').click();H01_REVIEW.pendingMesh.then(()=>true)")
        assert evaluate('document.getElementById("mesh-plot").data[0].visible===false')
        evaluate("document.getElementById('show-parent').click();H01_REVIEW.pendingMesh.then(()=>true)")
        evaluate("document.getElementById('export-json').click();document.getElementById('export-csv').click();true")
        for _ in range(80):
            if len(list(downloads.glob('*.json')))==1 and len(list(downloads.glob('*.csv')))==1:break
            time.sleep(.1)
        else:raise AssertionError('JSON/CSV buttons did not download')
        exported=json.loads(next(downloads.glob('*.json')).read_text())
        assert len(exported['cases'])==24 and exported['cases'][0]['annotation']['marks']==marks
        import csv
        with next(downloads.glob('*.csv')).open(encoding='utf-8-sig',newline='') as f:csvrows=list(csv.DictReader(f))
        assert len(csvrows)==24 and csvrows[0]['comment']==expected['comment']
        snapshot=evaluate('JSON.stringify(H01_REVIEW.payload().cases)')
        invalid=copy.deepcopy(exported);invalid['cases'][0]['target_sha256']='wrong'
        error=evaluate('(()=>{try{H01_REVIEW.importPayload('+json.dumps(invalid)+');return false;}catch(e){return true;}})()')
        assert error and evaluate('JSON.stringify(H01_REVIEW.payload().cases)')==snapshot
        evaluate('H01_REVIEW.importPayload('+json.dumps(exported)+')')
        command('Page.reload');ready()
        reloaded=evaluate('H01_REVIEW.answers.C01')
        for k in ['anatomy','quality','confidence','em_readability','comment']:assert reloaded[k]==expected[k],k
        assert reloaded['marks']==marks
        # Do not leave synthetic human annotations in the delivered artifact.
        evaluate("document.getElementById('anatomy').value='';document.getElementById('confidence').value='';document.getElementById('em-readability').value='';document.getElementById('comment').value='';document.querySelectorAll('.quality').forEach(e=>e.checked=false);document.getElementById('remove-mark').click();H01_REVIEW.saveForm();true")
        screenshot=command('Page.captureScreenshot',dict(format='png',captureBeyondViewport=False))
        (out/'browser_preview.png').write_bytes(base64.b64decode(screenshot['data']))
        # Also verify the responsive layout and preserve a visual sample.
        command('Emulation.setDeviceMetricsOverride',dict(width=760,height=1100,deviceScaleFactor=1,mobile=False))
        evaluate("new Promise(r=>setTimeout(r,400))")
        assert evaluate('document.documentElement.scrollWidth<=window.innerWidth')
        screenshot=command('Page.captureScreenshot',dict(format='png',captureBeyondViewport=False))
        (out/'browser_preview_narrow.png').write_bytes(base64.b64decode(screenshot['data']))
        exceptions=[e for e in events if e.get('method')=='Runtime.exceptionThrown'];assert not exceptions,exceptions[:2]
        failures=[e for e in events if e.get('method')=='Network.loadingFailed'];assert not failures,failures[:2]
        result=dict(status='passed',cases_checked=args.limit,real_em_slices_checked=checked_frames,
                    coordinate_axes_and_voxel_centres=True,mesh_vertex_and_face_identity=True,
                    works_offline_file_url=True,slice_navigation=True,plane_alignment=True,
                    uncertain_annotation=True,mark_nm_coordinates=True,export_json_csv=True,
                    import_and_mismatch_rejection=True,reload_persistence=True,
                    responsive_no_horizontal_overflow=True,uncaught_exceptions=0)
        (out/'browser_verification.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
    finally:
        if ws:ws.close()
        process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        log.close()
        # Chrome's utility processes can finish writing after its parent exits.
        shutil.rmtree(tmp, ignore_errors=True)


if __name__=='__main__':main()
