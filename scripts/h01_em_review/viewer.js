'use strict';
(() => {
  const index = window.REVIEW_INDEX;
  const $ = id => document.getElementById(id);
  const storageKey = 'h01-em-review:' + index.manifest_id;
  const colors = {target:'#ffb134', parent:'#52d7ea', syn:'#ff66b8', mark:'#baff72'};
  const qualityValues = new Set(Array.from(document.querySelectorAll('.quality'), e => e.value));
  const anatomyValues = new Set(['','one_spine','multiple_spines','shaft_bump','uncertain']);
  const confidenceValues = new Set(['','high','medium','low']);
  const readabilityValues = new Set(['','helpful','difficult','unreadable']);
  let answers = {}, currentIndex = -1, current = null, frame = 0, activeImage = null;
  let generation = 0, frameGeneration = 0, marking = false, storageAvailable = true;
  let view = {zoom:1, cx:0, cy:0};
  let pendingCase = Promise.resolve(), pendingFrame = Promise.resolve(), pendingMesh = Promise.resolve();
  const imageCache = new Map();
  const blank = () => ({anatomy:'', quality:[], confidence:'', em_readability:'', comment:'',
                       marks:[], visited_slices:[], last_slice:null, updated_at:null});
  const answer = id => answers[id] || (answers[id] = blank());
  const escapeHTML = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  function toast(message) {
    $('toast').textContent = message; $('toast').hidden = false;
    clearTimeout(toast.timer); toast.timer = setTimeout(() => {$('toast').hidden = true;}, 5500);
  }
  function validateAnswers(rows) {
    if (!Array.isArray(rows)) throw Error('В файле нет списка ответов.');
    const result = {}, known = new Map(index.cases.map(c => [c.id,c]));
    for (const row of rows) {
      const c = known.get(row.case_id), a = row.annotation;
      if (!c || result[c.id] || row.target_sha256 !== c.target_sha256 || row.parent_sha256 !== c.parent_sha256)
        throw Error('Идентификатор или меш примера не совпадает с этим набором.');
      if (!a || !anatomyValues.has(a.anatomy) || !Array.isArray(a.quality) || !a.quality.every(v => qualityValues.has(v)) ||
          !confidenceValues.has(a.confidence) || !readabilityValues.has(a.em_readability) || typeof a.comment !== 'string')
        throw Error('Некорректный формат оценки.');
      if (!Array.isArray(a.marks) || !a.marks.every(m => ['neck_base','missing_head','other'].includes(m.kind) &&
          Array.isArray(m.xyz_nm) && m.xyz_nm.length === 3 && m.xyz_nm.every(Number.isFinite) && Number.isInteger(m.slice_index) && m.slice_index >= 0) ||
          !Array.isArray(a.visited_slices) || !a.visited_slices.every(k => Number.isInteger(k) && k >= 0))
        throw Error('Некорректные метки или срезы.');
      result[c.id] = {...blank(), ...JSON.parse(JSON.stringify(a))};
    }
    return result;
  }
  function payload() {
    return {schema:'h01-em-annotations/1', manifest_id:index.manifest_id,
            exported_at:new Date().toISOString(), dataset:'H01', image_source:index.image_source,
            resolution_nm:index.resolution_nm, annotations_are:'Human judgments, uncertainty permitted',
            cases:index.cases.map(c => ({case_id:c.id, key:c.key, audit_id:c.audit_id, raw_id:c.raw_id,
                part_id:c.part_id, final_index:c.final_index, target_sha256:c.target_sha256,
                parent_sha256:c.parent_sha256, annotation:answer(c.id)}))};
  }
  function store() {
    try {
      localStorage.setItem(storageKey, JSON.stringify(payload()));
      storageAvailable = true; $('save-status').classList.remove('error');
      $('save-status').textContent = 'Ответы сохранены в этом браузере. JSON — переносимая копия.';
    } catch (error) {
      storageAvailable = false; $('save-status').classList.add('error');
      $('save-status').textContent = 'Браузер не сохраняет ответы автоматически. Сохрани файл JSON до закрытия.';
    }
  }
  try {
    const saved = localStorage.getItem(storageKey);
    if (saved) {
      const data = JSON.parse(saved);
      if (data.manifest_id !== index.manifest_id || data.schema !== 'h01-em-annotations/1') throw Error('Не совпадает набор.');
      answers = validateAnswers(data.cases);
    }
  } catch (error) {toast('Не удалось прочитать сохранённые ответы. Можно загрузить резервный JSON.');}
  function renderList() {
    const assessed = index.cases.filter(c => answer(c.id).anatomy !== '').length;
    const uncertain = index.cases.filter(c => answer(c.id).anatomy === 'uncertain').length;
    $('progress').textContent = assessed + ' / ' + index.cases.length;
    $('progress').title = 'Оценено: ' + assessed + '; из них «не уверен»: ' + uncertain;
    $('case-list').replaceChildren(...index.cases.map((c,n) => {
      const a = answer(c.id), btn = document.createElement('button');
      btn.className = 'case-button' + (n === currentIndex ? ' active' : '') +
          (a.anatomy === 'uncertain' ? ' unsure' : a.anatomy ? ' answered' : '');
      btn.innerHTML = '<span>Пример '+String(n+1).padStart(2,'0')+'<small>'+c.key.split('_')[1]+'</small></span><span class="state">'+
          (a.anatomy === 'uncertain' ? '?' : a.anatomy ? '✓' : '○')+'</span>';
      btn.title = c.key + ', raw ' + c.raw_id;
      btn.setAttribute('aria-current', n === currentIndex ? 'true' : 'false');
      btn.onclick = () => {pendingCase = selectCase(n);};
      return btn;
    }));
    $('prev-case').disabled = currentIndex <= 0; $('next-case').disabled = currentIndex >= index.cases.length-1;
  }
  function setFormDisabled(disabled) {
    for (const e of document.querySelectorAll('.assessment input,.assessment select,.assessment textarea,.assessment button')) e.disabled=disabled;
  }
  function restoreForm() {
    const a = current ? answer(current.id) : blank();
    $('anatomy').value=a.anatomy; $('confidence').value=a.confidence;
    $('em-readability').value=a.em_readability; $('comment').value=a.comment;
    document.querySelectorAll('.quality').forEach(e => {e.checked=a.quality.includes(e.value);});
  }
  function saveForm() {
    if (!current) return;
    Object.assign(answer(current.id), {anatomy:$('anatomy').value,
      quality:Array.from(document.querySelectorAll('.quality:checked'), e => e.value),
      confidence:$('confidence').value, em_readability:$('em-readability').value,
      comment:$('comment').value, updated_at:new Date().toISOString()});
    store(); renderList();
  }
  for (const id of ['anatomy','confidence','em-readability','comment']) $(id).addEventListener('input',saveForm);
  document.querySelectorAll('.quality').forEach(e => e.addEventListener('change',() => {
    if (e.checked) {
      document.querySelectorAll('.quality').forEach(other => {
        if (other !== e && (['looks_ok','uncertain'].includes(e.value) || ['looks_ok','uncertain'].includes(other.value))) other.checked=false;
      });
    }
    saveForm();
  }));
  function loadCase(c) {
    if (window.H01_CASES?.[c.id]) return Promise.resolve(window.H01_CASES[c.id]);
    return new Promise((resolve,reject) => {
      const script=document.createElement('script'); script.src=c.path+'/data.js';
      script.onload=()=>{script.remove(); const data=window.H01_CASES?.[c.id];
        if (!data || data.target_sha256!==c.target_sha256 || data.parent_sha256!==c.parent_sha256) reject(Error('Не совпали данные примера.'));
        else resolve(data);};
      script.onerror=()=>{script.remove(); reject(Error('Не удалось открыть '+c.path+'/data.js. Сохрани папки cases и assets рядом с HTML.'));};
      document.head.appendChild(script);
    });
  }
  async function selectCase(n) {
    if (n<0 || n>=index.cases.length) return;
    saveForm(); const ticket=++generation; ++frameGeneration; currentIndex=n; current=null; activeImage=null;
    setFormDisabled(true); restoreForm(); renderList(); drawBoth();
    $('mesh-plot').style.opacity='0.2';$('mesh-plot').style.pointerEvents='none';
    $('case-title').textContent='Пример '+String(n+1).padStart(2,'0')+' / '+index.cases.length;
    $('identity').textContent=index.cases[n].key+' · raw '+index.cases[n].raw_id;
    $('frame-status').textContent='Загрузка фрагмента…'; $('slice-label').textContent=''; $('visited').textContent='';
    try {
      const c=await loadCase(index.cases[n]); if (ticket!==generation) return;
      current=c; view={zoom:1,cx:c.shape_xyz[0]/2,cy:c.shape_xyz[1]/2};
      $('contrast').value='100'; marking=false; updateMarkMode(); setFormDisabled(false); restoreForm();
      $('slice').max=c.sections.length-1;
      const stored=answer(c.id).last_slice;
      frame=Number.isInteger(stored) && stored>=0 && stored<c.sections.length ? stored : c.initial_slice;
      renderSources(); updateMarkCount();
      pendingMesh=renderMesh(); pendingFrame=requestFrame(frame);
      await Promise.all([pendingMesh,pendingFrame]);
      if(ticket===generation){$('mesh-plot').style.opacity='1';$('mesh-plot').style.pointerEvents='auto';}
    } catch (error) {if (ticket===generation) {$('frame-status').textContent=error.message;}}
  }
  function loadImage(c,k) {
    const key=c.id+'/'+k;
    if (imageCache.has(key)) return imageCache.get(key);
    const p=new Promise((resolve,reject) => {const img=new Image();
      img.onload=()=>resolve(img); img.onerror=()=>reject(Error('Не удалось открыть EM-срез '+k));
      img.src=c.path+'/slices/'+String(k).padStart(3,'0')+'.png';});
    imageCache.set(key,p);
    if (imageCache.size>16) imageCache.delete(imageCache.keys().next().value);
    return p;
  }
  async function requestFrame(k) {
    if (!current) return;
    k=Math.max(0,Math.min(current.sections.length-1,k)); frame=k; const c=current, ticket=++frameGeneration;
    $('slice').value=k; const s=c.sections[k];
    $('slice-label').textContent='Срез '+(k+1)+' / '+c.sections.length+' · z = '+(s.z_nm/1000).toFixed(3)+' мкм · воксель '+s.z_voxel;
    $('prev-slice').disabled=k===0; $('next-slice').disabled=k===c.sections.length-1;
    $('frame-status').textContent=''; activeImage=null; drawBoth();
    try {
      const img=await loadImage(c,k);
      if (ticket!==frameGeneration || current!==c) return;
      activeImage=img; const a=answer(c.id); a.last_slice=k;
      if (!a.visited_slices.includes(k)) a.visited_slices.push(k);
      a.visited_slices.sort((a,b)=>a-b); store();
      $('visited').textContent='Открыто срезов: '+a.visited_slices.length+' / '+c.sections.length;
      drawBoth(); updatePlane();
      for (const next of [k-1,k+1]) if(next>=0&&next<c.sections.length) loadImage(c,next).catch(()=>{});
    } catch(error) {if (ticket===frameGeneration) $('frame-status').textContent=error.message;}
  }
  function transform(canvas) {
    const w=canvas.clientWidth,h=canvas.clientHeight;
    const scale=Math.min(w/current.shape_xyz[0],h/current.shape_xyz[1])*view.zoom;
    return {w,h,scale,tx:w/2-view.cx*scale,ty:h/2-view.cy*scale};
  }
  function pointAt(canvas,x,y) {
    const t=transform(canvas);return [(x-t.tx)/t.scale, current.shape_xyz[1]-(y-t.ty)/t.scale];
  }
  function draw(canvas,overlay) {
    const dpr=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;
    if (canvas.width!==Math.round(w*dpr) || canvas.height!==Math.round(h*dpr)) {canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}
    const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
    ctx.fillStyle='#12212a';ctx.fillRect(0,0,w,h);
    if (!current || !activeImage) {ctx.fillStyle='#b2c3ca';ctx.font='14px system-ui';ctx.fillText('Загрузка EM…',20,35);return;}
    const t=transform(canvas),nx=current.shape_xyz[0],ny=current.shape_xyz[1];
    ctx.imageSmoothingEnabled=false;ctx.filter='contrast('+$('contrast').value+'%)';
    ctx.drawImage(activeImage,t.tx,t.ty,nx*t.scale,ny*t.scale);ctx.filter='none';
    const screen=(x,y)=>[t.tx+x*t.scale,t.ty+(ny-y)*t.scale];
    if(overlay) {
      const sections=current.sections[frame];
      for(const [kind,checkbox] of [['parent','show-parent'],['target','show-target']]) if($(checkbox).checked) {
        ctx.lineWidth=kind==='target'?1.9:1.15;ctx.strokeStyle=colors[kind];
        for (const line of sections[kind]) {ctx.beginPath();line.forEach((p,i)=>{const q=screen(...p);i?ctx.lineTo(...q):ctx.moveTo(...q);});ctx.stroke();}
      }
      if($('show-syn').checked) current.synapses.forEach(syn=>{
        if(Math.abs(syn.xyz_nm[2]-sections.z_nm)>current.resolution_nm[2]/2) return;
        const p=screen((syn.xyz_nm[0]-current.origin_nm[0])/current.resolution_nm[0],(syn.xyz_nm[1]-current.origin_nm[1])/current.resolution_nm[1]);
        ctx.strokeStyle='#24142b';ctx.lineWidth=2;ctx.fillStyle=colors.syn;ctx.beginPath();ctx.arc(...p,5,0,Math.PI*2);ctx.fill();ctx.stroke();
      });
      answer(current.id).marks.filter(m=>m.slice_index===frame).forEach((m,i)=>{
        const p=screen((m.xyz_nm[0]-current.origin_nm[0])/current.resolution_nm[0],(m.xyz_nm[1]-current.origin_nm[1])/current.resolution_nm[1]);
        ctx.strokeStyle=colors.mark;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(p[0]-6,p[1]);ctx.lineTo(p[0]+6,p[1]);ctx.moveTo(p[0],p[1]-6);ctx.lineTo(p[0],p[1]+6);ctx.stroke();
      });
    }
    const barNm=500,bar=barNm/current.resolution_nm[0]*t.scale;
    if(bar<w-50) {
      ctx.fillStyle='#000a';ctx.fillRect(14,h-48,bar+20,37);ctx.strokeStyle='white';ctx.lineWidth=3;
      ctx.beginPath();ctx.moveTo(24,h-20);ctx.lineTo(24+bar,h-20);ctx.stroke();ctx.fillStyle='white';ctx.font='12px system-ui';ctx.fillText('500 нм',24,h-30);
    }
  }
  function drawBoth(){draw($('em-raw'),false);draw($('em-overlay'),true);}
  new ResizeObserver(()=>{drawBoth();if($('mesh-plot').data) Plotly.Plots.resize($('mesh-plot'));}).observe(document.querySelector('main'));
  for(const canvas of [$('em-raw'),$('em-overlay')]) {
    let drag=null;
    canvas.addEventListener('wheel',e=>{if(!current)return;e.preventDefault();
      const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top,t=transform(canvas);
      const px=(x-t.tx)/t.scale,py=(y-t.ty)/t.scale;
      view.zoom=Math.max(0.7,Math.min(12,view.zoom*Math.exp(-e.deltaY*.002)));
      const nt=transform(canvas);view.cx=px-(x-nt.w/2)/nt.scale;view.cy=py-(y-nt.h/2)/nt.scale;drawBoth();
    },{passive:false});
    canvas.addEventListener('pointerdown',e=>{if(!current)return;const r=canvas.getBoundingClientRect();
      drag={x:e.clientX,y:e.clientY,cx:view.cx,cy:view.cy,scale:transform(canvas).scale,
        localX:e.clientX-r.left,localY:e.clientY-r.top,mark:marking&&canvas.id==='em-overlay'};canvas.setPointerCapture(e.pointerId);});
    canvas.addEventListener('pointermove',e=>{if(!drag || drag.mark)return;
      view.cx=drag.cx-(e.clientX-drag.x)/drag.scale;view.cy=drag.cy-(e.clientY-drag.y)/drag.scale;drawBoth();});
    canvas.addEventListener('pointerup',e=>{if(drag?.mark&&Math.hypot(e.clientX-drag.x,e.clientY-drag.y)<6) {
        const p=pointAt(canvas,drag.localX,drag.localY);
        if(p[0]>=0&&p[0]<current.shape_xyz[0]&&p[1]>=0&&p[1]<current.shape_xyz[1]) addMark(p);
      } drag=null;});
    canvas.addEventListener('pointercancel',()=>{drag=null;});
  }
  function addMark(pixelXY) {
    if(!current || !activeImage) return;
    const s=current.sections[frame];
    answer(current.id).marks.push({kind:$('mark-kind').value,slice_index:frame,z_voxel:s.z_voxel,
      xyz_nm:[current.origin_nm[0]+pixelXY[0]*current.resolution_nm[0],current.origin_nm[1]+pixelXY[1]*current.resolution_nm[1],s.z_nm],
      created_at:new Date().toISOString()});
    marking=false;updateMarkMode();updateMarkCount();store();drawBoth();update3DMarks();
  }
  function updateMarkMode(){
    $('mark-mode').setAttribute('aria-pressed',String(marking));$('em-overlay').classList.toggle('marking',marking);
    $('mark-mode').textContent=marking?'Кликни на правом EM-срезе':'Поставить метку';
  }
  function updateMarkCount(){const a=current?answer(current.id):blank();$('mark-count').textContent='Меток: '+a.marks.length;}
  function meshTrace(mesh,name,color,opacity) {
    return {type:'mesh3d',x:mesh.vertices.map(v=>v[0]),y:mesh.vertices.map(v=>v[1]),z:mesh.vertices.map(v=>v[2]),
      i:mesh.faces.map(f=>f[0]),j:mesh.faces.map(f=>f[1]),k:mesh.faces.map(f=>f[2]),
      name,color,opacity,flatshading:true,hoverinfo:'skip',showlegend:false,
      lighting:{ambient:.65,diffuse:.7,specular:.05,roughness:1},lightposition:{x:100,y:-200,z:300}};
  }
  function planeTrace() {
    const dims=current.shape_xyz.map((n,i)=>n*current.resolution_nm[i]/1000),z=(current.sections[frame].z_nm-current.origin_nm[2])/1000;
    return {type:'mesh3d',x:[0,dims[0],dims[0],0],y:[0,0,dims[1],dims[1]],z:[z,z,z,z],
      i:[0,0],j:[1,2],k:[2,3],color:'#b6d8dc',opacity:.12,hoverinfo:'skip',showscale:false,name:'EM-плоскость',showlegend:false};
  }
  function marksTrace(){const marks=answer(current.id).marks.map(m=>m.xyz_nm.map((v,i)=>(v-current.origin_nm[i])/1000));
    return {type:'scatter3d',mode:'markers',x:marks.map(p=>p[0]),y:marks.map(p=>p[1]),z:marks.map(p=>p[2]),
      marker:{size:4,color:colors.mark},hoverinfo:'skip',showlegend:false,name:'Твои метки'};}
  const camera=()=>({eye:{x:1.65,y:-1.65,z:1.2},up:{x:0,y:0,z:1},center:{x:0,y:0,z:0}});
  async function renderMesh(){
    const c=current,dims=c.shape_xyz.map((n,i)=>n*c.resolution_nm[i]/1000);
    const syn=c.synapses.map(s=>s.xyz_nm.map((v,i)=>(v-c.origin_nm[i])/1000));
    const synTrace={type:'scatter3d',mode:'markers',x:syn.map(p=>p[0]),y:syn.map(p=>p[1]),z:syn.map(p=>p[2]),
      text:c.synapses.map(s=>'Контакт '+s.id+' · '+s.kind_label),hovertemplate:'%{text}<extra></extra>',
      marker:{size:4,color:colors.syn},name:'Контакты',showlegend:false,visible:$('show-syn').checked};
    const data=[meshTrace(c.parent_mesh,'Ветвь','#a4b3bd',.28),meshTrace(c.target_mesh,'Кандидат',colors.target,1),planeTrace(),synTrace,marksTrace()];
    data[0].visible=$('show-parent').checked;data[1].visible=$('show-target').checked;
    await Plotly.react('mesh-plot',data,{margin:{l:0,r:0,t:0,b:0},paper_bgcolor:'#fff',showlegend:false,
      scene:{aspectmode:'data',camera:camera(),xaxis:{title:{text:'x, мкм'},range:[0,dims[0]],backgroundcolor:'#f6f8f8',showbackground:true},
        yaxis:{title:{text:'y, мкм'},range:[0,dims[1]],backgroundcolor:'#f6f8f8',showbackground:true},
        zaxis:{title:{text:'z, мкм'},range:[0,dims[2]],backgroundcolor:'#f6f8f8',showbackground:true}},
      uirevision:c.id},{responsive:true,displaylogo:false,modeBarButtonsToRemove:['toImage','resetCameraLastSave3d']});
  }
  function updatePlane(){if(!current||!$('mesh-plot').data)return;
    const p=planeTrace(),id=current.id;pendingMesh=pendingMesh.then(()=>current?.id===id?Plotly.restyle('mesh-plot',{z:[p.z]},[2]):undefined);}
  function update3DMarks(){if(!current||!$('mesh-plot').data)return;
    const t=marksTrace(),id=current.id;pendingMesh=pendingMesh.then(()=>current?.id===id?Plotly.restyle('mesh-plot',{x:[t.x],y:[t.y],z:[t.z]},[4]):undefined);}
  function renderSources(){const c=current;
    $('source-details').innerHTML='<p>Кандидат: <code>'+escapeHTML(c.key)+' · raw '+c.raw_id+' · part '+c.part_id+' · final '+c.final_index+'</code></p>'+
      '<p>EM: публичный H01, 20210601/4nm_raw, mip 1; разрешение 8 × 8 × 33 нм. Фрагмент: '+c.shape_xyz.join(' × ')+' вокселей. XY без перестановки осей; y направлен вверх. Сечения проходят через центры вокселей.</p>'+
      '<p>Окно отображения: 1–99 перцентили всего фрагмента ('+c.display_range.map(v=>v.toFixed(1)).join('…')+'), одинаковое для всех срезов. Ползунок меняет только отображение. Исходные uint8-интенсивности сохранены в em_volume.npz.</p>'+
      '<p>Начальный срез: максимальная площадь замкнутого сечения кандидата. Кроп охватывает выделенный меш с запасом 800 нм по x/y и 500 нм по z; это не гарантирует охват пропущенных частей за его пределами.</p>'+
      '<p>Локальных перечисленных контактов: '+c.synapses.length+'; всего в доступной части клетки: '+c.total_listed_neuron_contacts+'. Соответствие C3: '+escapeHTML(c.source_c3_id)+'. Использованы постсинаптические координаты в нм. Точки не присвоены шипику.</p>'+
      '<p>SHA256 меша: <code>'+c.target_sha256+'</code><br>SHA256 EM-фрагмента: <code>'+c.image_sha256+'</code></p>'+
      '<p><a href="'+c.path+'/target_mesh.off">Меш кандидата</a> · <a href="'+c.path+'/parent_roi.off">Фрагмент ветви</a> · <a href="'+c.path+'/em_volume.npz">Исходный EM-фрагмент</a> · <a href="manifest.json">Манифест набора</a></p>';
  }
  $('slice').addEventListener('input',()=>{pendingFrame=requestFrame(Number($('slice').value));});
  $('prev-slice').onclick=()=>{pendingFrame=requestFrame(frame-1);};$('next-slice').onclick=()=>{pendingFrame=requestFrame(frame+1);};
  $('reset-slice').onclick=()=>{if(current)pendingFrame=requestFrame(current.initial_slice);};
  $('prev-case').onclick=()=>{pendingCase=selectCase(currentIndex-1);};$('next-case').onclick=()=>{pendingCase=selectCase(currentIndex+1);};
  $('save-next').onclick=()=>{saveForm();pendingCase=selectCase(Math.min(currentIndex+1,index.cases.length-1));};
  $('next-unreviewed').onclick=()=>{saveForm();for(let offset=1;offset<=index.cases.length;offset++){
    const n=(currentIndex+offset)%index.cases.length;if(!answer(index.cases[n].id).anatomy){pendingCase=selectCase(n);return;}}
    toast('У всех примеров указана анатомическая оценка, включая «не уверен».');};
  $('reset-zoom').onclick=()=>{if(current){view={zoom:1,cx:current.shape_xyz[0]/2,cy:current.shape_xyz[1]/2};drawBoth();}};
  $('contrast').addEventListener('input',drawBoth);
  for(const [id,trace] of [['show-parent',0],['show-target',1],['show-syn',3]]) $(id).addEventListener('change',()=>{
    drawBoth();if($('mesh-plot').data) pendingMesh=pendingMesh.then(()=>Plotly.restyle('mesh-plot',{visible:$(id).checked},[trace]));});
  $('reset-camera').onclick=()=>{if($('mesh-plot').data)pendingMesh=Plotly.relayout('mesh-plot',{'scene.camera':camera()});};
  $('mark-mode').onclick=()=>{if(!current)return;marking=!marking;updateMarkMode();};
  $('remove-mark').onclick=()=>{if(!current)return;answer(current.id).marks.pop();store();updateMarkCount();drawBoth();update3DMarks();};
  document.addEventListener('keydown',e=>{
    if(['INPUT','TEXTAREA','SELECT','BUTTON'].includes(document.activeElement.tagName) || e.ctrlKey || e.metaKey || e.altKey)return;
    if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();pendingFrame=requestFrame(frame+(e.key==='ArrowLeft'?-1:1));}
  });
  function downloadFile(content,type,extension){const url=URL.createObjectURL(new Blob([content],{type})),a=document.createElement('a');
    a.href=url;a.download='h01_spine_answers_'+new Date().toISOString().replace(/[:.]/g,'-')+'.'+extension;
    document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),10000);}
  $('export-json').onclick=()=>{saveForm();downloadFile(JSON.stringify(payload(),null,2),'application/json;charset=utf-8','json');toast('JSON содержит все ответы, сомнения и метки.');};
  const csvCell=value=>{let text=String(value??'');if(/^[=+@\-]/.test(text))text="'"+text;return '"'+text.replace(/"/g,'""')+'"';};
  $('export-csv').onclick=()=>{saveForm();const rows=[['case_id','key','raw_id','part_id','final_index','anatomy','quality','confidence','em_readability','comment','marks_json','visited_slices','target_sha256']];
    index.cases.forEach(c=>{const a=answer(c.id);rows.push([c.id,c.key,c.raw_id,c.part_id,c.final_index,a.anatomy,a.quality.join('|'),a.confidence,a.em_readability,a.comment,JSON.stringify(a.marks),a.visited_slices.join('|'),c.target_sha256]);});
    downloadFile('\ufeff'+rows.map(r=>r.map(csvCell).join(',')).join('\r\n'),'text/csv;charset=utf-8','csv');};
  function importPayload(data){if(data.schema!=='h01-em-annotations/1'||data.manifest_id!==index.manifest_id)throw Error('Файл относится к другому набору или формату.');
    const validated=validateAnswers(data.cases);answers={...answers,...validated};store();restoreForm();renderList();updateMarkCount();drawBoth();update3DMarks();return Object.keys(validated).length;}
  $('import-json').addEventListener('change',async e=>{const file=e.target.files[0];if(!file)return;
    try{const count=importPayload(JSON.parse(await file.text()));toast('Загружены ответы: '+count+' примеров.');}catch(error){toast('Импорт отклонён: '+error.message);}e.target.value='';});
  // A small public API allows checks in the real browser without duplicating renderer logic.
  window.H01_REVIEW={get current(){return current;},get frame(){return frame;},get answers(){return answers;},get storageAvailable(){return storageAvailable;},
    get pendingCase(){return pendingCase;},get pendingFrame(){return pendingFrame;},get pendingMesh(){return pendingMesh;},
    selectCase(n){pendingCase=selectCase(n);return pendingCase;},requestFrame(k){pendingFrame=requestFrame(k);return pendingFrame;},
    payload,importPayload,pointAt,addMark,saveForm,get view(){return view;}};
  renderList();store();pendingCase=selectCase(0);
})();
