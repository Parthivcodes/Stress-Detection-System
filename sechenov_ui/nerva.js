/* Procedural conceptual anatomy. Highlight color denotes a selected input group. */
(() => {
  'use strict';
  const host = document.getElementById('scene');
  const note = document.getElementById('system-note');
  const notes = {
    cardio:'Cardiovascular context: heart rate and blood pressure are inputs to the stress models.',
    sleep:'Sleep context: reported duration, quality and sleep-disorder category are model inputs.',
    activity:'Movement context: daily steps and physical activity describe the input profile.',
    all:'Full-body overview: a conceptual map of the input groups. This is not a medical scan.'
  };
  let active = 'cardio', highlight = () => {}, redraw = () => {};
  document.querySelectorAll('[data-system]').forEach(button => {
    button.addEventListener('click', () => {
      active = button.dataset.system;
      document.querySelectorAll('[data-system]').forEach(item => {
        item.classList.toggle('active',item.dataset.system===active);
        item.setAttribute('aria-pressed',String(item.dataset.system===active));
      });
      note.textContent=notes[active];highlight();redraw();
    });
    button.setAttribute('aria-pressed',String(button.dataset.system===active));
    button.classList.toggle('active',button.dataset.system===active);
  });
  function updateCards() {
    const specs=[['hr','heart_rate',55,110,'bpm'],['sleep','sleep_duration',4,10,'hours'],['steps','daily_steps',0,15000,'steps']];
    specs.forEach(([id,input,min,max,unit])=>{
      const value=Number(document.getElementById(input).value);
      const label=document.getElementById('metric-'+id);
      label.replaceChildren(document.createTextNode(value.toLocaleString()));
      const small=document.createElement('small');small.textContent=unit;label.append(small);
      document.getElementById('track-'+id).style.left=((value-min)/(max-min)*100)+'%';
    });
  }
  document.getElementById('profile').addEventListener('input',updateCards);
  document.querySelectorAll('[data-preset]').forEach(button=>button.addEventListener('click',updateCards));
  updateCards();
  try {
    if(!window.THREE)throw Error('WebGL unavailable');
    const T=window.THREE;
    const scene=new T.Scene();
    const camera=new T.PerspectiveCamera(37,1,.1,100);
    camera.position.set(0,.25,10.5);
    const renderer=new T.WebGLRenderer({alpha:true,antialias:true,powerPreference:'low-power'});
    renderer.setPixelRatio(Math.min(devicePixelRatio,1.6));
    host.prepend(renderer.domElement);
    host.querySelector('.anatomy-fallback')?.remove();
    const body=new T.Group();scene.add(body);
    const base=new T.LineBasicMaterial({color:0xd0ddcc,transparent:true,opacity:.15});
    const bones=new T.LineBasicMaterial({color:0xe5ebda,transparent:true,opacity:.43});
    const regions={cardio:[],sleep:[],activity:[]};
    function ellipsoid(x,y,z,sx,sy,sz,detail=2,region=null){
      const geometry=new T.IcosahedronGeometry(1,detail);
      const material=region?new T.LineBasicMaterial({color:0xd0ddcc,transparent:true,opacity:.19}):base;
      const mesh=new T.LineSegments(new T.WireframeGeometry(geometry),material);
      mesh.position.set(x,y,z);mesh.scale.set(sx,sy,sz);body.add(mesh);
      if(region)regions[region].push(material);
      return mesh;
    }
    function curve(points,material=bones){
      const path=new T.CatmullRomCurve3(points.map(p=>new T.Vector3(...p)));
      const line=new T.Line(new T.BufferGeometry().setFromPoints(path.getPoints(60)),material);
      body.add(line);return line;
    }
    function limb(from,to,width,region){
      const a=new T.Vector3(...from),b=new T.Vector3(...to),middle=a.clone().add(b).multiplyScalar(.5);
      const mesh=ellipsoid(middle.x,middle.y,middle.z,width,a.distanceTo(b)/2,width*.85,2,region);
      mesh.quaternion.setFromUnitVectors(new T.Vector3(0,1,0),b.clone().sub(a).normalize());
      curve([from,to]);
    }
    // Head, jaw, neck and shoulder girdle.
    ellipsoid(0,2.5,0,.49,.61,.44,2,'sleep');
    ellipsoid(0,2.14,.12,.32,.28,.32,1);
    ellipsoid(0,1.83,0,.2,.28,.2,1);
    ellipsoid(0,.65,0,.87,1.02,.43,2);
    ellipsoid(0,-.51,0,.64,.44,.36,2);
    // Spine, sternum and paired ribs.
    curve([[0,1.7,-.19],[0,.65,-.28],[0,-.55,-.12]],bones);
    curve([[0,1.52,.35],[0,.6,.43],[0,.05,.27]],bones);
    for(let i=0;i<19;i++)ellipsoid(0,1.7-i*.12,-.2,.09,.045,.09,0);
    for(let i=0;i<9;i++){
      const y=1.45-i*.15,span=.48+Math.sin((i+1)/10*Math.PI)*.33;
      [-1,1].forEach(side=>curve([[0,y,-.22],[side*span*.75,y+.04,-.3],[side*span,y-.05,0],[side*span*.7,y-.15,.35],[0,y-.18,.38]]));
    }
    [-1,1].forEach(side=>{
      curve([[0,1.52,.2],[side*.48,1.62,.12],[side*.94,1.51,0]]);
      ellipsoid(side*.93,1.43,0,.26,.28,.26,1,'activity');
      limb([side*.96,1.42,0],[side*1.24,.4,.04],.19,'activity');
      ellipsoid(side*1.24,.36,.04,.16,.16,.16,1);
      limb([side*1.24,.35,.04],[side*1.38,-.48,.12],.14,'activity');
      ellipsoid(side*1.4,-.7,.12,.13,.24,.07,1,'activity');
      limb([side*.36,-.68,0],[side*.42,-1.81,.03],.25,'activity');
      ellipsoid(side*.42,-1.85,.05,.19,.18,.19,1);
      limb([side*.42,-1.96,.03],[side*.46,-2.94,.1],.16,'activity');
      ellipsoid(side*.47,-3.08,.22,.18,.13,.32,1,'activity');
    });
    // Transparent chest volumes indicate cardiovascular context, not pathology.
    ellipsoid(-.35,.95,.1,.27,.51,.25,2,'cardio');
    ellipsoid(.35,.95,.1,.27,.51,.25,2,'cardio');
    const heart=ellipsoid(.17,.8,.43,.17,.23,.15,2,'cardio');heart.rotation.z=-.35;
    const heartFill=new T.Mesh(new T.IcosahedronGeometry(.17,2),new T.MeshBasicMaterial({color:0xe59885,transparent:true,opacity:.18}));
    heartFill.position.set(.17,.8,.43);heartFill.scale.y=1.4;body.add(heartFill);
    highlight=()=>{
      Object.entries(regions).forEach(([name,materials])=>materials.forEach(material=>{
        const selected=active===name||active==='all';
        material.color.set(selected?(name==='cardio'?0xe8a496:0xd3e995):0xc0d0bc);
        material.opacity=selected?.48:.12;
      }));
      heartFill.visible=active==='cardio'||active==='all';
    };
    highlight();
    let yaw=-.12,pitch=0,drag=false,lastX=0,lastY=0,time=0,visible=true;
    let zoom=10.5;
    redraw=()=>{
      body.rotation.y=yaw+(paused?0:Math.sin(time*.2)*.09);
      body.rotation.x=pitch;
      const pulse=1+(paused?0:Math.sin(time*Number(document.getElementById('heart_rate').value)/60*Math.PI*2)*.045);
      heartFill.scale.set(pulse,pulse*1.4,pulse);
      renderer.render(scene,camera);
    };
    host.addEventListener('pointerdown',e=>{
      if(e.target!==renderer.domElement||e.pointerType!=='mouse')return;
      drag=true;lastX=e.clientX;lastY=e.clientY;host.setPointerCapture(e.pointerId);
    });
    host.addEventListener('pointermove',e=>{
      if(!drag)return;yaw+=(e.clientX-lastX)*.008;pitch=Math.max(-.4,Math.min(.4,pitch+(e.clientY-lastY)*.004));
      lastX=e.clientX;lastY=e.clientY;redraw();
    });
    host.addEventListener('pointerup',()=>drag=false);host.addEventListener('pointercancel',()=>drag=false);
    const resize=()=>{
      const w=host.clientWidth,h=host.clientHeight;renderer.setSize(w,h);camera.aspect=w/h;
      camera.position.z=zoom+(camera.aspect<.65?1.5:0);camera.updateProjectionMatrix();redraw();
    };
    document.getElementById('zoom-in').addEventListener('click',()=>{zoom=Math.max(7.5,zoom-.7);resize();});
    document.getElementById('zoom-out').addEventListener('click',()=>{zoom=Math.min(15,zoom+.7);resize();});
    document.getElementById('reset-view').addEventListener('click',()=>{yaw=-.12;pitch=0;zoom=10.5;time=0;resize();});
    document.getElementById('motion').addEventListener('click',redraw);
    new ResizeObserver(resize).observe(host);
    new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;if(visible)redraw();}).observe(host);
    resize();let previous=0;
    function frame(now){requestAnimationFrame(frame);const dt=Math.min((now-previous)/1000,.05);previous=now;if(document.hidden||!visible||paused)return;time+=dt;redraw();}
    requestAnimationFrame(frame);
    renderer.domElement.addEventListener('webglcontextlost',event=>{event.preventDefault();visible=false;note.textContent='3D rendering paused. Reload to restore the anatomy view; model analysis remains available.';});
  }catch(error){
    host.querySelector('canvas')?.remove();
    ['zoom-in','zoom-out','motion','reset-view'].forEach(id=>document.getElementById(id).disabled=true);
    note.textContent='Static anatomy preview. Your browser could not initialize 3D; model analysis remains available.';
  }
})();
