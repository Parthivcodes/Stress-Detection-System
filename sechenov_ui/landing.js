/* NERVA particle study. One renderer, one animation clock, no external requests. */
(() => {
  'use strict';
  const host = document.getElementById('particle-field');
  const fallback = document.getElementById('particle-fallback');
  const formName = document.getElementById('form-name');
  const formIndex = document.getElementById('form-index');
  const progress = document.getElementById('form-progress');
  const motionButton = document.getElementById('particle-motion');
  const selectors = [...document.querySelectorAll('[data-form]')];
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const names = ['Sphere', 'Diamond', 'Helix', 'Torus', 'Wave'];
  const count = matchMedia('(max-width: 600px)').matches ? 13000 : 22000;
  const cycleDuration = 6.5, morphDuration = 1.75;
  let current = 2, target = 2, elapsed = 0, time = 0, morph = 1;
  let isPaused = reduced.matches, isFallback = false, frameId = 0;
  let renderer, scene, camera, cloud, material, geometry;
  let source, destination, seeds, colors, last = 0, nameTimer;
  let width = host.clientWidth, height = host.clientHeight;
  const pointer = {x: 0, y: 0, targetX: 0, targetY: 0, strength: 0, targetStrength: 0};
  // A deterministic distribution keeps morphs coherent and comparisons repeatable.
  let randomState = 91827;
  function random() { randomState = (Math.imul(1664525, randomState) + 1013904223) >>> 0; return randomState / 4294967296; }
  const samples = Array.from({length: count}, () => [random(),random(),random(),random(),random()]);
  const TAU = Math.PI * 2;
  const shapes = names.map((_, shape) => {
    const positions = new Float32Array(count * 3);
    samples.forEach(([u,v,w,jitter], i) => {
      let x, y, z;
      if (shape === 0) {
        const azimuth = TAU*u, cosine = 2*v-1, sine = Math.sqrt(1-cosine*cosine);
        const radius = 2.53 + (w-.5)*.09;
        x=radius*sine*Math.cos(azimuth); y=radius*cosine; z=radius*sine*Math.sin(azimuth);
      } else if (shape === 1) {
        // Two faceted pyramids, with a narrow bright equator like the recording.
        const side=Math.floor(u*4), a=u*4-side;
        const upper=v<.44, h=(v<.12 ? w*.018 : 1-Math.sqrt(w));
        const radius=2.85*(1-h), corners=[[1,0],[0,1],[-1,0],[0,-1],[1,0]];
        x=radius*(corners[side][0]*(1-a)+corners[side+1][0]*a);
        z=radius*(corners[side][1]*(1-a)+corners[side+1][1]*a);
        y=(upper?1.9:-3.6)*h;
        const angle=.42; const px=x; x=px*Math.cos(angle)-z*Math.sin(angle); z=px*Math.sin(angle)+z*Math.cos(angle);
        z+=(jitter-.5)*.025;
      } else if (shape === 2) {
        y=(u-.5)*10.1;
        const angle=y*2.15 + (v>.5 ? Math.PI : 0);
        const radius=1.37+(w-.5)*.07;
        x=Math.sin(angle)*radius; z=Math.cos(angle)*radius;
        y+=(jitter-.5)*.065;
        if(i%23===0){x*=w;z*=w;}
      } else if (shape === 3) {
        const t=u*TAU;
        const center = angle => {
          const r=1.0*(2+Math.cos(3*angle));
          return [r*Math.cos(2*angle),r*Math.sin(2*angle),Math.sin(3*angle)*1.0];
        };
        const p=center(t), q=center(t+.001);
        let tx=q[0]-p[0],ty=q[1]-p[1],tz=q[2]-p[2];
        const tl=Math.hypot(tx,ty,tz);tx/=tl;ty/=tl;tz/=tl;
        let nx=ty,ny=-tx,nz=0;const nl=Math.hypot(nx,ny);nx/=nl;ny/=nl;
        const bx=ty*nz-tz*ny,by=tz*nx-tx*nz,bz=tx*ny-ty*nx;
        const tube=.65+(w-.5)*.035, a=v*TAU;
        x=(p[0]+tube*(Math.cos(a)*nx+Math.sin(a)*bx))*.82;
        y=(p[1]+tube*(Math.cos(a)*ny+Math.sin(a)*by))*.94;
        z=(p[2]+tube*(Math.cos(a)*nz+Math.sin(a)*bz))*.85;
        const yy=y; y=yy*Math.cos(.42)-z*Math.sin(.42);z=yy*Math.sin(.42)+z*Math.cos(.42);
        const xx=x; x=(xx*Math.cos(.4)+z*Math.sin(.4))*.96;z=-xx*Math.sin(.4)+z*Math.cos(.4);
        y=y*.86-.32;
      } else {
        x=(u-.5)*5.7;y=(v-.5)*4.5;
        z=Math.sin(x*1.25+y*.55)*.7+Math.cos(y*1.1)*.42;
        const yy=y; y=yy*.86-z*.4; z=yy*.4+z*.86;
        y+=x*.22-.7;
      }
      positions.set([x,y,z], i*3);
    });
    return positions;
  });
  const easing = t => t*t*(3-2*t);
  function updateLabel() {
    clearTimeout(nameTimer);
    selectors.forEach(button=>button.setAttribute('aria-pressed',String(Number(button.dataset.form)===target)));
    formName.classList.add('changing');
    const commit=()=>{formName.textContent=names[target];formIndex.textContent=`0${target+1} / 05`;formName.classList.remove('changing');};
    if(reduced.matches || isPaused)commit();else nameTimer=setTimeout(commit,190);
  }
  function selectForm(index) {
    if(index===target)return;
    // Snapshot the current interpolation so rapid manual selections never jump.
    if(source){const blend=easing(morph);for(let i=0;i<source.length;i++)source[i]+=(destination[i]-source[i])*blend;}
    current=target;target=index;elapsed=0;morph=isPaused||reduced.matches?1:0;
    if(destination){destination.set(shapes[target]);geometry.attributes.position.needsUpdate=true;geometry.attributes.aTarget.needsUpdate=true;}
    updateLabel();if(isFallback)drawFallback();else render();
  }
  selectors.forEach(button=>button.addEventListener('click',()=>selectForm(Number(button.dataset.form))));
  document.querySelector('.form-selectors').addEventListener('keydown',event=>{
    if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
    event.preventDefault();const index=event.key==='Home'?0:event.key==='End'?4:(target+(event.key==='ArrowRight'?1:4))%5;
    selectForm(index);selectors[index].focus();
  });
  function motionLabel(){motionButton.setAttribute('aria-pressed',String(isPaused));motionButton.setAttribute('aria-label',isPaused?'Play animation':'Pause animation');motionButton.firstElementChild.textContent=isPaused?'▷':'Ⅱ';}
  motionButton.addEventListener('click',()=>{isPaused=!isPaused;motionLabel();last=0;if(!isPaused)startLoop();});
  reduced.addEventListener('change',event=>{isPaused=event.matches;motionLabel();last=0;if(!isPaused)startLoop();else render();});
  motionLabel();
  const dialog=document.getElementById('about-dialog');
  document.getElementById('about-open').addEventListener('click',()=>dialog.showModal());
  document.getElementById('about-close').addEventListener('click',()=>dialog.close());
  dialog.addEventListener('click',event=>{const r=dialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)dialog.close();});
  const experience=document.getElementById('experience');
  experience.addEventListener('pointermove',event=>{
    if(event.pointerType==='touch'||isPaused||reduced.matches)return;
    const r=host.getBoundingClientRect();pointer.targetX=((event.clientX-r.left)/r.width-.5)*2;pointer.targetY=-((event.clientY-r.top)/r.height-.5)*2;pointer.targetStrength=1;
  });
  experience.addEventListener('pointerleave',()=>{pointer.targetStrength=0;pointer.targetX=pointer.targetY=0;});
  function drawFallback(){
    const ctx=fallback.getContext('2d');if(!ctx)return;
    const dpr=Math.min(devicePixelRatio||1,1.5);fallback.width=width*dpr;fallback.height=height*dpr;ctx.scale(dpr,dpr);ctx.clearRect(0,0,width,height);
    const coordinates=shapes[target],scale=height/8.2;
    for(let i=0;i<count;i+=3){const k=i*3;const x=coordinates[k],y=coordinates[k+1],z=coordinates[k+2];const perspective=9.5/(9.5-z);const alpha=.12+samples[i][3]*.58;ctx.fillStyle=`rgba(215,212,198,${alpha})`;ctx.fillRect(width*(width<601?.67:.5)+x*scale*perspective,height*.5-y*scale*perspective,.8,.8);}
  }
  function useFallback(message){
    isFallback=true;cancelAnimationFrame(frameId);frameId=0;
    if(renderer){renderer.domElement.hidden=true;renderer.dispose();}
    fallback.hidden=false;drawFallback();document.getElementById('render-status').textContent=message;
    isPaused=true;motionLabel();motionButton.disabled=true;
  }
  function resize(){
    width=host.clientWidth;height=host.clientHeight;
    if(isFallback){drawFallback();return;}
    if(!renderer)return;
    renderer.setSize(width,height);camera.aspect=width/height;camera.updateProjectionMatrix();
    cloud.position.x=width<601?1.0:0;
    cloud.scale.setScalar(width<601?.82:1);
    material.uniforms.uPixelRatio.value=Math.min(devicePixelRatio||1,1.6);
    renderer.setPixelRatio(material.uniforms.uPixelRatio.value);render();
  }
  function render(){
    if(!renderer||isFallback)return;
    material.uniforms.uTime.value=time;material.uniforms.uMorph.value=easing(morph);
    material.uniforms.uPointer.value.set(pointer.x*camera.aspect*3.5,pointer.y*3.5);
    material.uniforms.uInteraction.value=isPaused?0:pointer.strength;
    material.uniforms.uWave.value=target===4?easing(morph):current===4?1-easing(morph):0;
    material.uniforms.uHelix.value=target===2?easing(morph):current===2?1-easing(morph):0;
    cloud.rotation.y=Math.sin(time*.17)*.3+pointer.x*.11;
    cloud.rotation.x=Math.sin(time*.12)*.06-pointer.y*.055;
    renderer.render(scene,camera);
  }
  function frame(now){
    frameId=0;if(document.hidden||isPaused||isFallback)return;
    const dt=last?Math.min((now-last)/1000,.05):0;last=now;time+=dt;elapsed+=dt;
    morph=Math.min(1,morph+dt/morphDuration);
    const follow=1-Math.exp(-dt*3);
    pointer.x+=(pointer.targetX-pointer.x)*follow;pointer.y+=(pointer.targetY-pointer.y)*follow;pointer.strength+=(pointer.targetStrength-pointer.strength)*follow;
    if(elapsed>=cycleDuration)selectForm((target+1)%5);
    progress.style.transform=`scaleX(${Math.max(0,1-elapsed/cycleDuration)})`;render();frameId=requestAnimationFrame(frame);
  }
  function startLoop(){if(!frameId&&!isPaused&&!isFallback&&!document.hidden)frameId=requestAnimationFrame(frame);}
  document.addEventListener('visibilitychange',()=>{if(document.hidden){cancelAnimationFrame(frameId);frameId=0;last=0;}else startLoop();});
  try{
    if(!window.THREE)throw new Error('Three.js is unavailable');
    const T=window.THREE;
    renderer=new T.WebGLRenderer({alpha:true,antialias:false,powerPreference:'low-power'});
    renderer.setClearColor(0x101010,0);host.append(renderer.domElement);fallback.hidden=true;
    scene=new T.Scene();camera=new T.PerspectiveCamera(45,width/height,.1,40);camera.position.z=9.5;
    geometry=new T.BufferGeometry();source=new Float32Array(shapes[2]);destination=new Float32Array(source);
    seeds=new Float32Array(count);colors=new Float32Array(count*3);
    for(let i=0;i<count;i++){seeds[i]=samples[i][3];colors.set([.88+samples[i][4]*.12,.87+samples[i][4]*.12,.83+samples[i][4]*.14],i*3);}
    geometry.setAttribute('position',new T.BufferAttribute(source,3));geometry.setAttribute('aTarget',new T.BufferAttribute(destination,3));geometry.setAttribute('aSeed',new T.BufferAttribute(seeds,1));geometry.setAttribute('color',new T.BufferAttribute(colors,3));
    material=new T.ShaderMaterial({
      transparent:true,depthWrite:false,depthTest:false,vertexColors:true,blending:T.AdditiveBlending,
      uniforms:{uTime:{value:0},uMorph:{value:1},uPixelRatio:{value:1},uPointer:{value:new T.Vector2()},uInteraction:{value:0},uWave:{value:0},uHelix:{value:1}},
      vertexShader:`
        attribute vec3 aTarget;
        attribute float aSeed;
        uniform float uTime, uMorph, uPixelRatio, uInteraction, uWave, uHelix;
        uniform vec2 uPointer;
        varying vec3 vColor;
        varying float vAlpha;
        void main(){
          vec3 p=mix(position,aTarget,uMorph);
          float scatter=sin(uMorph*3.14159265);
          p+=vec3(sin(aSeed*83.0+uTime*.6),cos(aSeed*117.0+uTime*.4),sin(aSeed*67.0))*scatter*.33;
          p.z+=sin(p.x*1.6+p.y*.7+uTime*.85)*.22*uWave;
          p.x+=sin(p.y*1.6+uTime*.7)*.045*uHelix;
          vec4 world=modelMatrix*vec4(p,1.0);
          vec2 difference=world.xy-uPointer;
          float distanceToPointer=length(difference);
          float force=exp(-distanceToPointer*distanceToPointer*1.4)*uInteraction;
          world.xy+=normalize(difference+vec2(.001))*force*.36;
          world.z+=force*.5;
          vec4 mv=viewMatrix*world;
          gl_Position=projectionMatrix*mv;
          gl_PointSize=clamp((.95+aSeed*.85)*uPixelRatio*(9.5/-mv.z),.8,2.7*uPixelRatio);
          float depth=clamp((world.z+3.1)/6.2,.0,1.0);
          vAlpha=(.32+aSeed*.65)*(.65+depth*.55);
          vColor=color;
          vColor=mix(vColor,mix(vec3(.69,.92,.89),vec3(.98,.94,.73),aSeed),uHelix*.38);
        }`,
      fragmentShader:`
        varying vec3 vColor;
        varying float vAlpha;
        void main(){float d=length(gl_PointCoord-vec2(.5));if(d>.5)discard;float edge=1.0-smoothstep(.3,.5,d);gl_FragColor=vec4(vColor,vAlpha*edge);}`
    });
    cloud=new T.Points(geometry,material);cloud.frustumCulled=false;scene.add(cloud);
    renderer.domElement.addEventListener('webglcontextlost',event=>{event.preventDefault();useFallback('Static preview · 3D context unavailable');});
    resize();startLoop();
  }catch(error){useFallback('Static preview · 3D unavailable');}
  new ResizeObserver(resize).observe(host);
})();
