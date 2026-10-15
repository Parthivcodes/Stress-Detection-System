/* STRAIN: scroll position morphs a 3D thread field. No medical inference here. */
(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const reduce = matchMedia('(prefers-reduced-motion: reduce)');
  const host = byId('tension-scene');
  const chapters = [...document.querySelectorAll('[data-phase]')];
  const slider = byId('tension');
  const motion = byId('story-motion');
  let frozen = reduce.matches, target = 0, current = 0, time = 0;
  let manual = false, dirty = true, onScreen = true;
  let draw = () => {};

  function label() {
    const phase = Math.min(2, Math.round(target * 2));
    byId('stage-count').textContent = `0${phase + 1} / 03`;
    byId('stage-label').textContent = ['OVERLOAD', 'FINDING SPACE', 'A CLEARER PATTERN'][phase];
    slider.value = String(Math.round((1 - target) * 100));
    byId('tension-value').textContent = slider.value + '%';
  }
  function scrollState() {
    manual = false;
    const mobile = innerWidth <= 700;
    const focus = innerHeight * (mobile ? .72 : .5);
    const first = chapters[0].getBoundingClientRect();
    const last = chapters[2].getBoundingClientRect();
    const firstAnchor = first.top + (mobile ? 140 : first.height / 2);
    const lastAnchor = last.top + (mobile ? 140 : last.height / 2);
    target = Math.max(0, Math.min(1, (focus - firstAnchor) / (lastAnchor - firstAnchor)));
    dirty = true;
    label();
  }
  addEventListener('scroll', scrollState, {passive:true});
  addEventListener('resize', scrollState);
  slider.addEventListener('input', () => {
    manual = true;
    target = 1 - Number(slider.value) / 100;
    dirty = true;
    label();
  });
  function motionLabel() {
    motion.textContent = frozen ? 'Resume motion' : 'Pause motion';
    motion.setAttribute('aria-pressed', String(frozen));
  }
  motion.addEventListener('click', () => { frozen = !frozen; dirty = true; motionLabel(); });
  reduce.addEventListener('change', e => { frozen = e.matches; dirty = true; motionLabel(); });
  motionLabel();
  scrollState();

  try {
    if (!window.THREE) throw Error('WebGL unavailable');
    const T = window.THREE;
    const scene = new T.Scene();
    const camera = new T.PerspectiveCamera(40, 1, .1, 60);
    camera.position.set(0, 0, 10);
    const renderer = new T.WebGLRenderer({alpha:true, antialias:true, powerPreference:'low-power'});
    renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
    host.append(renderer.domElement);
    host.classList.add('has-webgl');
    const sculpture = new T.Group();
    scene.add(sculpture);
    const count = innerWidth < 700 ? 42 : 64, samples = 150;
    const paths = [];
    for (let thread = 0; thread < count; thread++) {
      const positions = new Float32Array(samples * 3);
      const tangled = new Float32Array(samples * 3);
      const ordered = new Float32Array(samples * 3);
      const offset = thread / count;
      for (let j = 0; j < samples; j++) {
        const u = j / (samples - 1), angle = u * Math.PI * 4;
        const phi = offset * Math.PI * 2;
        const r = 1.3 + .65 * Math.sin(3 * angle + phi);
        const index = j * 3;
        tangled[index] = r * Math.cos(angle + phi * .4) + .32 * Math.sin(phi * 3);
        tangled[index+1] = r * Math.sin(angle * 1.5 + phi) * .88;
        tangled[index+2] = 1.2 * Math.cos(angle * 1.5 + phi * .7);
        ordered[index] = (u - .5) * 6.1;
        ordered[index+1] = (offset - .5) * 3.4 + .22 * Math.sin(u * Math.PI * 3 + offset * 3);
        ordered[index+2] = .4 * Math.cos(u * Math.PI * 2 + offset * 2);
      }
      positions.set(tangled);
      const geometry = new T.BufferGeometry();
      geometry.setAttribute('position', new T.BufferAttribute(positions, 3).setUsage(T.DynamicDrawUsage));
      const material = new T.LineBasicMaterial({color:new T.Color().setHSL(.025 + offset * .035, .48, .36 + offset * .26), transparent:true, opacity:.72});
      const line = new T.Line(geometry, material);
      line.frustumCulled = false;
      sculpture.add(line);
      paths.push({geometry, positions, tangled, ordered});
    }
    let pointerX = 0, pointerY = 0;
    host.addEventListener('pointermove', e => {
      if (e.pointerType !== 'mouse' || reduce.matches) return;
      const box = host.getBoundingClientRect();
      pointerX = (e.clientX - box.left) / box.width - .5;
      pointerY = (e.clientY - box.top) / box.height - .5;
      dirty = true;
    });
    host.addEventListener('pointerleave', () => {pointerX = pointerY = 0; dirty = true;});
    draw = () => {
      const blend = current * current * (3 - 2 * current);
      paths.forEach(({geometry,positions,tangled,ordered}, i) => {
        for (let j=0; j<samples; j++) {
          const index = j*3;
          for(let axis=0; axis<3; axis++) positions[index+axis] = tangled[index+axis]*(1-blend)+ordered[index+axis]*blend;
          positions[index+1] += Math.sin(j*.06 + time*.7 + i*.1)*.045;
        }
        geometry.attributes.position.needsUpdate = true;
      });
      sculpture.rotation.set(.18+pointerY*.13, -.35*(1-blend)+pointerX*.25, -.25*(1-blend));
      renderer.render(scene, camera);
    };
    const resize = () => {
      const width = host.clientWidth, height = host.clientHeight;
      renderer.setSize(width,height);
      camera.aspect = width/height;
      camera.position.z = camera.aspect<1 ? 11.5 : 9;
      camera.updateProjectionMatrix();
      dirty = true;
      draw();
    };
    new ResizeObserver(resize).observe(host);
    new IntersectionObserver(entries => {onScreen=entries[0].isIntersecting;dirty=true;}).observe(host);
    resize();
    let previous = 0;
    function frame(now) {
      requestAnimationFrame(frame);
      const dt = Math.min((now-previous)/1000,.05);
      previous = now;
      if (document.hidden || !onScreen) return;
      if (!frozen) time += dt;
      const settling = Math.abs(target-current)>.0005;
      current = reduce.matches ? target : current+(target-current)*.09;
      if (!frozen || dirty || settling) {draw();dirty=false;}
    }
    requestAnimationFrame(frame);
    renderer.domElement.addEventListener('webglcontextlost', e => {
      e.preventDefault();onScreen=false;
      renderer.domElement.style.display='none';host.classList.remove('has-webgl');
      motion.disabled=true;motion.textContent='Static artwork';
    });
  } catch (error) {
    host.querySelector('canvas')?.remove();
    host.classList.remove('has-webgl');
    motion.disabled=true;motion.textContent='Static artwork';
    slider.addEventListener('input',()=>{
      host.querySelector('.thread-fallback').style.transform=`rotate(${-30+target*60}deg) scaleX(${1+target*.2})`;
    });
  }

  // Input-linked illustration in the lab; never presented as recorded waveforms.
  const lab=byId('scene');
  const canvas=document.createElement('canvas');
  canvas.setAttribute('aria-hidden','true');lab.append(canvas);
  const ctx=canvas.getContext('2d');
  let phase=0,labVisible=true;
  if (ctx) {
    lab.querySelector('.lab-fallback')?.remove();
    const resize=()=>{canvas.width=lab.clientWidth*devicePixelRatio;canvas.height=lab.clientHeight*devicePixelRatio;};
    new ResizeObserver(resize).observe(lab);resize();
    new IntersectionObserver(entries=>labVisible=entries[0].isIntersecting).observe(lab);
    let previous=0;
    function render(now) {
      requestAnimationFrame(render);
      const dt=Math.min((now-previous)/1000,.05);previous=now;
      if(document.hidden||!labVisible)return;
      if(!paused)phase+=dt;
      const w=canvas.width,h=canvas.height,dpr=devicePixelRatio;
      ctx.clearRect(0,0,w,h);
      const activity=Number(byId('physical_activity').value)/120;
      const sleep=Number(byId('sleep_duration').value)/10;
      for(let line=0;line<26;line++){
        ctx.beginPath();ctx.strokeStyle=`rgba(164,78,46,${.16+line/90})`;ctx.lineWidth=dpr;
        for(let step=0;step<=120;step++){
          const x=step/120*w;
          const wave=Math.sin(step*.065+phase*.45+line*.07)*(1-sleep*.65)*h*.14;
          const wave2=Math.sin(step*.035+line*.12+activity*3)*h*.09;
          const y=h*.28+line*h*.016+wave+wave2;
          if(step===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
        }
        ctx.stroke();
      }
    }
    requestAnimationFrame(render);
    byId('reset-view').addEventListener('click',()=>phase=0);
  }
})();
