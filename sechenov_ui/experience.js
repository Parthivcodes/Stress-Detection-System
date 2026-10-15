/* Presentation effects are decorative; inference remains in app.js. */
(() => {
  'use strict';
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  let artworkPaused = reducedMotion.matches;
  const progress = document.getElementById('scroll-progress');
  function updateProgress() {
    const distance = document.documentElement.scrollHeight - innerHeight;
    progress.style.transform = `scaleX(${distance > 0 ? scrollY / distance : 0})`;
  }
  addEventListener('scroll', updateProgress, {passive:true});
  addEventListener('resize', updateProgress);
  updateProgress();

  if (!reducedMotion.matches && 'IntersectionObserver' in window) {
    document.body.classList.add('motion-enabled');
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.remove('pending');
          observer.unobserve(entry.target);
        }
      });
    }, {threshold:0.08});
    document.querySelectorAll('.reveal').forEach(el => {
      el.classList.add('pending');
      observer.observe(el);
    });
  }
  document.querySelectorAll('.signal-card').forEach(card => {
    card.addEventListener('pointermove', event => {
      if (reducedMotion.matches || event.pointerType !== 'mouse') return;
      const box = card.getBoundingClientRect();
      const x = (event.clientX - box.left) / box.width - .5;
      const y = (event.clientY - box.top) / box.height - .5;
      card.style.transform = `rotateX(${-y * 9}deg) rotateY(${x * 12}deg) translateY(-6px)`;
    });
    card.addEventListener('pointerleave', () => card.style.transform = '');
  });

  const host = document.getElementById('hero-scene');
  const button = document.getElementById('hero-motion');
  let renderOnce = () => {};
  function updateMotionButton() {
    button.textContent = artworkPaused ? 'Resume artwork' : 'Pause artwork';
    button.setAttribute('aria-pressed', String(artworkPaused));
  }
  updateMotionButton();
  button.addEventListener('click', () => {
    artworkPaused = !artworkPaused;
    updateMotionButton();
    renderOnce();
  });
  reducedMotion.addEventListener('change', event => {
    artworkPaused = event.matches;
    if (event.matches) {
      document.body.classList.remove('motion-enabled');
      document.querySelectorAll('.signal-card').forEach(card => card.style.transform = '');
    }
    updateMotionButton();
  });

  try {
    if (!window.THREE) throw new Error('WebGL library unavailable');
    const T = window.THREE;
    const scene = new T.Scene();
    const camera = new T.PerspectiveCamera(38, 1, .1, 100);
    camera.position.set(0, 0, 8.8);
    const renderer = new T.WebGLRenderer({alpha:true, antialias:true, powerPreference:'low-power'});
    renderer.setPixelRatio(Math.min(devicePixelRatio, 1.6));
    renderer.outputEncoding = T.sRGBEncoding;
    host.append(renderer.domElement);
    host.classList.add('hero-webgl');
    const sculpture = new T.Group();
    sculpture.rotation.set(.25, -.35, -.2);
    scene.add(sculpture);
    scene.add(new T.AmbientLight(0xb0cdb3, .65));
    const key = new T.DirectionalLight(0xf0ffd8, 1.65);
    key.position.set(-3, 5, 4);
    scene.add(key);
    const rim = new T.PointLight(0x79d6b7, 2.2, 25);
    rim.position.set(4, -2, 2);
    scene.add(rim);
    const material = new T.MeshStandardMaterial({
      color:0x9abf99, metalness:.65, roughness:.23,
      emissive:0x16392d, emissiveIntensity:.25
    });
    const knot = new T.Mesh(new T.TorusKnotGeometry(1.35, .39, 220, 28, 2, 3), material);
    sculpture.add(knot);
    const cage = new T.Mesh(new T.IcosahedronGeometry(2.15, 1), new T.MeshBasicMaterial({
      color:0xbfe3a5, wireframe:true, transparent:true, opacity:.075
    }));
    sculpture.add(cage);
    const rings = [];
    for (let i = 0; i < 3; i++) {
      const ring = new T.Mesh(new T.TorusGeometry(2.35 + i*.13, .009, 6, 160),
        new T.MeshBasicMaterial({color:i===1 ? 0xd5e7a7 : 0x76aa93, transparent:true, opacity:.45}));
      ring.rotation.set(.5+i*.7, .5+i*.45, i*.6);
      sculpture.add(ring);
      rings.push(ring);
    }
    const satellites = [];
    const satelliteGeo = new T.SphereGeometry(.055, 12, 12);
    const satelliteMat = new T.MeshBasicMaterial({color:0xdceeb4});
    for (let i=0; i<8; i++) {
      const dot = new T.Mesh(satelliteGeo, satelliteMat);
      sculpture.add(dot);
      satellites.push(dot);
    }
    const points = [];
    // Deterministic distribution keeps the artwork stable across reloads.
    for (let i=0; i<700; i++) {
      const a=i*2.399963, y=1-2*(i+.5)/700;
      const r=Math.sqrt(1-y*y), radius=3.1+.35*Math.sin(i*1.71);
      points.push(Math.cos(a)*r*radius,y*radius,Math.sin(a)*r*radius);
    }
    const dust = new T.Points(new T.BufferGeometry().setAttribute('position',new T.Float32BufferAttribute(points,3)),
      new T.PointsMaterial({color:0xc8e6b6,size:.018,transparent:true,opacity:.45}));
    scene.add(dust);
    let drag=false, oldX=0, oldY=0, userX=0, userY=0;
    host.addEventListener('pointerdown', event => {
      if (event.pointerType !== 'mouse') return; // Preserve touch scrolling.
      drag=true;oldX=event.clientX;oldY=event.clientY;
      host.setPointerCapture(event.pointerId);
    });
    host.addEventListener('pointermove', event => {
      if (!drag) return;
      userY+=(event.clientX-oldX)*.006;
      userX+=(event.clientY-oldY)*.006;
      oldX=event.clientX;oldY=event.clientY;
      renderOnce();
    });
    host.addEventListener('pointerup',()=>drag=false);
    host.addEventListener('pointercancel',()=>drag=false);
    let visible=true, elapsed=0, previous=0;
    new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;},{threshold:0}).observe(host);
    function draw() {
      sculpture.rotation.x=.25+userX;
      sculpture.rotation.y=-.35+userY+elapsed*.11;
      sculpture.position.y=Math.sin(elapsed*.65)*.12;
      knot.rotation.z=elapsed*.055;
      cage.rotation.y=-elapsed*.035;
      dust.rotation.y=-elapsed*.02;
      satellites.forEach((dot,i)=>{
        const a=elapsed*.18+i*Math.PI/4;
        dot.position.set(Math.cos(a)*2.48,Math.sin(a)*1.4,Math.sin(a+i)*1.7);
      });
      renderer.render(scene,camera);
    }
    renderOnce=draw;
    const resize=()=>{
      const width=host.clientWidth,height=host.clientHeight;
      renderer.setSize(width,height);
      camera.aspect=width/height;
      camera.position.z=width<500?10.4:8.8;
      camera.updateProjectionMatrix();
      draw();
    };
    new ResizeObserver(resize).observe(host);
    resize();
    function animate(now) {
      requestAnimationFrame(animate);
      const delta=Math.min((now-previous)/1000,.05);
      previous=now;
      if (document.hidden || !visible || artworkPaused) return;
      elapsed+=delta;
      draw();
    }
    requestAnimationFrame(animate);
    renderer.domElement.addEventListener('webglcontextlost',event=>{
      event.preventDefault();artworkPaused=true;
      host.classList.remove('hero-webgl');
      renderer.domElement.style.display='none';
      button.textContent='Static artwork';button.disabled=true;
    });
  } catch (error) {
    host.classList.remove('hero-webgl');
    host.querySelector('canvas')?.remove();
    button.textContent='Static artwork';
    button.disabled=true;
  }
})();
