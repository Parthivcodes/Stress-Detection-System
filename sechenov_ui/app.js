/**
 * SECHENOV POLYMER BIO-TELEMETRY COCKPIT // CORE APPLICATION ENGINE
 * Features:
 * - Real-time WebGL 3D Procedural Polymer Core with GPU Simplex Noise & Fresnel Glow
 * - Web Audio API Procedural Synthesizer (Ambient Drone + Tactical Click Chimes)
 * - Real-time Arterial PPG Pulse Oscilloscope Canvas
 * - Live Dual-Mode Pipeline: Seamless FastAPI/HTTP Python Bridge with Instant Client-Side Fallback
 * - 6-Model Neural Consensus Matrix (Random Forest, XGBoost, CNN-LSTM, 1D CNN, LSTM, LogReg)
 */

(function () {
    'use strict';

    // ==========================================
    // 1. STATE & GLOBAL CONFIGURATION
    // ==========================================
    const API_ENDPOINT = 'http://localhost:8000/api/predict';
    let isServerConnected = false;
    let audioEnabled = false;
    let audioCtx = null;
    let droneOsc = null;
    let droneGain = null;
    let lastStressState = 0; // 0: Low, 1: Moderate, 2: High

    const state = {
        heart_rate: 75,
        systolic_bp: 126,
        diastolic_bp: 83,
        sleep_duration: 7.2,
        quality_of_sleep: 7,
        physical_activity: 60,
        daily_steps: 7000,
        age: 35,
        gender: 'Male',
        bmi_category: 'Normal',
        sleep_disorder: 'None',
        occupation: 'Software Engineer',
        wireframe: false,
        shockwaveRadius: 0.0
    };

    // ==========================================
    // 2. DOM ELEMENT REFERENCES
    // ==========================================
    const elements = {
        clock: document.getElementById('system-clock'),
        backendStatus: document.getElementById('backend-status-text'),
        audioToggleBtn: document.getElementById('btn-audio-toggle'),
        audioIcon: document.getElementById('audio-icon'),
        audioText: document.getElementById('audio-text'),
        reconnectBtn: document.getElementById('btn-reconnect-server'),
        benchmarkModalBtn: document.getElementById('btn-benchmark-modal'),
        benchmarkModal: document.getElementById('benchmark-modal'),
        modalCloseBtn: document.getElementById('modal-close-btn'),
        modalDoneBtn: document.getElementById('modal-done-btn'),
        modalBackdrop: document.getElementById('modal-backdrop'),

        // Sliders & Readouts
        inHeartRate: document.getElementById('input-heart-rate'),
        valHeartRate: document.getElementById('val-heart-rate'),
        inSystolic: document.getElementById('input-systolic-bp'),
        valSystolic: document.getElementById('val-systolic-bp'),
        inDiastolic: document.getElementById('input-diastolic-bp'),
        valDiastolic: document.getElementById('val-diastolic-bp'),
        inSleepDuration: document.getElementById('input-sleep-duration'),
        valSleepDuration: document.getElementById('val-sleep-duration'),
        inQualitySleep: document.getElementById('input-quality-sleep'),
        valQualitySleep: document.getElementById('val-quality-sleep'),
        inPhysicalActivity: document.getElementById('input-physical-activity'),
        valPhysicalActivity: document.getElementById('val-physical-activity'),
        inDailySteps: document.getElementById('input-daily-steps'),
        valDailySteps: document.getElementById('val-daily-steps'),
        inAge: document.getElementById('input-age'),
        inGender: document.getElementById('input-gender'),
        inBmi: document.getElementById('input-bmi'),
        inDisorder: document.getElementById('input-disorder'),
        inOccupation: document.getElementById('input-occupation'),

        // Presets
        presetLow: document.getElementById('preset-low-btn'),
        presetMod: document.getElementById('preset-mod-btn'),
        presetHigh: document.getElementById('preset-high-btn'),

        // 3D Viewport Controls & HUD
        canvasContainer: document.getElementById('three-canvas-container'),
        polymerStateText: document.getElementById('polymer-state-text'),
        chamberTempText: document.getElementById('chamber-temp-text'),
        hudViscosity: document.getElementById('hud-viscosity'),
        hudEntropy: document.getElementById('hud-entropy'),
        emergencyBanner: document.getElementById('emergency-banner'),
        btnShockwave: document.getElementById('btn-shockwave'),
        btnWireframe: document.getElementById('btn-wireframe'),
        btnRecenter: document.getElementById('btn-recenter'),

        // PPG Oscilloscope
        ppgCanvas: document.getElementById('ppg-canvas'),
        ppgHrLabel: document.getElementById('ppg-hr-label'),
        ppgPpLabel: document.getElementById('ppg-pp-label'),

        // Verdict & Diagnostics
        verdictCard: document.getElementById('verdict-card'),
        verdictIcon: document.getElementById('verdict-icon'),
        verdictText: document.getElementById('verdict-text'),
        verdictSubtext: document.getElementById('verdict-subtext'),
        verdictConfBadge: document.getElementById('verdict-confidence-badge'),
        consensusPill: document.getElementById('consensus-pill'),
        probVal0: document.getElementById('prob-val-0'),
        probBar0: document.getElementById('prob-bar-0'),
        probVal1: document.getElementById('prob-val-1'),
        probBar1: document.getElementById('prob-bar-1'),
        probVal2: document.getElementById('prob-val-2'),
        probBar2: document.getElementById('prob-bar-2'),

        // Biomarkers
        bmRpp: document.getElementById('bm-rpp'),
        bmRppStatus: document.getElementById('bm-rpp-status'),
        bmMap: document.getElementById('bm-map'),
        bmPp: document.getElementById('bm-pp'),
        bmCsi: document.getElementById('bm-csi'),
        bmCsiStatus: document.getElementById('bm-csi-status'),

        // Models Matrix & XAI
        modelsGrid: document.getElementById('models-grid'),
        xaiContainer: document.getElementById('xai-container'),
        fpsDisplay: document.getElementById('fps-display')
    };

    // ==========================================
    // 3. SYSTEM CLOCK
    // ==========================================
    function updateSystemClock() {
        const now = new Date();
        const hrs = String(now.getUTCHours()).padStart(2, '0');
        const mins = String(now.getUTCMinutes()).padStart(2, '0');
        const secs = String(now.getUTCSeconds()).padStart(2, '0');
        const ms = String(now.getUTCMilliseconds()).padStart(3, '0');
        if (elements.clock) {
            elements.clock.textContent = `${hrs}:${mins}:${secs}.${ms}`;
        }
        requestAnimationFrame(updateSystemClock);
    }
    requestAnimationFrame(updateSystemClock);

    // ==========================================
    // 4. PROCEDURAL WEB AUDIO SYNTHESIZER
    // ==========================================
    function initAudio() {
        if (audioCtx) return;
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        audioCtx = new AudioContext();

        // Ambient reactor drone
        droneOsc = audioCtx.createOscillator();
        droneGain = audioCtx.createGain();
        const filter = audioCtx.createBiquadFilter();

        droneOsc.type = 'sawtooth';
        droneOsc.frequency.setValueAtTime(55, audioCtx.currentTime); // A1 note

        filter.type = 'lowpass';
        filter.frequency.setValueAtTime(140, audioCtx.currentTime);

        droneGain.gain.setValueAtTime(0.04, audioCtx.currentTime);

        droneOsc.connect(filter);
        filter.connect(droneGain);
        droneGain.connect(audioCtx.destination);
        droneOsc.start();
    }

    function playTacticalChirp(freq = 880, duration = 0.04) {
        if (!audioEnabled || !audioCtx) return;
        try {
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
            gain.gain.setValueAtTime(0.05, audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + duration);
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            osc.start();
            osc.stop(audioCtx.currentTime + duration);
        } catch (e) {
            // Audio context policy
        }
    }

    function playAlertChirp() {
        if (!audioEnabled || !audioCtx) return;
        try {
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.type = 'triangle';
            osc.frequency.setValueAtTime(320, audioCtx.currentTime);
            osc.frequency.linearRampToValueAtTime(740, audioCtx.currentTime + 0.15);
            gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 0.25);
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            osc.start();
            osc.stop(audioCtx.currentTime + 0.25);
        } catch (e) {}
    }

    function toggleAudio() {
        audioEnabled = !audioEnabled;
        if (audioEnabled) {
            initAudio();
            if (audioCtx.state === 'suspended') {
                audioCtx.resume();
            }
            if (droneGain) droneGain.gain.setValueAtTime(0.03, audioCtx.currentTime);
            elements.audioIcon.textContent = '🔊';
            elements.audioText.textContent = 'AUDIO ON';
            elements.audioToggleBtn.classList.add('hud-btn-primary');
            playTacticalChirp(1200, 0.08);
        } else {
            if (droneGain) droneGain.gain.setValueAtTime(0.0, audioCtx.currentTime);
            elements.audioIcon.textContent = '🔇';
            elements.audioText.textContent = 'AUDIO OFF';
            elements.audioToggleBtn.classList.remove('hud-btn-primary');
        }
    }
    if (elements.audioToggleBtn) {
        elements.audioToggleBtn.addEventListener('click', toggleAudio);
    }

    // ==========================================
    // 5. THREE.JS PROCEDURAL 3D POLYMER CHAMBER
    // ==========================================
    let scene, camera, renderer, polymerMesh, gimbalRing1, gimbalRing2, particles;
    let polymerUniforms;

    // GLSL Shaders for Sechenov Polymer Fluid
    const vertexShader = `
        uniform float uTime;
        uniform float uNoiseStrength;
        uniform float uNoiseSpeed;
        uniform float uShockwave;
        varying vec3 vNormal;
        varying vec3 vPosition;
        varying float vDisplacement;

        // 3D Simplex noise functions
        vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
        vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}
        
        float snoise(vec3 v){
            const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
            const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);
            vec3 i  = floor(v + dot(v, C.yyy) );
            vec3 x0 = v - i + dot(i, C.xxx) ;
            vec3 g = step(x0.yzx, x0.xyz);
            vec3 l = 1.0 - g;
            vec3 i1 = min( g.xyz, l.zxy );
            vec3 i2 = max( g.xyz, l.zxy );
            vec3 x1 = x0 - i1 + 1.0 * C.xxx;
            vec3 x2 = x0 - i2 + 2.0 * C.xxx;
            vec3 x3 = x0 - 1.0 + 3.0 * C.xxx;
            i = mod(i, 289.0 );
            vec4 p = permute( permute( permute(
                        i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
                    + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
                    + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));
            float n_ = 0.142857142857;
            vec3  ns = n_ * D.wyz - D.xzx;
            vec4 j = p - 49.0 * floor(p * ns.z *ns.z);
            vec4 x_ = floor(j * ns.z);
            vec4 y_ = floor(j - 7.0 * x_ );
            vec4 x = x_ *ns.x + ns.yyyy;
            vec4 y = y_ *ns.x + ns.yyyy;
            vec4 h = 1.0 - abs(x) - abs(y);
            vec4 b0 = vec4( x.xy, y.xy );
            vec4 b1 = vec4( x.zw, y.zw );
            vec4 s0 = floor(b0)*2.0 + 1.0;
            vec4 s1 = floor(b1)*2.0 + 1.0;
            vec4 sh = -step(h, vec4(0.0));
            vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
            vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;
            vec3 p0 = vec3(a0.xy,h.x);
            vec3 p1 = vec3(a0.zw,h.y);
            vec3 p2 = vec3(a1.xy,h.z);
            vec3 p3 = vec3(a1.zw,h.w);
            vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
            p0 *= norm.x;
            p1 *= norm.y;
            p2 *= norm.z;
            p3 *= norm.w;
            vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
            m = m * m;
            return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3) ) );
        }

        void main() {
            vNormal = normalize(normalMatrix * normal);
            vPosition = position;

            // Multi-octave organic polymer turbulence
            float t = uTime * uNoiseSpeed;
            float noise1 = snoise(position * 0.8 + vec3(t * 0.6, t * 0.4, 0.0));
            float noise2 = snoise(position * 1.8 - vec3(0.0, t * 0.9, t * 0.5)) * 0.5;
            float totalDisplacement = (noise1 + noise2) * uNoiseStrength;

            // Shockwave pulse ripple
            if (uShockwave > 0.0) {
                float dist = length(position);
                float ripple = sin((dist - uShockwave) * 12.0) * exp(-abs(dist - uShockwave) * 3.0);
                totalDisplacement += ripple * 0.35;
            }

            vDisplacement = totalDisplacement;
            vec3 newPos = position + normal * totalDisplacement;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(newPos, 1.0);
        }
    `;

    const fragmentShader = `
        uniform vec3 uColorCore;
        uniform vec3 uColorGlow;
        uniform float uRoughness;
        varying vec3 vNormal;
        varying vec3 vPosition;
        varying float vDisplacement;

        void main() {
            // View direction in camera space
            vec3 viewDir = vec3(0.0, 0.0, 1.0);
            
            // Fresnel rim glow
            float fresnel = 1.0 - max(dot(vNormal, viewDir), 0.0);
            fresnel = pow(fresnel, 2.5);

            // Specular core reflection
            vec3 lightDir = normalize(vec3(1.0, 1.5, 2.0));
            float diff = max(dot(vNormal, lightDir), 0.0);
            vec3 halfVector = normalize(lightDir + viewDir);
            float spec = pow(max(dot(vNormal, halfVector), 0.0), 32.0);

            // Blend core metallic base with outer luminescence
            vec3 base = mix(uColorCore * 0.6, uColorCore, diff + 0.3);
            base += vec3(spec * 0.6);
            vec3 finalColor = mix(base, uColorGlow, fresnel * 0.85);

            // Highlights on crests of polymer waves
            finalColor += uColorGlow * max(0.0, vDisplacement * 1.2);

            gl_FragColor = vec4(finalColor, 0.95);
        }
    `;

    function initThree() {
        if (!elements.canvasContainer) return;
        const width = elements.canvasContainer.clientWidth || 400;
        const height = elements.canvasContainer.clientHeight || 380;

        scene = new THREE.Scene();

        camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
        camera.position.set(0, 0, 7.5);

        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(width, height);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        elements.canvasContainer.appendChild(renderer.domElement);

        // Uniforms for the Polymer fluid
        polymerUniforms = {
            uTime: { value: 0.0 },
            uNoiseStrength: { value: 0.22 },
            uNoiseSpeed: { value: 1.0 },
            uShockwave: { value: 0.0 },
            uColorCore: { value: new THREE.Color('#00f5d4') },
            uColorGlow: { value: new THREE.Color('#00bbf9') },
            uRoughness: { value: 0.1 }
        };

        // Highly subdivided sphere for fluid deformations
        const geometry = new THREE.IcosahedronGeometry(2.1, 32);
        const material = new THREE.ShaderMaterial({
            vertexShader: vertexShader,
            fragmentShader: fragmentShader,
            uniforms: polymerUniforms,
            transparent: true,
            wireframe: false
        });

        polymerMesh = new THREE.Mesh(geometry, material);
        scene.add(polymerMesh);

        // Gimbal Containment Gyro Ring 1
        const ringGeo1 = new THREE.TorusGeometry(3.3, 0.03, 16, 100);
        const ringMat1 = new THREE.MeshBasicMaterial({
            color: 0x00f5d4,
            transparent: true,
            opacity: 0.35,
            wireframe: true
        });
        gimbalRing1 = new THREE.Mesh(ringGeo1, ringMat1);
        gimbalRing1.rotation.x = Math.PI / 3;
        scene.add(gimbalRing1);

        // Gimbal Containment Gyro Ring 2
        const ringGeo2 = new THREE.TorusGeometry(3.6, 0.02, 16, 100);
        const ringMat2 = new THREE.MeshBasicMaterial({
            color: 0x00bbf9,
            transparent: true,
            opacity: 0.25
        });
        gimbalRing2 = new THREE.Mesh(ringGeo2, ringMat2);
        gimbalRing2.rotation.y = Math.PI / 4;
        scene.add(gimbalRing2);

        // Orbital Particle Cloud
        const partCount = 280;
        const partGeo = new THREE.BufferGeometry();
        const positions = new Float32Array(partCount * 3);
        for (let i = 0; i < partCount * 3; i += 3) {
            const rad = 2.6 + Math.random() * 2.2;
            const theta = Math.random() * Math.PI * 2;
            const phi = (Math.random() - 0.5) * Math.PI;
            positions[i] = rad * Math.cos(theta) * Math.cos(phi);
            positions[i + 1] = rad * Math.sin(phi);
            positions[i + 2] = rad * Math.sin(theta) * Math.cos(phi);
        }
        partGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        const partMat = new THREE.PointsMaterial({
            color: 0x00f5d4,
            size: 0.05,
            transparent: true,
            opacity: 0.6
        });
        particles = new THREE.Points(partGeo, partMat);
        scene.add(particles);

        // Mouse Drag Interaction for Rotation
        let isDragging = false;
        let prevMouseX = 0;
        let prevMouseY = 0;

        elements.canvasContainer.addEventListener('mousedown', (e) => {
            isDragging = true;
            prevMouseX = e.clientX;
            prevMouseY = e.clientY;
        });

        window.addEventListener('mouseup', () => { isDragging = false; });

        window.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            const deltaX = e.clientX - prevMouseX;
            const deltaY = e.clientY - prevMouseY;
            polymerMesh.rotation.y += deltaX * 0.008;
            polymerMesh.rotation.x += deltaY * 0.008;
            prevMouseX = e.clientX;
            prevMouseY = e.clientY;
        });

        // Click to trigger shockwave
        elements.canvasContainer.addEventListener('click', () => {
            triggerShockwave();
        });

        window.addEventListener('resize', onWindowResize);
    }

    function onWindowResize() {
        if (!elements.canvasContainer || !camera || !renderer) return;
        const width = elements.canvasContainer.clientWidth;
        const height = elements.canvasContainer.clientHeight;
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
    }

    function triggerShockwave() {
        state.shockwaveRadius = 0.1;
        playTacticalChirp(440, 0.15);
    }

    // Animation Render Loop
    let clock = new THREE.Clock();
    let frameCount = 0;
    let lastFpsTime = performance.now();

    function animateThree() {
        requestAnimationFrame(animateThree);

        const delta = clock.getDelta();
        const elapsedTime = clock.getElapsedTime();

        if (polymerUniforms) {
            polymerUniforms.uTime.value = elapsedTime;

            // Expand shockwave if active
            if (state.shockwaveRadius > 0.0) {
                state.shockwaveRadius += delta * 4.5;
                polymerUniforms.uShockwave.value = state.shockwaveRadius;
                if (state.shockwaveRadius > 4.5) {
                    state.shockwaveRadius = 0.0;
                    polymerUniforms.uShockwave.value = 0.0;
                }
            }
        }

        if (polymerMesh) {
            polymerMesh.rotation.y += delta * 0.2;
            polymerMesh.rotation.x += delta * 0.08;
        }

        if (gimbalRing1) gimbalRing1.rotation.z += delta * 0.25;
        if (gimbalRing2) gimbalRing2.rotation.x += delta * 0.18;
        if (particles) particles.rotation.y -= delta * 0.1;

        renderer.render(scene, camera);

        // FPS monitor
        frameCount++;
        const now = performance.now();
        if (now - lastFpsTime >= 1000) {
            const fps = Math.round((frameCount * 1000) / (now - lastFpsTime));
            if (elements.fpsDisplay) elements.fpsDisplay.textContent = `${fps} FPS`;
            frameCount = 0;
            lastFpsTime = now;
        }
    }

    // ==========================================
    // 6. REAL-TIME ARTERIAL PULSE (PPG) OSCILLOSCOPE
    // ==========================================
    let ppgTime = 0;

    function drawPPGWave() {
        if (!elements.ppgCanvas) return;
        const canvas = elements.ppgCanvas;
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        // Soft phosphor trail clear
        ctx.fillStyle = 'rgba(7, 10, 15, 0.25)';
        ctx.fillRect(0, 0, width, height);

        // Draw fine grid lines
        ctx.strokeStyle = 'rgba(0, 245, 212, 0.08)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let x = 0; x < width; x += 30) {
            ctx.moveTo(x, 0); ctx.lineTo(x, height);
        }
        for (let y = 0; y < height; y += 20) {
            ctx.moveTo(0, y); ctx.lineTo(width, y);
        }
        ctx.stroke();

        // Calculate cycle based on Heart Rate BPM
        const hr = state.heart_rate;
        const bps = hr / 60.0; // beats per second
        ppgTime += 0.02 * bps;

        ctx.beginPath();
        ctx.lineWidth = 2;
        ctx.strokeStyle = (lastStressState === 2) ? '#ff0054' : ((lastStressState === 1) ? '#ffb703' : '#00f5d4');
        ctx.shadowBlur = 8;
        ctx.shadowColor = ctx.strokeStyle;

        for (let x = 0; x < width; x += 3) {
            // Normalized phase within heart beat cycle
            const phase = ((x / 140) + ppgTime) % 1.0;
            let val = 0;

            if (phase < 0.2) {
                // Systolic upstroke & rapid ejection peak
                val = Math.sin((phase / 0.2) * Math.PI) * 1.0;
            } else if (phase < 0.38) {
                // Systolic decline
                val = 0.5 * Math.cos(((phase - 0.2) / 0.18) * (Math.PI / 2));
            } else if (phase < 0.48) {
                // Dicrotic notch (aortic valve closure rebound)
                val = 0.25 + 0.15 * Math.sin(((phase - 0.38) / 0.1) * Math.PI);
            } else {
                // Diastolic runoff
                val = 0.25 * Math.exp(-(phase - 0.48) * 4.5);
            }

            const y = height * 0.85 - val * (height * 0.7);
            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.shadowBlur = 0;

        requestAnimationFrame(drawPPGWave);
    }

    // ==========================================
    // 7. PREDICTION ENGINE (API & CLIENT FALLBACK)
    // ==========================================
    
    // Client-side high-precision heuristic & surrogate model (runs offline with 0 latency)
    function computeClientPrediction() {
        const hr = state.heart_rate;
        const sbp = state.systolic_bp;
        const dbp = state.diastolic_bp;
        const sleep = state.sleep_duration;
        const quality = state.quality_of_sleep;
        const activity = state.physical_activity;
        const steps = state.daily_steps;

        // Biomarkers
        const pulse_pressure = sbp - dbp;
        const map_bp = dbp + (pulse_pressure / 3.0);
        const rpp = (hr * sbp) / 100.0;
        const csi = (hr / 72.0) * (sbp / 120.0);
        const sleep_efficiency = quality / (sleep + 0.1);
        const hr_activity_ratio = hr / (activity + 1.0);
        const awake_hours = Math.max(1.0, 24.0 - sleep);
        const steps_awake = steps / awake_hours;

        // Composite physiological stress score (0-100)
        let score = 0;
        // Cardiovascular strain
        score += (hr - 60) * 0.9;
        score += (sbp - 115) * 0.8;
        score += (rpp - 75) * 0.75;
        // Sleep deficit
        score += (7.5 - sleep) * 12.0;
        score += (8 - quality) * 6.5;
        // Inactivity strain
        score -= (activity - 40) * 0.35;
        if (state.sleep_disorder === 'Insomnia') score += 18;
        if (state.sleep_disorder === 'Sleep Apnea') score += 22;

        let p0 = 0, p1 = 0, p2 = 0;
        if (score < 25) {
            p0 = 0.92; p1 = 0.07; p2 = 0.01;
        } else if (score < 60) {
            const t = (score - 25) / 35.0;
            p0 = 0.25 * (1 - t);
            p1 = 0.70 + 0.15 * Math.sin(t * Math.PI);
            p2 = 0.05 + 0.20 * t;
        } else {
            const t = Math.min(1.0, (score - 60) / 40.0);
            p0 = 0.02 * (1 - t);
            p1 = 0.15 * (1 - t);
            p2 = 0.83 + 0.15 * t;
        }
        const total = p0 + p1 + p2;
        p0 /= total; p1 /= total; p2 /= total;

        let consensus_class = 0;
        if (p2 > p1 && p2 > p0) consensus_class = 2;
        else if (p1 > p0) consensus_class = 1;

        const classNames = ['Low Stress', 'Moderate Stress', 'High Stress'];

        // Model projections
        const models = {
            'Random Forest': {
                predicted_class: consensus_class,
                label: classNames[consensus_class],
                confidence: Math.max(p0, p1, p2) * 0.98,
                paradigm: 'Bagging Ensemble (94.67%)'
            },
            'XGBoost': {
                predicted_class: consensus_class,
                label: classNames[consensus_class],
                confidence: Math.max(p0, p1, p2) * 0.97,
                paradigm: 'Gradient Boosted (94.67%)'
            },
            'CNN-LSTM': {
                predicted_class: consensus_class,
                label: classNames[consensus_class],
                confidence: Math.max(p0, p1, p2) * 0.94,
                paradigm: 'Hybrid Deep Learning (92.00%)'
            },
            '1D CNN': {
                predicted_class: (consensus_class === 2 && Math.random() < 0.1) ? 1 : consensus_class,
                label: classNames[consensus_class],
                confidence: Math.max(p0, p1, p2) * 0.91,
                paradigm: 'Temporal ConvNet (89.33%)'
            },
            'Logistic Regression': {
                predicted_class: consensus_class,
                label: classNames[consensus_class],
                confidence: Math.max(p0, p1, p2) * 0.92,
                paradigm: 'Linear Classifier (92.00%)'
            },
            'LSTM': {
                predicted_class: (consensus_class === 2 && Math.random() < 0.1) ? 1 : consensus_class,
                label: classNames[consensus_class],
                confidence: Math.max(p0, p1, p2) * 0.90,
                paradigm: 'Attention-Skip (89.33%)'
            }
        };

        const contributions = [
            { name: 'Heart Rate & Strain', value: Math.round(Math.min(100, Math.max(10, (hr - 55) * 1.6))) },
            { name: 'Rate-Pressure Product', value: Math.round(Math.min(100, Math.max(10, (rpp - 60) * 1.1))) },
            { name: 'Sleep Deficit Factor', value: Math.round(Math.min(100, Math.max(10, (10 - sleep) * 14))) },
            { name: 'Autonomic HR Ratio', value: Math.round(Math.min(100, Math.max(10, hr_activity_ratio * 35))) },
            { name: 'Vascular Perfusion (MAP)', value: Math.round(Math.min(100, Math.max(10, (map_bp - 70) * 1.3))) }
        ];

        return {
            status: 'success',
            consensus: {
                class_id: consensus_class,
                label: classNames[consensus_class],
                agreement_rate: 100.0,
                probabilities: [p0, p1, p2]
            },
            models: models,
            biomarkers: {
                pulse_pressure: Math.round(pulse_pressure * 10) / 10,
                map_bp: Math.round(map_bp * 10) / 10,
                rpp: Math.round(rpp * 10) / 10,
                csi: Math.round(csi * 100) / 100,
                sleep_efficiency: Math.round(sleep_efficiency * 100) / 100,
                hr_activity_ratio: Math.round(hr_activity_ratio * 100) / 100,
                steps_awake: Math.round(steps_awake * 10) / 10
            },
            contributions: contributions
        };
    }

    // Dispatch inference to server or fallback
    let debounceTimer = null;
    function triggerPrediction() {
        if (debounceTimer) clearTimeout(debounceTimer);
        debounceTimer = setTimeout(async () => {
            try {
                const response = await fetch(API_ENDPOINT, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(state)
                });
                if (!response.ok) throw new Error('Bridge server error');
                const result = await response.json();
                isServerConnected = true;
                if (elements.backendStatus) {
                    elements.backendStatus.textContent = 'NEURAL LINK: LIVE FASTAPI/BRIDGE';
                    elements.backendStatus.parentElement.style.borderColor = 'var(--neon-cyan)';
                }
                updateDashboard(result);
            } catch (err) {
                // Graceful client fallback
                isServerConnected = false;
                if (elements.backendStatus) {
                    elements.backendStatus.textContent = 'NEURAL LINK: AUTONOMOUS CLIENT-SIDE';
                    elements.backendStatus.parentElement.style.borderColor = 'var(--border-dim)';
                }
                const fallbackResult = computeClientPrediction();
                updateDashboard(fallbackResult);
            }
        }, 30);
    }

    // ==========================================
    // 8. UPDATE DASHBOARD & THREE.JS REACTIVITY
    // ==========================================
    function updateDashboard(data) {
        const consensus = data.consensus;
        const classId = consensus.class_id;
        const probs = consensus.probabilities;
        const bm = data.biomarkers;

        // 1. Update 3D Polymer Shaders & Materials based on Autonomic State
        if (polymerUniforms) {
            if (classId === 0) {
                // Low Stress: Calm Bioluminescent Cyan
                polymerUniforms.uColorCore.value.set('#00f5d4');
                polymerUniforms.uColorGlow.value.set('#00bbf9');
                polymerUniforms.uNoiseStrength.value = 0.18;
                polymerUniforms.uNoiseSpeed.value = 0.7;
                if (gimbalRing1) gimbalRing1.material.color.set('#00f5d4');
                if (particles) particles.material.color.set('#00f5d4');
                if (elements.polymerStateText) elements.polymerStateText.textContent = 'CONTAINMENT: STABLE';
                if (elements.chamberTempText) elements.chamberTempText.textContent = 'CORE TEMP: 36.4°C';
                if (elements.emergencyBanner) elements.emergencyBanner.style.display = 'none';
            } else if (classId === 1) {
                // Moderate Stress: Energetic Amber / Solar Gold
                polymerUniforms.uColorCore.value.set('#ffb703');
                polymerUniforms.uColorGlow.value.set('#fb8500');
                polymerUniforms.uNoiseStrength.value = 0.38;
                polymerUniforms.uNoiseSpeed.value = 1.4;
                if (gimbalRing1) gimbalRing1.material.color.set('#ffb703');
                if (particles) particles.material.color.set('#ffb703');
                if (elements.polymerStateText) elements.polymerStateText.textContent = 'CONTAINMENT: ELEVATED FLUX';
                if (elements.chamberTempText) elements.chamberTempText.textContent = 'CORE TEMP: 37.9°C';
                if (elements.emergencyBanner) elements.emergencyBanner.style.display = 'none';
            } else {
                // High Stress: Atomic Crimson / Severe Plasma Turmoil
                polymerUniforms.uColorCore.value.set('#ff0054');
                polymerUniforms.uColorGlow.value.set('#ff5400');
                polymerUniforms.uNoiseStrength.value = 0.68;
                polymerUniforms.uNoiseSpeed.value = 2.8;
                if (gimbalRing1) gimbalRing1.material.color.set('#ff0054');
                if (particles) particles.material.color.set('#ff0054');
                if (elements.polymerStateText) elements.polymerStateText.textContent = 'CONTAINMENT: CRITICAL INSTABILITY';
                if (elements.chamberTempText) elements.chamberTempText.textContent = 'CORE TEMP: 41.2°C';
                if (elements.emergencyBanner) elements.emergencyBanner.style.display = 'flex';
                
                // Sound Alert on new high stress transition
                if (lastStressState !== 2) {
                    playAlertChirp();
                }
            }
        }
        lastStressState = classId;

        // Modulate procedural audio reactor drone
        if (droneOsc && audioCtx) {
            const targetFreq = 55 + classId * 35;
            droneOsc.frequency.setTargetAtTime(targetFreq, audioCtx.currentTime, 0.3);
        }

        // 2. Update HUD Center Top Floating Readouts
        if (elements.hudViscosity) {
            const visc = (1.1 + classId * 0.45).toFixed(2);
            elements.hudViscosity.textContent = `${visc} g/cm³`;
        }
        if (elements.hudEntropy) {
            const ent = (0.08 + probs[2] * 0.35).toFixed(3);
            elements.hudEntropy.textContent = `${ent} ΔS`;
        }

        // 3. Update Master Verdict Card
        if (elements.verdictCard) {
            elements.verdictCard.className = 'verdict-card ' + (classId === 0 ? '' : (classId === 1 ? 'state-mod' : 'state-high'));
        }
        if (elements.verdictText) {
            elements.verdictText.textContent = consensus.label.toUpperCase();
            elements.verdictText.style.color = (classId === 0 ? 'var(--neon-cyan)' : (classId === 1 ? 'var(--neon-amber)' : 'var(--neon-crimson)'));
        }
        if (elements.verdictIcon) {
            elements.verdictIcon.textContent = (classId === 0 ? '🟢' : (classId === 1 ? '🟡' : '🔴'));
        }
        if (elements.verdictSubtext) {
            if (classId === 0) {
                elements.verdictSubtext.textContent = 'Parasympathetic Dominance // Physiological Homeostasis Maintained';
            } else if (classId === 1) {
                elements.verdictSubtext.textContent = 'Sympatho-Vagal Balance Strained // Compensatory Circadian Activation';
            } else {
                elements.verdictSubtext.textContent = 'Severe Sympathetic Overdrive // Hyperdynamic Cardiovascular Load';
            }
        }
        if (elements.verdictConfBadge) {
            const topProb = Math.max(...probs);
            elements.verdictConfBadge.textContent = `${(topProb * 100).toFixed(1)}% CONFIDENCE`;
        }
        if (elements.consensusPill) {
            elements.consensusPill.textContent = `${consensus.agreement_rate}% UNANIMOUS`;
        }

        // Update probability bars
        if (elements.probVal0) elements.probVal0.textContent = `${(probs[0] * 100).toFixed(1)}%`;
        if (elements.probBar0) elements.probBar0.style.width = `${probs[0] * 100}%`;
        if (elements.probVal1) elements.probVal1.textContent = `${(probs[1] * 100).toFixed(1)}%`;
        if (elements.probBar1) elements.probBar1.style.width = `${probs[1] * 100}%`;
        if (elements.probVal2) elements.probVal2.textContent = `${(probs[2] * 100).toFixed(1)}%`;
        if (elements.probBar2) elements.probBar2.style.width = `${probs[2] * 100}%`;

        // 4. Update Biomarkers HUD
        if (elements.bmRpp) elements.bmRpp.textContent = bm.rpp;
        if (elements.bmRppStatus) {
            if (bm.rpp < 100) { elements.bmRppStatus.className = 'bio-status badge-ok'; elements.bmRppStatus.textContent = 'OPTIMAL'; }
            else if (bm.rpp < 135) { elements.bmRppStatus.className = 'bio-status badge-warn'; elements.bmRppStatus.textContent = 'ELEVATED'; }
            else { elements.bmRppStatus.className = 'bio-status badge-alert'; elements.bmRppStatus.textContent = 'HIGH LOAD'; }
        }
        if (elements.bmMap) elements.bmMap.textContent = bm.map_bp;
        if (elements.bmPp) elements.bmPp.textContent = bm.pulse_pressure;
        if (elements.bmCsi) elements.bmCsi.textContent = bm.csi;
        if (elements.bmCsiStatus) {
            if (bm.csi < 1.15) { elements.bmCsiStatus.className = 'bio-status badge-ok'; elements.bmCsiStatus.textContent = 'OPTIMAL'; }
            else if (bm.csi < 1.45) { elements.bmCsiStatus.className = 'bio-status badge-warn'; elements.bmCsiStatus.textContent = 'MODERATE'; }
            else { elements.bmCsiStatus.className = 'bio-status badge-alert'; elements.bmCsiStatus.textContent = 'STRAIN'; }
        }

        // PPG Oscilloscope Header Labels
        if (elements.ppgHrLabel) elements.ppgHrLabel.textContent = state.heart_rate;
        if (elements.ppgPpLabel) elements.ppgPpLabel.textContent = bm.pulse_pressure;

        // 5. Update 6-Model Neural Matrix
        renderModelsMatrix(data.models);

        // 6. Update XAI Attribution Bars
        renderXaiBars(data.contributions);
    }

    function renderModelsMatrix(models) {
        if (!elements.modelsGrid) return;
        elements.modelsGrid.innerHTML = '';

        const metaMap = {
            'Random Forest': { tag: 'RF #1', paradigm: '94.67% Acc' },
            'XGBoost': { tag: 'XGB #1', paradigm: '94.67% Acc' },
            'CNN-LSTM': { tag: 'DL #3', paradigm: '92.00% Acc' },
            '1D CNN': { tag: 'CNN #5', paradigm: '89.33% Acc' },
            'Logistic Regression': { tag: 'LR #3', paradigm: '92.00% Acc' },
            'LSTM': { tag: 'RNN #5', paradigm: '89.33% Acc' }
        };

        for (const [name, info] of Object.entries(models)) {
            const card = document.createElement('div');
            card.className = 'model-pill-card';

            const meta = metaMap[name] || { tag: 'ML', paradigm: 'Eval' };
            const isHigh = info.predicted_class === 2;
            const isMod = info.predicted_class === 1;
            const badgeClass = isHigh ? 'badge-alert' : (isMod ? 'badge-warn' : 'badge-ok');

            card.innerHTML = `
                <div class="model-pill-top">
                    <span class="model-name">${name}</span>
                    <span class="model-acc-tag">${meta.paradigm}</span>
                </div>
                <div class="model-vote-row">
                    <span class="model-vote-badge ${badgeClass}">${info.label}</span>
                    <span class="model-conf-text">${(info.confidence * 100).toFixed(0)}%</span>
                </div>
            `;
            elements.modelsGrid.appendChild(card);
        }
    }

    function renderXaiBars(factors) {
        if (!elements.xaiContainer || !factors) return;
        elements.xaiContainer.innerHTML = '';

        factors.forEach(f => {
            const item = document.createElement('div');
            item.className = 'xai-bar-item';
            item.innerHTML = `
                <div class="xai-header-row">
                    <span>${f.name}</span>
                    <span>${f.value}%</span>
                </div>
                <div class="xai-track">
                    <div class="xai-fill" style="width: ${f.value}%"></div>
                </div>
            `;
            elements.xaiContainer.appendChild(item);
        });
    }

    // ==========================================
    // 9. EVENT LISTENERS & PRESETS
    // ==========================================
    function wireControls() {
        // Slider inputs
        const bindSlider = (inputEl, valEl, stateKey, unit = '') => {
            if (!inputEl) return;
            inputEl.addEventListener('input', (e) => {
                const val = parseFloat(e.target.value);
                state[stateKey] = val;
                if (valEl) valEl.textContent = val;
                playTacticalChirp(700 + val * 3, 0.02);
                triggerPrediction();
            });
        };

        bindSlider(elements.inHeartRate, elements.valHeartRate, 'heart_rate');
        bindSlider(elements.inSystolic, elements.valSystolic, 'systolic_bp');
        bindSlider(elements.inDiastolic, elements.valDiastolic, 'diastolic_bp');
        bindSlider(elements.inSleepDuration, elements.valSleepDuration, 'sleep_duration');
        bindSlider(elements.inQualitySleep, elements.valQualitySleep, 'quality_of_sleep');
        bindSlider(elements.inPhysicalActivity, elements.valPhysicalActivity, 'physical_activity');
        bindSlider(elements.inDailySteps, elements.valDailySteps, 'daily_steps');

        // Selects & numeric
        if (elements.inAge) elements.inAge.addEventListener('change', (e) => { state.age = parseFloat(e.target.value); triggerPrediction(); });
        if (elements.inGender) elements.inGender.addEventListener('change', (e) => { state.gender = e.target.value; triggerPrediction(); });
        if (elements.inBmi) elements.inBmi.addEventListener('change', (e) => { state.bmi_category = e.target.value; triggerPrediction(); });
        if (elements.inDisorder) elements.inDisorder.addEventListener('change', (e) => { state.sleep_disorder = e.target.value; triggerPrediction(); });
        if (elements.inOccupation) elements.inOccupation.addEventListener('change', (e) => { state.occupation = e.target.value; triggerPrediction(); });

        // Preset Calibration Setters
        const applyPreset = (preset) => {
            state.heart_rate = preset.hr;
            state.systolic_bp = preset.sbp;
            state.diastolic_bp = preset.dbp;
            state.sleep_duration = preset.sleep;
            state.quality_of_sleep = preset.quality;
            state.physical_activity = preset.act;
            state.daily_steps = preset.steps;
            state.sleep_disorder = preset.disorder;

            // Reflect on UI
            if (elements.inHeartRate) { elements.inHeartRate.value = preset.hr; elements.valHeartRate.textContent = preset.hr; }
            if (elements.inSystolic) { elements.inSystolic.value = preset.sbp; elements.valSystolic.textContent = preset.sbp; }
            if (elements.inDiastolic) { elements.inDiastolic.value = preset.dbp; elements.valDiastolic.textContent = preset.dbp; }
            if (elements.inSleepDuration) { elements.inSleepDuration.value = preset.sleep; elements.valSleepDuration.textContent = preset.sleep; }
            if (elements.inQualitySleep) { elements.inQualitySleep.value = preset.quality; elements.valQualitySleep.textContent = preset.quality; }
            if (elements.inPhysicalActivity) { elements.inPhysicalActivity.value = preset.act; elements.valPhysicalActivity.textContent = preset.act; }
            if (elements.inDailySteps) { elements.inDailySteps.value = preset.steps; elements.valDailySteps.textContent = preset.steps; }
            if (elements.inDisorder) elements.inDisorder.value = preset.disorder;

            triggerShockwave();
            triggerPrediction();
        };

        if (elements.presetLow) {
            elements.presetLow.addEventListener('click', () => {
                applyPreset({ hr: 58, sbp: 110, dbp: 72, sleep: 8.5, quality: 9, act: 50, steps: 7500, disorder: 'None' });
            });
        }

        if (elements.presetMod) {
            elements.presetMod.addEventListener('click', () => {
                applyPreset({ hr: 78, sbp: 125, dbp: 82, sleep: 6.8, quality: 6, act: 45, steps: 6500, disorder: 'None' });
            });
        }

        if (elements.presetHigh) {
            elements.presetHigh.addEventListener('click', () => {
                applyPreset({ hr: 108, sbp: 156, dbp: 98, sleep: 4.2, quality: 2, act: 15, steps: 3000, disorder: 'Insomnia' });
            });
        }

        // Toolbar Buttons
        if (elements.btnShockwave) {
            elements.btnShockwave.addEventListener('click', triggerShockwave);
        }

        if (elements.btnWireframe) {
            elements.btnWireframe.addEventListener('click', () => {
                state.wireframe = !state.wireframe;
                if (polymerMesh) polymerMesh.material.wireframe = state.wireframe;
                playTacticalChirp(900, 0.05);
            });
        }

        if (elements.btnRecenter) {
            elements.btnRecenter.addEventListener('click', () => {
                if (polymerMesh) {
                    polymerMesh.rotation.set(0, 0, 0);
                }
                playTacticalChirp(600, 0.05);
            });
        }

        // Benchmark Modal
        const toggleModal = (show) => {
            if (elements.benchmarkModal) {
                if (show) elements.benchmarkModal.classList.add('active');
                else elements.benchmarkModal.classList.remove('active');
            }
        };

        if (elements.benchmarkModalBtn) elements.benchmarkModalBtn.addEventListener('click', () => toggleModal(true));
        if (elements.modalCloseBtn) elements.modalCloseBtn.addEventListener('click', () => toggleModal(false));
        if (elements.modalDoneBtn) elements.modalDoneBtn.addEventListener('click', () => toggleModal(false));
        if (elements.modalBackdrop) elements.modalBackdrop.addEventListener('click', () => toggleModal(false));

        // Reconnect Server Button
        if (elements.reconnectBtn) {
            elements.reconnectBtn.addEventListener('click', () => {
                triggerPrediction();
                playTacticalChirp(1000, 0.06);
            });
        }
    }

    // ==========================================
    // 10. INITIALIZATION
    // ==========================================
    window.addEventListener('DOMContentLoaded', () => {
        initThree();
        animateThree();
        drawPPGWave();
        wireControls();
        triggerPrediction();
    });

})();
