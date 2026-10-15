# Reference recreation — NERVA

## Preview and implementation

- Landing page: http://localhost:8000/
- Integrated predictions: http://localhost:8000/#predictions . The former `/workspace.html` now redirects here; the human-body dashboard has been removed from the active experience.
- Start with the existing `Start-Dashboard.ps1` or the command in README.md.
- Frontend remains buildless HTML/CSS/JavaScript, served by the existing Python backend. No frontend package installation or animation framework was required.
- `sechenov_ui/landing.js` uses the existing local Three.js renderer, a single animation clock, GPU-interpolated particle positions, and a pointer repulsion field.
- Five selectable forms: sphere, diamond, helix, torus knot, wave. Desktop uses 22,000 particles; mobile uses 13,000. Device pixel ratio is capped at 1.6.
- Pause/resume, keyboard form selection, reduced-motion preference, background-tab suspension, static Canvas fallback, and responsive layouts are included.
- Navigation and primary actions scroll to the predictions, methodology, and benchmark evidence on this same page. About opens an accessible native dialog.
- `prediction.css` and `prediction.js` provide glass-effect profile controls, six live model cards with class probabilities, an ensemble summary, and JSON export using the unchanged Python prediction endpoint.

## Recording observations

The supplied recording is 39 seconds at 30 fps, 1196 × 796 pixels. Contact sheet and full-resolution samples are in `output/reference/`.

The visible experience is a single fixed viewport with a near-black background, thin rounded border, small wordmark and numbered navigation. Left-aligned text remains stationary as a particle sculpture changes shape in the center. A right-side label and progress indicator identify the current form; five lower-right dots select forms.

The visible sequence starts partway through Helix, then moves through Torus, Wave, Sphere, Diamond, and back to Helix. The later recording includes playback controls; those are video-player chrome, not website UI. No scrolling sections, pinned scroll sequences, or destination-page contents are shown.

The recreation therefore uses time-based morphs, not invented scroll-driven sections. The cycle is 6.5 seconds per form with a 1.75-second smoothstep transition, subtle scatter during morphs, slow rotation, and pointer deformation.

## Assets and deliberate differences

- VELA identity and design-studio copy are replaced with the project's NERVA identity and stress-research content. The three-line headline composition is retained.
- Particle coordinates, material shaders, motion parameters, and camera placement are reconstructed from observation. They are not the original site's source, and are not claimed to match its trajectories exactly.
- Instrument Serif is a close visual font match, not a verified identification. Regular and italic files plus their license are self-hosted from the Google Fonts repository: https://github.com/google/fonts/tree/main/ofl/instrumentserif . Outfit is reused from the existing project with its license.
- No AI-generated raster assets or external models were needed. The centerpiece is real procedural 3D. Exact source shaders, font declarations, and original timing parameters would enable a closer match.
- Added pause control and About content support accessibility and the project context. Mobile/tablet composition is an adaptation because those sizes are absent from the recording.
- At the user's request, the prediction UI now sits below the hero in the same dark-and-sand theme, replacing the separate anatomy workspace. These added sections are project functionality, not content shown in the reference recording.

## Verification

- Browser comparison at 1196 × 796, plus responsive checks at 390 × 844 and 768 × 1024.
- All five form endpoints inspected; live transition start and an interrupted intermediate morph inspected.
- Navigation into the preserved workspace, About dialog and Escape dismissal checked.
- Actual six-model backend inference completed successfully; the example profile returned Moderate Stress.
- No browser console errors observed on the landing page or tested workspace flow.
- `node --check sechenov_ui/landing.js`
- `node tests/landing-smoke.cjs`: reduced-motion startup, manual form selection, finite interrupted interpolation, pause/resume, mobile particle budget, and missing-WebGL fallback passed. These use lightweight renderer mocks; real GPU visuals were checked in the browser.

No backend, dataset, or trained model changes were needed for this recreation.

## Current local runtime limitation

Update, 15 October 2026: the dashboard is restored through Ubuntu/WSL and `/opt/stress-venv` (Python 3.12, CPU-only PyTorch and XGBoost). All six actual predictions, three API test profiles, invalid inputs, preprocessing parity, and browser Analyze interaction passed. Use `Start-Dashboard.ps1` and http://localhost:8000/#predictions. Native Windows DLL restrictions remain unchanged; the paragraph below records the earlier failure, not the current dashboard status.

On the final resumed session, Windows Application Control blocked `.runtime/torch/lib/torch_python.dll` (WinError 4551), including outside the sandbox. The model server cannot currently start. All six live predictions had passed in the previous session; those results are not simulated in the frontend. A static layout preview is available at http://localhost:8001/#predictions while the Python runtime issue is unresolved. Mobile layout and the legacy workspace redirect were checked in this preview. No security policy changes were made.
