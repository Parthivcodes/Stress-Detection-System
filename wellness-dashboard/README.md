# Nerva — Stress Monitoring & Wellness

A standalone, front-end-only React dashboard inspired by the supplied biometric interface references. All people, readings, histories, classifications, and recommendations are fictional demo fixtures. It does not call the existing Python backend or load trained models.

## Run

Requires Node.js 20.19+ or 22.12+ (verified here with Node 24).

```powershell
cd wellness-dashboard
npm install
npm run dev
```

Open the URL printed by Vite, normally **http://127.0.0.1:5173**. This is a separate app from the older Python dashboard on port 8000. Keep the terminal running. No Python server, credentials, network APIs, or environment variables are needed. Fonts are bundled locally.

For a repeatable dependency installation, use `npm ci` with the committed lockfile.

```powershell
npm run build
npm run preview
npm test
```

The static production build is in `dist/`. Serve that directory over HTTP; do not double-click `index.html`. The relative Vite base also supports deployment beneath a static site's subdirectory.

## Structure

```text
src/
  App.tsx                     Dashboard state and layout
  main.tsx                    React entry point, fonts, reduced-motion configuration
  components/
    GlassCard.tsx             Shared glass surface and hover effect
    MetricCard.tsx            Reading, delta, status, mini chart
    StatusChip.tsx            Icon + text + color status
    DotPlot.tsx               Keyboard/hover-accessible SVG history
    TimelineBar.tsx           Period buttons and a keyboard-accessible scrubber
    BodyViewer3D.tsx          Lazy-loaded procedural Three.js anatomy
    RegionDetailCard.tsx     Context panel for a selected body region
    ProfilePanel.tsx         Profile, records, stress score, counselor, recommendations
    Modal.tsx                Native dialog with focus containment and Escape support
  data/
    wellness.ts              All mock readings, people, periods, records and suggestions
    wellness.test.ts         Timeline/data/filtering tests
  hooks/
    useAnimatedNumber.ts     Reduced-motion-aware number transitions
  styles/
    index.css                Layout, visual tokens, Tailwind import, responsive rules
```

## Try these interactions

- Scrub the timeline from May to October. August is Critical: the profile and ambient glow become red, the stress ring updates, and the body and readings change. October returns to Optimal.
- Select Hormone, Cardiovascular, Nervous System, Sleep, Mood or Digestive. Arrow keys, Home and End navigate the category tabs. The `+` button opens the category chooser.
- Overview shows all metrics. Search finds metrics across categories; clearing it restores the selected view.
- Select a highlighted body region or use the four labeled icon buttons beside the figure. Escape or X closes its detail card.
- Drag the body to rotate. Use +/- to zoom, reset to restore the camera, pause to stop ambient motion, and 3D to switch between rotation and a fixed front view.
- Change Program: Rest & recovery selects Sleep; Mindful movement selects Nervous System; Everyday balance selects Hormone. Scan type selects a related body region. These are UI filters on the same mock profile, not separate scans.
- Collapse User Profile, change the record selector, open HRV/Journal folders, choose a counselor, or open a recommendation.
- Hover or keyboard-focus a chart dot to inspect its period and value. Reference lines use each metric's fictional baseline. Delta chips compare against the previous month, with zero delta in the first month.

## Reference adaptations and assumptions

- The screenshots were photographs of a design. This is an interpretation of their three-zone composition, muted glass surfaces, lime pills, staggered cards, floating annotations, and timeline, not an extraction of their original assets.
- The body is generated from wireframe primitives, paired ribs, a spine, arms and highlighted volumes. It is deliberately an **illustration**, not anatomically validated medical imaging. No external 3D assets are needed. If WebGL fails, region buttons and the data panels remain usable.
- “Hormone” is retained as requested and includes the mock metabolic cards. Clinical-looking units and reference notes are illustrative. The status thresholds in `data/wellness.ts` are arbitrary demo logic, not medical reference ranges or a validated stress algorithm.
- The six months are mock snapshots, not live sensor readings. The scrubber updates every card and body highlight from the same selected period.
- Record type changes the profile's history context caption. The two folder buttons open their respective sample records. Counselor selection is local UI state and does not contact anyone. Changes reset when the page reloads.
- The 2D control means a fixed frontal rendering of the same model, not a different imaging modality.
- Tablets use two columns with metrics below. Small screens stack body, profile and metrics. Program and scan dropdowns remain available at all sizes. The timeline floats near the bottom while scrolling.
- Status always includes text and an icon. Motion respects the OS reduced-motion preference. Native selects, dialog focus containment, labeled icon buttons, focus outlines and keyboard-accessible chart points are included.

## Replacing the mock data

Keep the exported types in `data/wellness.ts` and replace the fixture exports with an API adapter. Period indices currently align metric arrays, so validate the data before rendering. Do not treat the demo status function as a clinical decision rule. No backend is included in this deliverable.

## Verification and limitations

TypeScript and the production build are checked. Data tests cover complete timeline values, region references, status transitions, search/category behavior, and paired blood-pressure values. The 3D module is code-split; its dependency chunk is still approximately 0.9 MB uncompressed, so Vite reports a size advisory.

A connected browser was not available for this session. Rendered visual matching, real-device WebGL behavior, and end-to-end mouse/keyboard interaction need a browser pass before presentation. Build and server checks do not substitute for that visual verification.
