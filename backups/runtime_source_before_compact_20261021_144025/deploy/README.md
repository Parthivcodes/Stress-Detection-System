# Vercel frontend + real Python inference backend

## Status

Local loading fix and deployment source are implemented. This is **not a live
cloud deployment** yet. Vercel/Render account connections, an approved backend
compute plan, and a deployed backend URL are still required. No subscriptions
were purchased, no secrets were committed, and no dataset was uploaded.

## Architecture

Browser → Vercel `/api/*` → authenticated HTTPS request → Render Python API
→ unchanged expanded preprocessor and six trained models.

Vercel does not compute or approximate predictions. Missing configuration returns
503; unreachable backend returns 502. Old formula-based Vercel handlers are
retained only in `legacy/vercel_demo_api/` and excluded from deployment.

## Backend settings

Create a Render Docker web service from a **private deployment repository or
private container image**. Do not make this entire research repository public
to deploy it. Review the upload contents first.

- Dockerfile: `deploy/Dockerfile`
- Build context: project root
- Health check: `/healthz`
- Start command: Dockerfile default (Gunicorn, one model process, four threads)
- Bind: `0.0.0.0:$PORT`
- Secret environment variable: `INFERENCE_API_KEY`, a newly generated random
  value of at least 32 characters. Use the identical value on Vercel.
- Select memory/compute after measuring container usage. Do not assume a free
  tier can hold PyTorch and all six models or stay awake for a demonstration.
  Plan/cost approval is required before provisioning.

The image includes current model weights, Python source, and two aggregate
audit/gate JSON files. The Docker allowlist excludes respondent CSVs, backups,
Git history, credentials and frontend assets. Model files are never served by
the API. Requests are not persisted and input values are not logged by this code.

The Dockerfile pins the numerical package versions observed in the working WSL
environment. Docker is not installed here, so the full image build and Gunicorn
runtime still require verification on the hosting service or a Docker-equipped
machine. CPU wheel availability and required memory must pass the build check.

## Vercel settings

- Project root: repository root
- Framework: Other
- Build: `npm run build`
- Output directory: `public`
- Server-only environment variables:
  - `INFERENCE_BACKEND_URL=https://<your-backend-host>` (origin only, no path)
  - `INFERENCE_API_KEY=<same secret as backend>`
- Set variables for the intended Preview/Production environments and redeploy.
- Do not prefix secrets with `VITE_` or `NEXT_PUBLIC_`.

`tools/build_vercel.cjs` copies only an explicit frontend allowlist into `public`.
It refuses an existing `public/` to avoid deleting user files: run release builds
in clean checkouts. Build and inspect the output before publishing.
`.vercelignore` excludes datasets, model weights, backups and reports from uploads.

## Release verification

1. Build the backend container and check its memory use and `/healthz`.
2. Verify unauthenticated `/api/predict` returns 401.
3. Configure Vercel and verify `/api/health` reports
   `model_version: brfss2022_expanded` and `models_loaded: true`.
4. Verify `/api/schema` has 19 inputs and run a sample request returning all six
   model results. Compare probabilities against local inference with 1e-5 tolerance.
5. Check unavailable/busy backend states show errors, not predictions.
6. Check `/models/`, `/data/` and `/backups/` cannot download artifacts.
7. Verify the public website in an external browser, including simultaneous tabs.

For public release, configure provider-level request limits/bot protection.
The shared backend secret blocks direct unauthorized inference calls; it does not
rate-limit the intentionally public Vercel frontend API. No clinical-use claims.

## Local tests

```powershell
node tests/backend-proxy.cjs
wsl.exe -d Ubuntu -u nobody --cd $PWD.Path --exec /opt/stress-venv/bin/python -B tests/website_structure_validation.py
wsl.exe -d Ubuntu -u nobody --cd $PWD.Path --exec /opt/stress-venv/bin/python -B tests/deployment_validation.py
```

## Provider documentation

- https://render.com/docs/docker
- https://render.com/docs/web-services
- https://vercel.com/docs/environment-variables
