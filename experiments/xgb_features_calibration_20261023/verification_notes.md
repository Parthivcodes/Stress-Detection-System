# Verification notes

- Synthetic feature tests passed: 69/31/38-column encodings, risk-code directions, sleep bands, training-fitted BMI clipping, JSON restore, temperature scaling and exact cutoff search versus brute force.
- Model-layout validation passed: 47 original data/model files and historical manifests matched.
- Actual six-model Gunicorn API tested locally with `MemoryMax=512M`, `MemorySwapMax=0`: 512 validation-only requests and 3,072 individual-model outputs passed, plus authentication, malformed input, consensus decision consistency and bounded concurrent overload checks.
- Runtime JSON samples the cgroup peak before shutdown (350.0 MiB). The supervisor reported the complete service lifetime peak as 358.7 MiB, including shutdown. No OOM or OOM kill occurred.
- Cloud services were not changed or deployed. This validates the existing active backend, not a candidate promoted to production.
- Optuna 5.0.0 and required dependencies were isolated in `/tmp/stress-optuna-20261023` in WSL. That temporary folder can disappear across WSL restarts. Recreate it if needed; do not modify the backend environment just to run this experiment.
- The previous attempt to install this isolated package set onto the Windows-mounted `tmp/optuna_runtime` failed on filesystem timestamp permissions. It is not used by this experiment or the backend.
