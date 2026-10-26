# Runtime verification

The unchanged active backend passed 512 validation-only requests producing 3,072 outputs across all six models. Authentication, invalid-input rejection, consensus label/probability rule, and overload backpressure checks passed.

The complete service ran under a real 512 MiB Linux cgroup limit with swap disabled. The supervisor reported a 363.5 MiB lifetime peak and no OOM kill. The JSON samples memory before shutdown, so its recorded peak is slightly lower. The request loop took 115.3 seconds while the separate tuning process was also active; this is not an uncontended latency benchmark.

This is a local test of the existing backend, not a deployment and not proof that an experimental candidate is compatible. No Render or Vercel changes were made.
