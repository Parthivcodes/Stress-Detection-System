// Vercel only forwards requests. It never manufactures model predictions.
const ROUTES = new Set(['health', 'schema', 'benchmarks', 'predict']);

module.exports = function proxy(route) {
  if (!ROUTES.has(route)) throw new Error('Unsupported backend route');
  return async function handler(req, res) {
    res.setHeader('Cache-Control', 'no-store');
    const method = route === 'predict' ? 'POST' : 'GET';
    if (req.method !== method) {
      res.setHeader('Allow', method);
      return res.status(405).json({status:'error', message:'Method not allowed'});
    }
    const origin = process.env.INFERENCE_BACKEND_URL;
    const token = process.env.INFERENCE_API_KEY;
    if (!origin || !token) {
      return res.status(503).json({status:'error', models_loaded:false, message:'Prediction backend is not configured.'});
    }
    try {
      const url = new URL(origin);
      if (url.protocol !== 'https:' || url.username || url.password ||
          url.pathname !== '/' || url.search || url.hash) throw new Error('Invalid backend origin');
      let body;
      if (method === 'POST') {
        body = typeof req.body === 'string' ? req.body : JSON.stringify(req.body ?? {});
        if (Buffer.byteLength(body) > 16384) {
          return res.status(413).json({status:'error', message:'Request is too large.'});
        }
      }
      const upstream = await fetch(new URL(`/api/${route}`, url), {
        method, body, redirect:'error',
        headers:{'Content-Type':'application/json', 'Authorization':`Bearer ${token}`},
        signal:AbortSignal.timeout(25000),
      });
      if (!upstream.headers.get('content-type')?.includes('application/json')) throw new Error('Invalid response');
      const data = await upstream.json();
      if (upstream.ok && route === 'predict' &&
          (data.model_version !== 'brfss2022_expanded' || Object.keys(data.models ?? {}).length !== 6)) {
        throw new Error('Wrong model response');
      }
      return res.status(upstream.status).json(data);
    } catch {
      return res.status(502).json({status:'error', models_loaded:false,
        message:'Prediction backend is unavailable or waking up. Please retry shortly.'});
    }
  };
};
