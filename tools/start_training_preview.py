"""Separate local simulation preview. Leaves the normal website/server unchanged."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import server
from src.training_preview_inference import TrainingPreviewPredictor

class TrainingPreviewHandler(server.SechenovServerHandler):
    def do_GET(self):
        path=self.path.split('?',1)[0]
        if path in ('/api/health','/api/schema','/api/benchmarks'):
            predictor=server.BRFSS
            if path=='/api/health':result=dict(models_loaded=True,dataset='model_training_simulation',model_version=predictor.version,data_origin='synthetic_simulation',experiment=predictor.experiment)
            elif path=='/api/schema':result=predictor.public_schema()
            else:result=predictor.benchmarks()
            encoded=json.dumps(result,allow_nan=False).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(encoded)));self.end_headers();self.wfile.write(encoded);return
        super().do_GET()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8001);a=p.parse_args();sys.stdout.reconfigure(encoding='utf-8')
    server.BRFSS=TrainingPreviewPredictor(ROOT);server.MODELS_LOADED=True
    print(f'NERVA Stress Detection System — Main Website ({server.BRFSS.experiment}): http://localhost:{a.port}/#predictions',flush=True)
    http=server.ThreadingHTTPServer(('127.0.0.1',a.port),TrainingPreviewHandler)
    try:http.serve_forever()
    except KeyboardInterrupt:http.server_close()
