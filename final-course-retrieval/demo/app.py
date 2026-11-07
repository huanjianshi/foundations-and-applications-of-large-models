"""Local-only demo. Run from any working directory with Python -B."""
import sys,json,argparse,time,threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from retrieval import ROOT,read_json
from advanced_retrieval import TextEncoder,Reranker,AdvancedEngine,KeywordEngine

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bge-path',type=Path,default=ROOT/'models/bge-small-zh-v1.5');parser.add_argument('--reranker-path',type=Path,default=ROOT/'models/bge-reranker-base');parser.add_argument('--device',choices=['cuda','cpu']);parser.add_argument('--lexical-only',action='store_true');parser.add_argument('--port',type=int,default=8765);args=parser.parse_args()
    cfg=read_json(ROOT/'results_v2/selected_config.json')
    if args.device:cfg['device']=args.device
    cards=read_json(ROOT/'data/cards.json')
    engine=KeywordEngine(cards,cfg['baseline_bm25_title_weight']) if args.lexical_only else AdvancedEngine(TextEncoder(args.bge_path,device=cfg['device']),Reranker(args.reranker_path,device=cfg['device']),cards,cfg);lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,fmt,*values): pass
        def send(self,status,data,kind='application/json; charset=utf-8'):
            raw=data if isinstance(data,bytes) else json.dumps(data,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            path=urlparse(self.path).path
            if path=='/': return self.send(200,(ROOT/'demo/index.html').read_bytes(),'text/html; charset=utf-8')
            if path=='/api/status': return self.send(200,dict(cards=len(engine.cards),dimension=0 if args.lexical_only else engine.encoder.dimension,device='CPU' if args.lexical_only else 'CPU + '+cfg['device'].upper(),lexical_only=args.lexical_only))
            if path=='/api/summary':
                f=ROOT/'results_v2/audited_summary.json'
                return self.send(200,[dict(**r,split='test') for r in read_json(f)] if f.exists() else [])
            return self.send(404,{'error':'页面不存在。'})
        def do_POST(self):
            if urlparse(self.path).path!='/api/search': return self.send(404,{'error':'接口不存在。'})
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=16384: return self.send(400,{'error':'请求长度不合法。'})
                body=json.loads(self.rfile.read(n));q=body.get('query');k=body.get('k',5);method=body.get('method','bm25')
                with lock:
                    start=time.perf_counter();result=engine.search(q,method,k);result['elapsed_ms']=(time.perf_counter()-start)*1000
                return self.send(200,result)
            except (ValueError,TypeError,AttributeError) as exc: return self.send(400,{'error':str(exc)})
            except Exception: return self.send(500,{'error':'检索失败，请查看本机服务状态。'})
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Local: http://127.0.0.1:{args.port}',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()

if __name__=='__main__': main()
