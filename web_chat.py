import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
lock = threading.Lock()
engine = None

def answer(question, mode):
    global engine
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    if engine is None:
        if not torch.cuda.is_available():
            raise ValueError('当前环境无法使用 CUDA，请检查显卡。')
        adapter = ROOT / 'run_150epochs' / 'adapter'
        tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(ROOT / 'models' / 'Qwen3-0.6B', local_files_only=True, torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, attn_implementation='sdpa')
        model = PeftModel.from_pretrained(base, adapter, local_files_only=True).to('cuda').eval()
        engine = tokenizer, model
    tokenizer, model = engine
    modes = json.loads((ROOT / 'mode_instructions.json').read_text(encoding='utf-8'))
    prompt = tokenizer.apply_chat_template([{'role':'user','content':modes[mode]+'\n\n'+question}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to('cuda')
    if inputs.input_ids.shape[1] > 1024:
        raise ValueError('问题过长，请缩短后重试。')
    with torch.inference_mode():
        result = model.generate(**inputs, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    tokens = result[0, inputs.input_ids.shape[1]:]
    return {'answer':tokenizer.decode(tokens, skip_special_tokens=True).strip(), 'truncated':len(tokens)==512 and int(tokens[-1]) != tokenizer.eos_token_id}

class Handler(BaseHTTPRequestHandler):
    def respond(self, status, data, content_type='application/json; charset=utf-8'):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        if self.path != '/':
            return self.respond(404,b'{}')
        self.respond(200,(ROOT/'web.html').read_bytes(),'text/html; charset=utf-8')
    def do_POST(self):
        if self.path != '/api/chat':
            return self.respond(404,b'{}')
        if self.headers.get('Origin') not in (None, 'http://127.0.0.1:7860', 'http://localhost:7860'):
            return self.respond(403,b'{}')
        try:
            length = int(self.headers.get('Content-Length','0'))
            if not 0 < length <= 16384:
                raise ValueError('请求过大或为空。')
            data = json.loads(self.rfile.read(length))
            question, mode = data.get('question'), data.get('mode')
            if not isinstance(question,str) or not question.strip() or mode not in ('通俗','正式','幽默'):
                raise ValueError('请输入问题并选择回答模式。')
            if not lock.acquire(blocking=False):
                return self.respond(409,json.dumps({'error':'正在回答，请稍后再试。'},ensure_ascii=False).encode())
            try:
                result = answer(question.strip(),mode)
            finally:
                lock.release()
            self.respond(200,json.dumps(result,ensure_ascii=False).encode())
        except ValueError as exc:
            self.respond(400,json.dumps({'error':str(exc)},ensure_ascii=False).encode())
        except Exception:
            import traceback
            traceback.print_exc()
            self.respond(500,json.dumps({'error':'模型加载或生成失败，请查看终端中的错误。'},ensure_ascii=False).encode())

if __name__ == '__main__':
    server = ThreadingHTTPServer(('127.0.0.1',7860),Handler)
    print('本地网页：http://127.0.0.1:7860；按 Ctrl+C 退出。首次提问时加载模型。',flush=True)
    webbrowser.open('http://127.0.0.1:7860')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
