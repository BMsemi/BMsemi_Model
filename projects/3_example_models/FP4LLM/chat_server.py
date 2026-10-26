"""Local Qwen chat UI. Run with the llmqwen Python environment."""
import argparse
import gc
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
lock = threading.Lock()
cancel = threading.Event()
state = {"status": "idle", "text": "", "detail": "Ready to chat", "mode": "standard"}
model = tokenizer = loaded_mode = None


def update(**values):
    with lock:
        state.update(values)


def generate(messages, mode, limit):
    global model, tokenizer, loaded_mode
    started = time.time()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(4)
        if loaded_mode != mode:
            update(detail="Loading Qwen3-1.7B from local disk…")
            model = None
            loaded_mode = None
            gc.collect()
            tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                args.model, local_files_only=True, dtype=torch.bfloat16,
                attn_implementation="eager").eval()
            if mode not in ("standard", "fpga"):
                from qwen_frontend import map_linears, install_attention
                config = json.loads(args.crossbar_config.read_text())
                config["noise_config"]["enabled"] = mode == "noisy"
                config["reliability_config"]["enabled"] = mode == "noisy"
                update(detail="Programming all ReRAM projections and attention crossbars…")
                map_linears(model, config, 1234)
                install_attention(model, config, 1234)
            loaded_mode = mode
        ids = tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
            return_tensors="pt", return_dict=False)
        if ids.shape[1] > (2048 if mode == "standard" else 128):
            raise ValueError("Conversation is too long for this mode. Start a new chat or shorten the message.")
        hardware = None
        if mode == "fpga":
            if cancel.is_set():
                update(status="done", detail="Stopped before FPGA execution")
                return
            from fpga_chat_bridge import prefill
            hardware, report = prefill(args.model, ids[0].tolist(), lambda detail: update(detail=detail))
            update(hardware=report)
        generated = []
        cache = None
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        update(status="generating", detail="Generating on CPU…")
        with torch.inference_mode():
            for _ in range(limit):
                if cancel.is_set():
                    break
                if hardware is not None and cache is None:
                    # Replace only the first prompt block; seed its actual hardware K/V.
                    # Restore the normal block before CPU autoregressive decoding.
                    layer = model.model.layers[0]
                    original = layer.forward
                    def hardware_forward(hidden_states, past_key_values=None, **kwargs):
                        if past_key_values is None:
                            raise RuntimeError("FPGA handoff requires an active KV cache")
                        key = torch.from_numpy(hardware['key_cache']).unsqueeze(0).to(hidden_states)
                        value = torch.from_numpy(hardware['value_cache']).unsqueeze(0).to(hidden_states)
                        past_key_values.update(key, value, 0)
                        return torch.from_numpy(hardware['output']).unsqueeze(0).to(hidden_states)
                    layer.forward = hardware_forward
                    try:
                        result = model(ids, use_cache=True)
                    finally:
                        layer.forward = original
                else:
                    result = model(ids, past_key_values=cache, use_cache=True)
                cache = result.past_key_values
                token = result.logits[:, -1].argmax(-1, keepdim=True)
                if token.item() in eos:
                    break
                generated.append(token.item())
                update(text=tokenizer.decode(generated, skip_special_tokens=True),
                       detail=f"{len(generated)} tokens · {time.time()-started:.0f}s · " + ("FPGA prefill + CPU decode" if mode == "fpga" else "CPU"))
                ids = token
        update(status="done", detail=f"{'Stopped' if cancel.is_set() else 'Complete'} · {len(generated)} tokens · {time.time()-started:.1f}s" + (" · physical FPGA block-0 prefill + CPU continuation/decode" if mode == "fpga" else ""))
    except Exception as exc:
        update(status="error", detail=str(exc))


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, data, content_type="application/json"):
        body = data if isinstance(data, bytes) else json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self.reply(200, (ROOT / "web" / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/logs":
            self.reply(200, (ROOT / "web" / "logs.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/logs":
            sources = {
                "Current server / crossbar activity": "results/chat_server.log",
                "FPGA hardware summary (saved run)": "results/qwen_model_execution_x2/fpga_summary.json",
                "FPGA + software continuation (saved run)": "results/qwen_model_execution_x2/run.log",
                "ReRAM noise comparison (completed CPU simulation)": "results/qwen_noise_comparison/run.log",
                "Crossbar configuration": "configs/qwen_crossbar_noise.json",
            }
            recent = sorted((ROOT/'results/fpga_chat').glob('*/fpga_summary.json'), key=lambda p: p.stat().st_mtime, reverse=True)
            for path in recent[:3]:
                sources['Physical FPGA chat run '+path.parent.name[:8]] = str(path.relative_to(ROOT))
            logs = []
            for title, relative in sources.items():
                path = ROOT / relative
                if path.exists():
                    logs.append({"title": title, "path": relative,
                                 "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)),
                                 "text": path.read_text(encoding="utf-8", errors="replace")[-100000:]})
            with lock:
                snapshot = dict(state)
            self.reply(200, {"logs": logs, "chat": snapshot})
        elif self.path == "/api/status":
            with lock:
                snapshot = dict(state)
            self.reply(200, snapshot)
        else:
            self.reply(404, {"error": "Not found"})

    def do_POST(self):
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{self.headers.get('Host')}":
            return self.reply(403, {"error": "Local requests only"})
        if self.path == "/api/stop":
            cancel.set()
            return self.reply(200, {"ok": True})
        if self.path != "/api/chat":
            return self.reply(404, {"error": "Not found"})
        try:
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 32000:
                raise ValueError("Message is too large or empty")
            data = json.loads(self.rfile.read(size))
            mode = data.get("mode", "standard")
            messages = data["messages"]
            if mode not in ("standard", "clean", "noisy", "fpga"):
                raise ValueError("Unknown mode")
            if not isinstance(messages, list) or not 1 <= len(messages) <= 40:
                raise ValueError("Start a new chat: history limit reached")
            if any(m.get("role") not in ("user", "assistant") or not isinstance(m.get("content"), str) for m in messages):
                raise ValueError("Invalid messages")
            limit = min(max(int(data.get("max_tokens", 128)), 1), 256 if mode in ("standard", "fpga") else 8)
            with lock:
                if state["status"] in ("loading", "generating"):
                    return self.reply(409, {"error": "A response is already running"})
                state.update(status="loading", text="", detail="Preparing model…", mode=mode, hardware=None)
                cancel.clear()
            threading.Thread(target=generate, args=(messages, mode, limit), daemon=True).start()
            self.reply(202, {"ok": True})
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            self.reply(400, {"error": str(exc)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="D:/hf_models/Qwen--Qwen3-1.7B")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--crossbar-config", type=Path, default=ROOT / "configs/qwen_crossbar_noise.json")
    args = parser.parse_args()
    print(f"Qwen chat: http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
