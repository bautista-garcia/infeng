from __future__ import annotations

import argparse
import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from runtime import InferenceEngine  # noqa: E402


class Handler(SimpleHTTPRequestHandler):
    root: Path
    engine = None
    loading = False
    args: argparse.Namespace

    @staticmethod
    def load_model():
        if Handler.engine is not None:
            return
        Handler.loading = True
        print("loading model on mps with float16", flush=True)
        Handler.engine = InferenceEngine(Handler.args.weights, Handler.args.tokenizer)
        Handler.loading = False
        print(
            f"model loaded on {Handler.engine.device} with {Handler.engine.dtype}; missing={len(Handler.engine.report['missing'])} unexpected={len(Handler.engine.report['unexpected'])}",
            flush=True,
        )

    def do_GET(self):
        if self.path == "/":
            self.path = "/index.html"
        if self.path == "/api/status":
            loaded = Handler.engine is not None
            body = {
                "loaded": loaded,
                "loading": Handler.loading,
                "device": ""
                if not loaded
                else str(Handler.engine.device),
            }
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        return super().do_GET()

    def end_headers(self):
        self.send_header("cache-control", "no-store")
        super().end_headers()

    def do_POST(self):
        if self.path == "/api/load":
            body = {"ok": True}
            if Handler.engine is None and not Handler.loading:
                import threading

                threading.Thread(target=Handler.load_model, daemon=True).start()
                body = {"ok": True, "loading": True}
            elif Handler.loading:
                body = {"ok": True, "loading": True}
            else:
                body = {"ok": True, "loaded": True}
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path != "/api/chat":
            self.send_error(404)
            return
        data = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        try:
            print("chat request received", flush=True)

            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("cache-control", "no-cache")
            self.send_header("connection", "keep-alive")
            self.end_headers()

            if Handler.engine is None:
                event = json.dumps({"status": "Loading model..."})
                self.wfile.write(f"data: {event}\n\n".encode())
                self.wfile.flush()
            Handler.load_model()

            event = json.dumps({"status": "Generating..."})
            self.wfile.write(f"data: {event}\n\n".encode())
            self.wfile.flush()

            messages = data.get("messages")
            if messages is None:
                messages = [{"role": "user", "content": data["message"]}]
            stop_ids = [
                Handler.engine.tokenizer.eos_token_id,
                Handler.engine.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            ]

            thinking_enabled = data.get("thinking", Handler.args.thinking)
            temperature = data.get("temperature", Handler.args.temperature)
            top_p = data.get("top_p", Handler.args.top_p)
            top_k = data.get("top_k", Handler.args.top_k)
            gen = Handler.engine.generate(
                messages,
                max_new_tokens=Handler.args.max_new_tokens,
                thinking=thinking_enabled,
                stop_token_ids=stop_ids,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            think_end_str = "</think>"
            accumulated = ""
            thinking = ""
            response_text = ""
            mode = "thinking" if thinking_enabled else "response"

            for token_id in gen:
                if not response_text and not thinking:
                    print("first token generated", flush=True)
                token_text = Handler.engine.tokenizer.decode(
                    [token_id], skip_special_tokens=False
                )
                accumulated += token_text

                if think_end_str in accumulated and mode == "thinking":
                    idx = accumulated.index(think_end_str)
                    thinking = accumulated[:idx]
                    accumulated = accumulated[idx + len(think_end_str) :]
                    mode = "response"

                if mode == "thinking":
                    thinking = accumulated
                else:
                    response_text = accumulated

                event = json.dumps(
                    {
                        "token": token_text,
                        "mode": mode,
                        "thinking": thinking,
                        "response": response_text,
                    }
                )
                self.wfile.write(f"data: {event}\n\n".encode())
                self.wfile.flush()

            response_text = accumulated
            done = json.dumps(
                {"done": True, "thinking": thinking, "response": response_text}
            )
            self.wfile.write(f"data: {done}\n\n".encode())
            self.wfile.flush()
            print("chat generation finished", flush=True)
        except Exception as exc:
            error = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
            self.wfile.write(f"data: {error}\n\n".encode())
            self.wfile.flush()

    def translate_path(self, path):
        return str(Handler.root / path.split("?", 1)[0].lstrip("/"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--weights", default="weights/Qwen3.5-9B-UD-Q4_K_XL.gguf"
    )
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--preload", action="store_true")
    Handler.args, Handler.root = parser.parse_args(), Path(__file__).parent
    Handler.args.temperature = (
        Handler.args.temperature
        if Handler.args.temperature is not None
        else 0.6
        if Handler.args.thinking
        else 0.7
    )
    Handler.args.top_p = (
        Handler.args.top_p
        if Handler.args.top_p is not None
        else 0.95
        if Handler.args.thinking
        else 0.8
    )
    print(f"http://{Handler.args.host}:{Handler.args.port}")
    if Handler.args.preload:
        Handler.load_model()
    ThreadingHTTPServer((Handler.args.host, Handler.args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
