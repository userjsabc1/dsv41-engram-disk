"""Quick latency/throughput check against the local vLLM server."""
import json
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"


def call(prompt, max_tokens, out, i):
    body = json.dumps({
        "model": "dsv41",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    t0 = time.perf_counter()
    r = json.load(urllib.request.urlopen(
        urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=600))
    dt = time.perf_counter() - t0
    out[i] = (dt, r["usage"]["prompt_tokens"], r["usage"]["completion_tokens"],
              r["choices"][0]["message"]["content"])


def run(label, prompts, max_tokens):
    out = [None] * len(prompts)
    ths = [threading.Thread(target=call, args=(p, max_tokens, out, i)) for i, p in enumerate(prompts)]
    t0 = time.perf_counter()
    [t.start() for t in ths]
    [t.join() for t in ths]
    wall = time.perf_counter() - t0
    toks = sum(o[2] for o in out)
    ptoks = sum(o[1] for o in out)
    print(f"{label:34s} wall {wall:6.2f}s  prompt_tok {ptoks:6d}  gen_tok {toks:5d}  "
          f"=> {toks / wall:6.1f} tok/s total, {toks / len(out) / wall:5.1f} tok/s/stream")
    return out


base = "请详细解释一下什么是 MoE（混合专家）模型，以及它和稠密模型的区别。"
long_text = "这是一段用于测试长上下文的文字。" * 400  # ~ several thousand tokens

run("warmup (1 stream, 64 tok)", [base], 64)
run("1 stream, 256 tok", [base], 256)
run("4 streams, 256 tok", [base + f"（第{i}次）" for i in range(4)], 256)
run("8 streams, 256 tok", [base + f"（第{i}次）" for i in range(8)], 256)
o = run("long prompt (~4-5k tok), 32 out", [long_text + "\n请用一句话概括以上内容。"], 32)
print("long-prompt answer:", o[0][3][:200].replace("\n", " "))
o = run("long prompt x2 concurrent", [long_text + f"\n（{i}）请用一句话概括以上内容。" for i in range(2)], 32)
