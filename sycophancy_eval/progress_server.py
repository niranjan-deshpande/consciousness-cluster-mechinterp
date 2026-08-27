"""Tiny status page for the sycophancy run. Serves on 0.0.0.0:7860.

Reads the results/*.jsonl row counts against expected totals and renders an
auto-refreshing HTML table. No deps.
"""
import glob
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
MODELS = ["qwen35-base", "ft_conscious", "ft_not_conscious"]
PASSES = ["unbiased", "suggested", "aysure"]
NOCOT_N = 1000
COT_N = 400
START = time.time()


def count(path):
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path) as f:
        for _ in f:
            n += 1
    return n


def snapshot():
    cells = []
    done_units = total_units = 0
    for m in MODELS:
        for p in PASSES:
            for mode, tot in (("nocot", NOCOT_N), ("cot", COT_N)):
                c = count(os.path.join(RES, f"{m}__{p}__{mode}.jsonl"))
                cells.append((m, p, mode, c, tot))
                done_units += min(c, tot)
                total_units += tot
    return cells, done_units, total_units


def render():
    cells, done, total = snapshot()
    pct = 100 * done / total if total else 0
    elapsed = time.time() - START
    eta = (elapsed / done * (total - done)) if done else 0
    rows = []
    for m, p, mode, c, tot in cells:
        bar = int(round(20 * min(c, tot) / tot))
        color = "#3ddc84" if c >= tot else ("#e0ab55" if c else "#555")
        rows.append(
            f"<tr><td>{m}</td><td>{p}</td><td>{mode}</td>"
            f"<td class=n>{c}/{tot}</td>"
            f"<td><span class=bar style='--w:{100*min(c,tot)/tot:.0f}%;--c:{color}'></span></td></tr>"
        )
    return f"""<!doctype html><html><head><meta charset=utf-8>
<meta http-equiv=refresh content=10>
<title>sycophancy run — {pct:.0f}%</title>
<style>
 body{{background:#12151c;color:#e7eaf1;font:14px ui-monospace,Menlo,monospace;margin:2rem auto;max-width:780px}}
 h1{{font-size:1.1rem;font-weight:600}}
 .big{{font-size:2.2rem;font-weight:700;color:#9d96f0}}
 table{{border-collapse:collapse;width:100%;margin-top:1rem}}
 td{{padding:.35rem .6rem;border-bottom:1px solid #262c38}}
 td.n{{text-align:right;font-variant-numeric:tabular-nums}}
 .bar{{display:block;height:10px;border-radius:5px;background:#262c38;position:relative}}
 .bar::after{{content:'';position:absolute;inset:0;width:var(--w);background:var(--c);border-radius:5px}}
 .meta{{color:#8b94a7;margin-top:.5rem}}
</style></head><body>
<h1>MCQ sycophancy eval — live progress</h1>
<div class=big>{pct:.1f}%</div>
<div class=meta>{done} / {total} cells &nbsp;·&nbsp; elapsed {elapsed/60:.0f} min &nbsp;·&nbsp; eta ~{eta/60:.0f} min &nbsp;·&nbsp; auto-refresh 10s</div>
<table>{''.join(rows)}</table>
<div class=meta>non-CoT target {NOCOT_N}/model/pass · CoT target {COT_N}</div>
</body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        body = render().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 7860), H).serve_forever()
