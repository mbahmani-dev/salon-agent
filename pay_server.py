"""
Tiny web server (standard library) with two pages:

  /fake-pay/<ref>   a FAKE bank page: shows the amount and two buttons (pay / cancel). Dev only.
  /callback?...     the address the customer's browser is sent back to after paying.
                    A REAL gateway will redirect to this same endpoint, so keep it.

It runs in a background thread inside agent_salon.py when PAYMENT_PROVIDER=fake.
For a real gateway this endpoint must be reachable from the customer's browser over public https
(a laptop at home is not), e.g. on a small server or hosting.
"""
import html
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

PORT = 8000

PAGE = """<!doctype html><html lang="fa" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>
<style>body{{font-family:Tahoma,sans-serif;max-width:420px;margin:40px auto;padding:0 16px;line-height:1.9}}
a.b{{display:block;text-align:center;padding:12px;margin:10px 0;border-radius:8px;text-decoration:none;color:#fff}}
.ok{{background:#2e7d32}}.no{{background:#b71c1c}}.box{{border:1px solid #ccc;border-radius:8px;padding:12px}}</style>
</head><body>{body}</body></html>"""


def make_handler(agent):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):          # keep the console clean
            pass

        def _send(self, title: str, body: str, code: int = 200) -> None:
            data = PAGE.format(title=html.escape(title), body=body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path.startswith("/fake-pay/"):
                ref = url.path.rsplit("/", 1)[-1]
                pay = agent.db.get_payment_by_ref(ref)
                if not pay:
                    return self._send("خطا", "<p>پرداخت پیدا نشد.</p>", 404)
                amount = html.escape(agent.fa(agent.fmt_toman(pay["amount_toman"])))
                body = (f'<h3>درگاه پرداخت آزمایشی</h3><div class="box">مبلغ: <b>{amount}</b><br>'
                        f'کد: {html.escape(ref)}</div>'
                        f'<a class="b ok" href="/callback?ref={html.escape(ref)}&status=OK">پرداخت موفق (شبیه‌سازی)</a>'
                        f'<a class="b no" href="/callback?ref={html.escape(ref)}&status=NOK">انصراف</a>'
                        f'<p><small>این صفحه فقط برای تست است و پولی جابه‌جا نمی‌شود.</small></p>')
                return self._send("پرداخت آزمایشی", body)
            if url.path == "/callback":
                # A real gateway sends its own parameter names; add them here when you switch.
                ref = q.get("ref") or q.get("Authority") or q.get("authority") or q.get("token")
                result = agent.on_payment_callback(ref, q)
                color = "ok" if result["ok"] else "no"
                body = (f'<h3>{html.escape(result["text"])}</h3>'
                        f'<p>می‌توانید به گفتگو برگردید؛ نتیجه آنجا هم برایتان ارسال می‌شود.</p>'
                        f'<div class="b {color}" style="padding:6px"></div>')
                return self._send("نتیجه پرداخت", body)
            self._send("404", "<p>صفحه پیدا نشد.</p>", 404)

    return Handler


def start(agent, port: int = PORT):
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(agent))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
