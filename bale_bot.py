"""
Bale adapter for the salon agent (standard library only).

Env vars:
    BALE_TOKEN       bot token from @botfather in Bale          (required)
    OWNER_CHAT_ID    the owner's Bale chat id                   (optional; send /myid to the bot to learn it)

Run:   py bale_bot.py

What it does
  - long-polls Bale (getUpdates) and sends every customer message to agent_salon.run_turn
  - conv_id = "bale:<chat_id>"  (a Telegram adapter later will use "tg:<chat_id>")
  - replaces agent_salon.push_to_customer / notify_owner, so payment confirmations, reminders
    and owner alerts really arrive in Bale
  - handles «لغو تبلیغات» / «فعال تبلیغات» before the model sees the message
  - background thread: marketing.run_cycle every 10 minutes + weekly post draft for the owner
  - owner commands:  /post <topic>   /weekly   /stats   /ad [platform] <topic>
                    /spend <code> <toman>   /adreport   /myid (anyone)
"""
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

import agent_salon as A
import ads
import marketing

TOKEN = os.environ.get("BALE_TOKEN", "")
OWNER = int(os.environ["OWNER_CHAT_ID"]) if os.environ.get("OWNER_CHAT_ID", "").isdigit() else None
API = f"https://tapi.bale.ai/bot{TOKEN}"
CYCLE_SECONDS = 600

LOCK = threading.RLock()          # run_turn uses global state: one conversation turn at a time
CONVS = {}                        # conv_id -> message list for the model
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if os.environ.get("BALE_NO_PROXY") \
    else urllib.request.build_opener()


def api(method: str, http_timeout: int = 40, **params) -> dict:
    req = urllib.request.Request(f"{API}/{method}", data=json.dumps(params).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with _opener.open(req, timeout=http_timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def send_message(chat_id: int, text: str) -> None:
    for i in range(0, len(text), 4000):                       # Bale limit is 4096 characters
        api("sendMessage", chat_id=chat_id, text=text[i:i + 4000])


def get_conv(conv_id: str) -> list:
    return CONVS.setdefault(conv_id, [{"role": "system", "content": ""}])


# ---------------------------------------------------------------- replace the CLI hooks in agent_salon
def push_to_customer(conv_id: str, text: str) -> None:
    if not conv_id.startswith("bale:"):
        print(f"\n[PUSH -> {conv_id}]\n{text}\n")
        return
    send_message(int(conv_id.split(":", 1)[1]), text)
    with LOCK:                                                # the model must know what was just sent
        get_conv(conv_id).append({"role": "assistant", "content": text})


_orig_notify = A.notify_owner


def notify_owner(text: str) -> None:
    _orig_notify(text)                                        # keeps print + database record
    if OWNER:
        try:
            send_message(OWNER, "🔔 " + text)
        except Exception as e:
            print(f"[bale] could not notify owner: {e}")


A.push_to_customer = push_to_customer
A.notify_owner = notify_owner


# ---------------------------------------------------------------- handling one incoming message
def reply(chat_id: int, text: str) -> None:
    try:
        send_message(chat_id, text)
    except Exception as e:
        print(f"[bale] send failed: {e}")


def handle_owner_command(chat_id: int, text: str) -> bool:
    cmd, _, arg = text.partition(" ")
    if cmd == "/post":
        reply(chat_id, "در حال نوشتن پیش‌نویس…")
        reply(chat_id, marketing.generate_post(arg.strip() or "سالن"))
    elif cmd == "/weekly":
        reply(chat_id, "در حال نوشتن پیش‌نویس هفتگی…")
        marketing.maybe_weekly_draft(lambda t: reply(chat_id, t), now=A.now().replace(hour=12))
    elif cmd == "/stats":
        reply(chat_id, marketing.stats_text())
    elif cmd == "/ad":                                        # /ad [instagram|telegram|bale|google] <topic>
        words = arg.split()
        plat = words.pop(0) if words and words[0] in ads.PLATFORMS else "instagram"
        reply(chat_id, "در حال نوشتن سه نسخه تبلیغ…")
        try:
            reply(chat_id, ads.create_campaign(" ".join(words) or "سالن", plat))
        except ValueError as e:
            reply(chat_id, f"نشد: {e}")
    elif cmd == "/spend":                                     # /spend ad3a 500000
        parts = arg.split()
        ok = len(parts) == 2 and A.norm(parts[1]).isdigit() and ads.set_spend(parts[0], int(A.norm(parts[1])))
        reply(chat_id, "ثبت شد ✅" if ok else "فرمت: /spend <کد نسخه> <مبلغ به تومان>")
    elif cmd == "/adreport":
        reply(chat_id, ads.report_text())
    else:
        return False
    return True


def handle_message(chat_id: int, text: str) -> None:
    conv = f"bale:{chat_id}"
    if text == "/myid":
        return reply(chat_id, f"chat id: {chat_id}")
    if OWNER and chat_id == OWNER and text.startswith("/") and handle_owner_command(chat_id, text):
        return
    if marketing.is_opt_out_text(text):
        marketing.set_opt_out(conv, True)
        return reply(chat_id, "حتماً. دیگر پیام تبلیغاتی برایتان نمی‌فرستیم؛ یادآوری نوبت‌ها ادامه دارد. "
                              "هر زمان خواستید بنویسید «فعال تبلیغات».")
    if marketing.is_opt_in_text(text):
        marketing.set_opt_out(conv, False)
        return reply(chat_id, "پیام‌های ویژه دوباره برایتان فعال شد 🌸")
    if text == "/start" or text.startswith("/start "):
        parts = text.split(maxsplit=1)
        if len(parts) == 2:
            ads.record_lead(conv, parts[1].strip())           # which ad brought this customer
        text = "سلام"
    with LOCK:
        msgs = get_conv(conv)
        if len(msgs) > 80:                                    # keep memory small; run_turn trims for the model anyway
            msgs[1:] = msgs[-40:]
        msgs.append({"role": "user", "content": text})
        try:
            answer = A.run_turn(msgs, conv)
        except SystemExit as e:                               # chat() raises SystemExit when Ollama is down
            print(f"[bale] model error: {e}")
            answer = "ببخشید، الان مشکلی پیش آمده. چند دقیقه دیگر دوباره پیام بدهید."
        except Exception as e:
            print(f"[bale] error: {e!r}")
            answer = "ببخشید، مشکلی پیش آمد. لطفاً دوباره بنویسید."
    reply(chat_id, answer)


# ---------------------------------------------------------------- background marketing thread
def scheduler() -> None:
    while True:
        try:
            marketing.run_cycle(push_to_customer)
            if OWNER:
                marketing.maybe_weekly_draft(lambda t: send_message(OWNER, t))
        except Exception as e:
            print(f"[marketing] cycle error: {e!r}")
        time.sleep(CYCLE_SECONDS)


# ---------------------------------------------------------------- main loop
def main() -> None:
    if not TOKEN:
        raise SystemExit("Set BALE_TOKEN first (PowerShell:  $env:BALE_TOKEN='...')")
    if os.environ.get("PAYMENT_PROVIDER", "fake").lower() == "fake":
        try:
            import pay_server
            pay_server.start(A)
        except OSError as e:
            print(f"[dev] fake payment page not started: {e}")
    threading.Thread(target=scheduler, daemon=True).start()
    print(f"Bale bot running for {A.SALON['name']} (owner: {OWNER or 'not set'}). Ctrl+C to stop.")
    offset = 0
    while True:
        try:
            data = api("getUpdates", http_timeout=40, offset=offset, timeout=25)
            for upd in data.get("result", []):
                offset = upd["update_id"] + 1
                msg = upd.get("message") or {}
                chat_id = (msg.get("chat") or {}).get("id")
                if not chat_id:
                    continue
                text = (msg.get("text") or "").strip()
                if not text:
                    reply(chat_id, "فعلاً فقط پیام متنی را می‌توانم بخوانم؛ لطفاً درخواستتان را بنویسید.")
                    continue
                handle_message(chat_id, text)
        except KeyboardInterrupt:
            sys.exit(0)
        except Exception as e:
            print(f"[bale] polling error: {e!r}")
            time.sleep(3)


if __name__ == "__main__":
    main()
