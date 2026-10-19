"""
Marketing module for the salon agent (standard library only).

What it does (all permission-based: only people who already talked to the salon bot):
  1. appointment reminders        (transactional, ~24h before the visit)
  2. review requests              (a few hours after the visit)
  3. win-back messages            ("it has been a while", after N days without a visit)
  4. content drafts for the owner (Instagram / Bale-channel captions from salon.json)
  5. opt-out list                 (customer writes «لغو تبلیغات» -> no more promos)

Every message is logged in `marketing_log` with UNIQUE(kind, ref), so the same message is
never sent twice (at-most-once: if sending fails it is NOT retried forever).

Optional block in salon.json (all keys optional, defaults below):
  "marketing": {"send_from": 9, "send_to": 21, "reminder_hours": 24, "winback_days": 60,
                "winback_cooldown_days": 90, "review_after_hours": 3, "max_per_cycle": 20,
                "winback_offer": "", "review_link": ""}

CLI:
  py marketing.py draft "کراتین"     generate a post draft
  py marketing.py cycle              dry run: show what WOULD be sent now (nothing is sent/logged)
  py marketing.py inactive           list inactive customers
"""
import datetime
import json
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request

import db
import agent_salon as A

CFG = {"send_from": 9, "send_to": 21, "reminder_hours": 24, "winback_days": 60,
       "winback_cooldown_days": 90, "review_after_hours": 3, "max_per_cycle": 20,
       "winback_offer": "", "review_link": ""}
CFG.update(A.SALON.get("marketing", {}))

OPT_OUT_PHRASES = {"لغو تبلیغات", "لغو عضویت", "توقف تبلیغات", "stop", "/stop"}
OPT_IN_PHRASES = {"فعال تبلیغات", "فعال سازی تبلیغات", "/subscribe"}
OPT_FOOTER = "برای عدم دریافت پیام‌های تبلیغاتی بنویسید «لغو تبلیغات»."

SCHEMA = """
CREATE TABLE IF NOT EXISTS marketing_log(
  id      INTEGER PRIMARY KEY AUTOINCREMENT,
  ts      TEXT NOT NULL,
  conv_id TEXT NOT NULL,
  kind    TEXT NOT NULL,
  ref     TEXT NOT NULL,
  text    TEXT,
  UNIQUE(kind, ref)
);
CREATE TABLE IF NOT EXISTS marketing_optout(
  conv_id TEXT PRIMARY KEY,
  ts      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS marketing_drafts(
  id     INTEGER PRIMARY KEY AUTOINCREMENT,
  ts     TEXT NOT NULL,
  source TEXT NOT NULL,
  topic  TEXT NOT NULL,
  text   TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'draft'
);
"""


def con():
    c = db.get_con()
    c.executescript(SCHEMA)         # IF NOT EXISTS: cheap and safe on every thread's connection
    return c


# ---------------------------------------------------------------- opt-out
def is_opt_out_text(text: str) -> bool:
    return A.norm(text) in {A.norm(p) for p in OPT_OUT_PHRASES}


def is_opt_in_text(text: str) -> bool:
    return A.norm(text) in {A.norm(p) for p in OPT_IN_PHRASES}


def opted_out(conv_id: str) -> bool:
    return con().execute("SELECT 1 FROM marketing_optout WHERE conv_id=?", (conv_id,)).fetchone() is not None


def set_opt_out(conv_id: str, flag: bool) -> None:
    c = con()
    if flag:
        c.execute("INSERT OR IGNORE INTO marketing_optout(conv_id, ts) VALUES(?,?)",
                  (conv_id, db.ts(datetime.datetime.now())))
    else:
        c.execute("DELETE FROM marketing_optout WHERE conv_id=?", (conv_id,))


# ---------------------------------------------------------------- log helpers
def _log(conv_id, kind, ref, text) -> bool:
    """Record a message BEFORE sending. False = this (kind, ref) was already sent."""
    try:
        con().execute("INSERT INTO marketing_log(ts,conv_id,kind,ref,text) VALUES(?,?,?,?,?)",
                      (db.ts(datetime.datetime.now()), conv_id, kind, str(ref), text))
        return True
    except sqlite3.IntegrityError:
        return False


def _already(kind, ref) -> bool:
    return con().execute("SELECT 1 FROM marketing_log WHERE kind=? AND ref=?", (kind, str(ref))).fetchone() is not None


def in_send_window(now) -> bool:
    return CFG["send_from"] <= now.hour < CFG["send_to"]


def _pick(options, n):
    return options[n % len(options)]


def _first_name(full: str) -> str:
    parts = str(full).split()
    return parts[0] if parts else ""


# ---------------------------------------------------------------- who should get what
def due_reminders(now):
    lo, hi = db.minutes_iso(now + datetime.timedelta(hours=1)), \
             db.minutes_iso(now + datetime.timedelta(hours=CFG["reminder_hours"]))
    out = []
    for b in con().execute("SELECT * FROM bookings WHERE status='booked' AND start>? AND start<=?", (lo, hi)):
        start = datetime.datetime.fromisoformat(b["start"])
        created = datetime.datetime.fromisoformat(b["created_at"])
        if start - created < datetime.timedelta(hours=6):      # they just booked: no reminder needed
            continue
        if not _already("reminder", b["id"]):
            out.append(b)
    return out


def review_candidates(now):
    lo = db.minutes_iso(now - datetime.timedelta(days=3))
    hi = db.minutes_iso(now)
    out = []
    for b in con().execute("SELECT * FROM bookings WHERE status='booked' AND start>? AND start<=?", (lo, hi)):
        end = datetime.datetime.fromisoformat(b["start"]) + datetime.timedelta(minutes=b["duration_min"])
        if end <= now - datetime.timedelta(hours=CFG["review_after_hours"]) \
                and not _already("review", b["id"]) and not opted_out(b["conv_id"]):
            out.append(b)
    return out


def inactive_customers(now, days=None):
    """Customers whose LAST booked visit is older than `days` and who have nothing booked later."""
    days = days or CFG["winback_days"]
    cutoff = db.minutes_iso(now - datetime.timedelta(days=days))
    cooldown = db.ts(now - datetime.timedelta(days=CFG["winback_cooldown_days"]))
    rows = con().execute(
        "SELECT conv_id, MAX(start) AS last_start, "
        "(SELECT customer_name FROM bookings b2 WHERE b2.conv_id=b.conv_id ORDER BY id DESC LIMIT 1) AS name, "
        "(SELECT services FROM bookings b3 WHERE b3.conv_id=b.conv_id AND b3.status='booked' ORDER BY id DESC LIMIT 1) AS services "
        "FROM bookings b WHERE status IN ('booked','pending_payment') GROUP BY conv_id HAVING last_start < ?",
        (cutoff,)).fetchall()
    out = []
    for r in rows:
        if r["services"] is None or opted_out(r["conv_id"]):
            continue
        recent = con().execute("SELECT 1 FROM marketing_log WHERE conv_id=? AND kind='winback' AND ts>?",
                               (r["conv_id"], cooldown)).fetchone()
        if not recent:
            out.append(r)
    return out


# ---------------------------------------------------------------- message texts
def reminder_text(b) -> str:
    intro = _pick(["یادآوری نوبت 🌸", "سلام! یادآوری نوبت شما 🌸"], b["id"])
    return "\n".join([intro, f"نوبت شما {A.booking_when(b)} است.", f"• خدمات: {b['services']}",
                      f"• آدرس: {A.SALON['address']}",
                      "اگر نمی‌توانید بیایید لطفاً همین‌جا بنویسید تا هماهنگ کنیم."])


def review_text(b) -> str:
    name = _first_name(b["customer_name"])
    base = _pick([f"{name} عزیز، امیدواریم از خدمات سالن راضی بوده باشید 🌷",
                  f"{name} عزیز، ممنون که به ما اعتماد کردید 🌷"], b["id"])
    lines = [base, "اگر وقت داشتید، تجربه‌تان را از ۱ تا ۵ امتیاز بدهید و هر نکته‌ای دارید همین‌جا بنویسید؛ نظر شما به ما کمک می‌کند بهتر شویم."]
    if CFG.get("review_link"):
        lines.append(f"اگر راضی بودید، ثبت نظر در این لینک هم لطف بزرگی است: {CFG['review_link']}")
    lines.append(OPT_FOOTER)
    return "\n".join(lines)


def winback_text(r) -> str:
    name = _first_name(r["name"])
    n = sum(map(ord, r["conv_id"]))
    lines = [_pick([f"{name} عزیز، مدتی است ندیدیمتان 🌸", f"سلام {name} عزیز! دلمان برایتان تنگ شده 🌸"], n),
             f"آخرین بار {r['services']} داشتید. اگر دوست دارید دوباره نوبت بگیرید، همین‌جا بنویسید تا زمان‌های خالی را بگویم."]
    if CFG.get("winback_offer"):
        lines.append(CFG["winback_offer"])
    lines.append(OPT_FOOTER)
    return "\n".join(lines)


# ---------------------------------------------------------------- the scheduler tick
def run_cycle(push, now=None, dry=False) -> dict:
    """Send everything that is due. `push(conv_id, text)` is supplied by the messenger adapter."""
    now = now or datetime.datetime.now()
    stats = {"reminder": 0, "review": 0, "winback": 0}
    if not dry and not in_send_window(now):
        return stats
    budget = [CFG["max_per_cycle"]]

    def send(conv, kind, ref, text):
        if budget[0] <= 0:
            return
        if dry:
            print(f"[{kind}] -> {conv}\n{text}\n")
            stats[kind] += 1
            budget[0] -= 1
            return
        if not _log(conv, kind, ref, text):
            return
        budget[0] -= 1
        try:
            push(conv, text)
            stats[kind] += 1
        except Exception as e:                                  # blocked bot, network error...: logged, not retried
            print(f"[marketing] send failed ({kind}, {conv}): {e}")
        time.sleep(1)                                           # be gentle with the messenger API

    for b in due_reminders(now):
        send(b["conv_id"], "reminder", b["id"], reminder_text(b))
    for b in review_candidates(now):
        send(b["conv_id"], "review", b["id"], review_text(b))
    for r in inactive_customers(now):
        send(r["conv_id"], "winback", f"{r['conv_id']}:{now:%Y-%m-%d}", winback_text(r))
    return stats


# ---------------------------------------------------------------- content drafts (for the owner to approve)
def _ollama(system: str, prompt: str) -> str:
    payload = {"model": A.MODEL, "stream": False, "think": False,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
               "options": {"num_ctx": A.NUM_CTX, "temperature": 0.7}}
    req = urllib.request.Request(A.OLLAMA_URL, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with A._opener.open(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))["message"]["content"].strip()


def _service_facts(topic: str):
    hits = A.find_services(topic)
    return hits if hits else []


def _numbers_ok(text: str, facts) -> bool:
    """Every number in the post must come from the salon data (prices, phone)."""
    allowed = set()
    for s in facts:
        allowed |= {str(s["price_toman"]), str(s["duration_min"])}
    allowed |= {re.sub(r"\D", "", A.norm(A.SALON["phone"]))}
    for tok in re.findall(r"\d[\d,،]*", A.norm(text)):
        if re.sub(r"\D", "", tok) not in allowed and len(re.sub(r"\D", "", tok)) >= 2:
            return False
    return True


def generate_post(topic: str, source: str = "manual", platform: str = "instagram") -> str:
    facts = _service_facts(topic)
    if facts:
        facts_txt = "\n".join(f"- {s['name']} | price {A.fmt_toman(s['price_toman'])} | duration {A.fmt_duration(s['duration_min'])}"
                              for s in facts)
    else:
        facts_txt = "(no specific service; write a general post about the salon)"
    system = ("You write social-media posts in natural, warm, professional Persian for a beauty salon. "
              "Rules: 60-110 words; sound human, not like an advert; no exaggerated or medical claims; "
              "no invented discounts, offers, testimonials or statistics; use ONLY the facts given; "
              "end with a short invitation to book (via this chat or by phone); add 3-5 relevant hashtags. "
              "Output only the post text.")
    prompt = (f"Salon: {A.SALON['name']}\nAddress: {A.SALON['address']}\nPhone: {A.SALON['phone']}\n"
              f"Platform: {platform}\nTopic: {topic}\nFacts:\n{facts_txt}")
    text = _ollama(system, prompt)
    if not _numbers_ok(text, facts):
        text += "\n\n⚠️ این متن عددی دارد که در اطلاعات سالن نیست؛ قبل از انتشار قیمت‌ها و اعداد را چک کنید."
    con().execute("INSERT INTO marketing_drafts(ts,source,topic,text) VALUES(?,?,?,?)",
                  (db.ts(datetime.datetime.now()), source, topic, text))
    return text


def maybe_weekly_draft(send_to_owner, now=None) -> bool:
    """Once a week: draft a post about the service we promoted least recently and send it to the owner."""
    now = now or datetime.datetime.now()
    if not in_send_window(now):
        return False
    since = db.ts(now - datetime.timedelta(days=7))
    if con().execute("SELECT 1 FROM marketing_drafts WHERE source='weekly' AND ts>?", (since,)).fetchone():
        return False
    last = {r["topic"]: r["ts"] for r in con().execute(
        "SELECT topic, MAX(ts) ts FROM marketing_drafts GROUP BY topic")}
    svc = min(A.SALON["services"], key=lambda s: last.get(s["name"], ""))
    text = generate_post(svc["name"], source="weekly")
    send_to_owner(f"📝 پیش‌نویس پست این هفته ({svc['name']}):\n\n{text}")
    return True


def stats_text() -> str:
    c = con()
    rows = c.execute("SELECT kind, COUNT(*) n FROM marketing_log GROUP BY kind").fetchall()
    opt = c.execute("SELECT COUNT(*) FROM marketing_optout").fetchone()[0]
    lines = [f"• {r['kind']}: {A.fa(r['n'])}" for r in rows] or ["• هنوز پیامی ارسال نشده"]
    return "آمار بازاریابی:\n" + "\n".join(lines) + f"\n• لغو تبلیغات: {A.fa(opt)} نفر"


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "draft":
        print(generate_post(" ".join(sys.argv[2:]) or "سالن"))
    elif cmd == "cycle":
        print(run_cycle(lambda c, t: None, dry=True))
    elif cmd == "inactive":
        for r in inactive_customers(datetime.datetime.now()):
            print(dict(r))
    else:
        print(__doc__)
