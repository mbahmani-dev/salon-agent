"""
Ads module for the salon agent (standard library only).

What it does
  1. writes 3 ad variants (A/B/C, different angles) for a service, from the facts in salon.json
  2. checks every variant: numbers must come from salon.json, no risky claims, length limits
  3. suggests targeting (area, age, interests, schedule)
  4. gives each variant its OWN tracked link:  https://ble.ir/<bot>?start=<code>
     -> when a customer opens the bot through that link we remember which ad brought them
  5. report: ad -> leads -> bookings -> revenue -> cost per lead / per booking

It does NOT run paid campaigns by itself (that needs your ad account). You paste the copy into
Instagram / Telegram Ads / Google..., enter what you spent with /spend, and the report does the rest.

Optional block in salon.json (all keys optional):
  "ads": {"city": "تهران", "area": "ونک", "radius_km": 5, "gender": "زنان", "age": "22-45",
          "offer": "", "min_leads_for_winner": 20}
  ("offer" = a REAL offer text. If empty, the ads may not mention any discount or free service.)

Env:  BALE_BOT_USERNAME   (for the tracked links)

CLI:
  py ads.py new "کراتین" [instagram|telegram|bale|google]
  py ads.py report
  py ads.py spend <code> <amount_toman>
"""
import datetime
import os
import re
import sqlite3
import sys

import db
import agent_salon as A
import marketing

CFG = {"city": "", "area": "", "radius_km": 5, "gender": "زنان", "age": "22-45", "offer": "",
       "min_leads_for_winner": 20}
CFG.update(A.SALON.get("ads", {}))

PLATFORMS = {"instagram": 300, "telegram": 160, "bale": 160, "google": 90}   # max body length; verify with each platform's current rules
HEADLINE_MAX = 40
ANGLES = {
    "A": "the RESULT the customer wants (how she will look/feel), concrete and calm",
    "B": "clarity and convenience: real price and real duration from the facts, booking in one chat",
    "C": "an occasion or lifestyle moment (a wedding, a holiday, a new season, treating yourself), warm and natural",
}
RISKY = ["تضمین", "بهترین", "ارزان‌ترین", "ارزان ترین", "100%", "۱۰۰٪", "100٪", "معجزه", "درمان"]
OFFER_WORDS = ["تخفیف", "رایگان", "هدیه", "مجانی"]

INTERESTS = {
    "مو": ["مراقبت از مو", "رنگ مو", "سالن زیبایی"],
    "رنگ": ["رنگ مو", "مدل مو", "سالن زیبایی"],
    "مش": ["مدل مو", "رنگ مو", "سالن زیبایی"],
    "کراتین": ["مراقبت از مو", "پروتئین تراپی", "سالن زیبایی"],
    "ناخن": ["مانیکور", "طراحی ناخن", "زیبایی"],
    "صورت": ["مراقبت از پوست", "پاکسازی پوست", "زیبایی"],
    "عروس": ["عروس", "مزون", "عروسی"],
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS ad_campaigns(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, topic TEXT NOT NULL,
  platform TEXT NOT NULL, goal TEXT, budget_toman INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS ad_variants(
  code TEXT PRIMARY KEY, campaign_id INTEGER NOT NULL REFERENCES ad_campaigns(id),
  label TEXT NOT NULL, headline TEXT, body TEXT, cta TEXT, warnings TEXT, spend_toman INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS ad_leads(
  conv_id TEXT PRIMARY KEY, code TEXT NOT NULL, ts TEXT NOT NULL
);
"""


def con():
    c = db.get_con()
    c.executescript(SCHEMA)
    return c


def bot_link(code: str) -> str:
    bot = os.environ.get("BALE_BOT_USERNAME", "YOUR_BOT")
    return f"https://ble.ir/{bot}?start={code}"


# ---------------------------------------------------------------- generating + checking copy
def _parse_variants(raw: str) -> dict:
    out = {}
    for m in re.finditer(r"\[([ABC])\]\s*(.*?)(?=\n\s*\[[ABC]\]|\Z)", raw, re.S):
        block = m.group(2)
        h = re.search(r"HEADLINE:\s*(.+)", block)
        t = re.search(r"TEXT:\s*(.+?)(?=\n\s*CTA:|\Z)", block, re.S)
        c = re.search(r"CTA:\s*(.+)", block)
        if h and t:
            out[m.group(1)] = {"headline": h.group(1).strip(), "body": t.group(1).strip(),
                               "cta": (c.group(1).strip() if c else "")}
    return out


def check_variant(v: dict, facts, platform: str) -> list:
    warns = []
    full = f"{v['headline']} {v['body']} {v['cta']}"
    if not marketing._numbers_ok(full, facts):
        warns.append("عددی دارد که در اطلاعات سالن نیست؛ قیمت و مدت را چک کنید")
    n = A.norm(full)
    for w in RISKY:
        if A.norm(w) in n:
            warns.append(f"عبارت پرریسک «{w}» (ممکن است تبلیغ رد شود یا اعتماد را کم کند)")
    if not CFG["offer"]:
        for w in OFFER_WORDS:
            if A.norm(w) in n:
                warns.append(f"از «{w}» حرف زده ولی در salon.json پیشنهاد واقعی تعریف نشده")
    if len(v["headline"]) > HEADLINE_MAX:
        warns.append(f"تیتر {A.fa(len(v['headline']))} کاراکتر است (بهتر است زیر {A.fa(HEADLINE_MAX)})")
    if len(v["body"]) > PLATFORMS[platform]:
        warns.append(f"متن {A.fa(len(v['body']))} کاراکتر است؛ سقف پیشنهادی {platform}: {A.fa(PLATFORMS[platform])}")
    return warns


def _facts_text(facts) -> str:
    if not facts:
        return "(no specific service: write about the salon in general; mention NO price)"
    return "\n".join(f"- {s['name']} | price {A.fmt_toman(s['price_toman'])} | duration {A.fmt_duration(s['duration_min'])}"
                     for s in facts)


def _write_variants(topic, platform, facts) -> dict:
    system = ("You write ads in natural, warm, professional Persian for a beauty salon. Rules: sound like a "
              "real person, not a loud advert; use ONLY the facts given; never invent discounts, offers, "
              "reviews, statistics, years of experience or medical claims; never say best/guaranteed/cheapest; "
              "headline up to 40 characters; text up to {n} characters. Write exactly three variants in this format:\n"
              "[A]\nHEADLINE: ...\nTEXT: ...\nCTA: ...\n[B]\n...\n[C]\n...\nNo other text.").format(n=PLATFORMS[platform])
    angles = "\n".join(f"[{k}] angle: {v}" for k, v in ANGLES.items())
    offer = f"REAL offer you may mention: {CFG['offer']}" if CFG["offer"] else "There is NO offer: do not mention discounts or free items."
    prompt = (f"Salon: {A.SALON['name']}\nAddress: {A.SALON['address']}\nPhone: {A.SALON['phone']}\n"
              f"Platform: {platform}\nTopic: {topic}\nFacts:\n{_facts_text(facts)}\n{offer}\n"
              f"The CTA should invite the reader to message the salon bot to book.\n{angles}")
    for _ in range(2):                                           # the small model sometimes breaks the format
        got = _parse_variants(marketing._ollama(system, prompt))
        if len(got) >= 2:
            return got
    raise ValueError("model output could not be parsed; try again")


def targeting(topic: str, facts) -> str:
    text = A.norm(topic + " " + " ".join(s["name"] for s in facts))
    interests = []
    for key, vals in INTERESTS.items():
        if A.norm(key) in text:
            interests += [v for v in vals if v not in interests]
    interests = interests or ["سالن زیبایی", "زیبایی و آرایش"]
    place = " ".join(x for x in (CFG["city"], CFG["area"]) if x) or A.SALON["address"]
    return (f"• موقعیت: شعاع {A.fa(CFG['radius_km'])} کیلومتر از {place}\n"
            f"• مخاطب: {CFG['gender']}، سن {A.fa(CFG['age'])}\n"
            f"• علایق پیشنهادی: {'، '.join(interests[:5])}\n"
            f"• زمان پخش: عصرها و شب‌ها (۱۷ تا ۲۲)، و پنجشنبه‌ها؛ برای نوبت هفته‌ی بعد\n"
            f"• هر سه نسخه را همزمان و با بودجه و مخاطب یکسان اجرا کنید و حداقل ۵ تا ۷ روز صبر کنید.")


def create_campaign(topic: str, platform: str = "instagram", goal: str = "booking", budget_toman: int = 0) -> str:
    platform = platform if platform in PLATFORMS else "instagram"
    facts = marketing._service_facts(topic)
    variants = _write_variants(topic, platform, facts)
    c = con()
    cid = c.execute("INSERT INTO ad_campaigns(ts,topic,platform,goal,budget_toman) VALUES(?,?,?,?,?)",
                    (db.ts(datetime.datetime.now()), topic, platform, goal, budget_toman)).lastrowid
    lines = [f"🎯 کمپین {A.fa(cid)} | {topic} | {platform}", ""]
    for label in sorted(variants):
        v = variants[label]
        warns = check_variant(v, facts, platform)
        code = f"ad{cid}{label.lower()}"
        c.execute("INSERT INTO ad_variants(code,campaign_id,label,headline,body,cta,warnings) VALUES(?,?,?,?,?,?,?)",
                  (code, cid, label, v["headline"], v["body"], v["cta"], "; ".join(warns)))
        lines += [f"━━ نسخه {label} ({code})", f"تیتر: {v['headline']}", v["body"], f"دکمه/دعوت: {v['cta']}",
                  f"لینک ردیابی: {bot_link(code)}"]
        lines += [f"⚠️ {w}" for w in warns]
        lines.append("")
    lines += ["هدف‌گیری پیشنهادی:", targeting(topic, facts), "",
              "بعد از اجرا، هزینه‌ی هر نسخه را با /spend ثبت کنید، مثلاً: /spend " + f"ad{cid}a 500000"]
    return "\n".join(lines)


# ---------------------------------------------------------------- tracking + report
def record_lead(conv_id: str, code: str) -> bool:
    """First touch wins: the first ad that brought this customer is the one we credit."""
    c = con()
    if not c.execute("SELECT 1 FROM ad_variants WHERE code=?", (code,)).fetchone():
        return False
    try:
        c.execute("INSERT INTO ad_leads(conv_id,code,ts) VALUES(?,?,?)",
                  (conv_id, code, db.ts(datetime.datetime.now())))
        return True
    except sqlite3.IntegrityError:
        return False


def set_spend(code: str, amount: int) -> bool:
    cur = con().execute("UPDATE ad_variants SET spend_toman=? WHERE code=?", (amount, code))
    return cur.rowcount == 1


def variant_stats(code: str) -> dict:
    c = con()
    leads = c.execute("SELECT conv_id, ts FROM ad_leads WHERE code=?", (code,)).fetchall()
    bookings, revenue = 0, 0
    for l in leads:
        rows = c.execute("SELECT total_toman FROM bookings WHERE conv_id=? AND status='booked' AND created_at>=?",
                         (l["conv_id"], l["ts"])).fetchall()
        bookings += 1 if rows else 0                          # count customers who booked, not visits
        revenue += sum(r["total_toman"] for r in rows)
    return {"leads": len(leads), "bookings": bookings, "revenue": revenue}


def _per(spend, n):
    return A.fa(A.fmt_toman(round(spend / n))) if n and spend else "—"


def report_text() -> str:
    c = con()
    camps = c.execute("SELECT * FROM ad_campaigns ORDER BY id DESC LIMIT 5").fetchall()
    if not camps:
        return "هنوز کمپینی ساخته نشده. با /ad یک کمپین بسازید."
    out = ["📊 گزارش تبلیغات (۵ کمپین آخر)"]
    for camp in camps:
        out += ["", f"کمپین {A.fa(camp['id'])} | {camp['topic']} | {camp['platform']}"]
        rows = []
        for v in c.execute("SELECT * FROM ad_variants WHERE campaign_id=? ORDER BY label", (camp["id"],)):
            s = variant_stats(v["code"])
            rows.append((v, s))
            out.append(f"• {v['label']} ({v['code']}): مراجعه {A.fa(s['leads'])} | نوبت {A.fa(s['bookings'])} | "
                        f"درآمد {A.fa(A.fmt_toman(s['revenue']))} | هزینه {A.fa(A.fmt_toman(v['spend_toman']))} | "
                        f"هزینه هر مراجعه {_per(v['spend_toman'], s['leads'])} | هزینه هر نوبت {_per(v['spend_toman'], s['bookings'])}")
        need = CFG["min_leads_for_winner"]
        if rows and all(s["leads"] >= need for _, s in rows):
            best = max(rows, key=lambda x: (x[1]["bookings"], x[1]["leads"]))
            out.append(f"✅ نسخه‌ی برتر تا اینجا: {best[0]['label']}")
        else:
            out.append(f"ℹ️ هنوز داده کافی نیست؛ برای قضاوت هر نسخه حداقل {A.fa(need)} مراجعه لازم است.")
    out += ["", "توضیح: «نوبت» یعنی مشتری از همین لینک وارد شده و نوبتِ قطعی (پرداخت‌شده) دارد."]
    return "\n".join(out)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "new":
        args = sys.argv[2:]
        plat = args[-1] if args and args[-1] in PLATFORMS else "instagram"
        topic = " ".join(a for a in args if a not in PLATFORMS) or "سالن"
        print(create_campaign(topic, plat))
    elif cmd == "report":
        print(report_text())
    elif cmd == "spend" and len(sys.argv) == 4:
        print("OK" if set_spend(sys.argv[2], int(sys.argv[3])) else "code not found")
    else:
        print(__doc__)
