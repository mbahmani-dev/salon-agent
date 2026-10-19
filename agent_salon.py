"""
Beauty-salon assistant agent (Persian), local model via Ollama. Standard library only.

Files next to this script:
    salon.json      salon data (services, hours, deposit...). EDIT THIS with real data.
    db.py           SQLite storage (bookings, payments, notifications, conversation log)
    payments.py     payment provider interface + FakeGateway (swap in the real gateway later)
    pay_server.py   dev web page for the fake payment + the /callback endpoint
Created at run time:
    salon.db        the database

Run:   py agent_salon.py        (the Ollama server must be running)
"""
import datetime
import inspect
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import db
import payments

BASE = Path(__file__).resolve().parent
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = os.environ.get("SALON_MODEL", "qwen3.5:4b")        # e.g. set SALON_MODEL=gemma4:e2b-it-qat to compare
TEMPERATURE = float(os.environ.get("SALON_TEMP", "0.2"))
MAX_STEPS = 6
NUM_CTX = 4096
KEEP_MESSAGES = 16        # history kept per conversation (small context on a 6GB GPU)
SLOT_STEP_MIN = 30

SALON = json.loads((BASE / "salon.json").read_text(encoding="utf-8"))
WEEKDAYS_FA = ["دوشنبه", "سه‌شنبه", "چهارشنبه", "پنجشنبه", "جمعه", "شنبه", "یکشنبه"]  # Mon=0
JALALI_MONTHS = ["فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور",
                 "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند"]


def now() -> datetime.datetime:
    return datetime.datetime.now()


# ---------------------------------------------------------------- text / date helpers
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def norm(text) -> str:
    """Normalize Persian/Arabic variants and digits so matching is forgiving."""
    t = str(text).translate(_DIGITS)
    t = t.replace("ي", "ی").replace("ك", "ک").replace("\u200c", " ").replace("ۀ", "ه")
    return re.sub(r"\s+", " ", t).strip().lower()


def gregorian_to_jalali(gy, gm, gd):
    g_d_m = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    if gy > 1600:
        jy, gy = 979, gy - 1600
    else:
        jy, gy = 0, gy - 621
    gy2 = gy + 1 if gm > 2 else gy
    days = (365 * gy + (gy2 + 3) // 4 - (gy2 + 99) // 100 + (gy2 + 399) // 400
            - 80 + gd + g_d_m[gm - 1])
    jy += 33 * (days // 12053)
    days %= 12053
    jy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        jy += (days - 1) // 365
        days = (days - 1) % 365
    if days < 186:
        jm, jd = 1 + days // 31, 1 + days % 31
    else:
        jm, jd = 7 + (days - 186) // 30, 1 + (days - 186) % 30
    return jy, jm, jd


def jalali_to_gregorian(jy, jm, jd):
    if jy > 979:
        gy, jy = 1600, jy - 979
    else:
        gy = 621
    days = (365 * jy + (jy // 33) * 8 + ((jy % 33) + 3) // 4 + 78 + jd
            + ((jm - 1) * 31 if jm < 7 else ((jm - 7) * 30 + 186)))
    gy += 400 * (days // 146097)
    days %= 146097
    if days > 36524:
        days -= 1
        gy += 100 * (days // 36524)
        days %= 36524
        if days >= 365:
            days += 1
    gy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        gy += (days - 1) // 365
        days = (days - 1) % 365
    gd = days + 1
    leap = (gy % 4 == 0 and gy % 100 != 0) or gy % 400 == 0
    sal_a = [0, 31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    gm = 0
    while gm < 13 and gd > sal_a[gm]:
        gd -= sal_a[gm]
        gm += 1
    return gy, gm, gd


def jalali_label(d: datetime.date) -> str:
    jy, jm, jd = gregorian_to_jalali(d.year, d.month, d.day)
    return fa(f"{WEEKDAYS_FA[d.weekday()]} {jd} {JALALI_MONTHS[jm - 1]} {jy}")


def parse_date(text) -> datetime.date:
    """Accepts 2026-10-08 (Gregorian) or 1405/07/16 (Jalali)."""
    m = re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", norm(text))
    if not m:
        raise ValueError("date must look like YYYY-MM-DD (or Jalali YYYY/MM/DD)")
    y, mo, d = map(int, m.groups())
    if 1300 <= y <= 1500:
        y, mo, d = jalali_to_gregorian(y, mo, d)
    return datetime.date(y, mo, d)


def parse_time(text) -> datetime.time:
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", norm(text))
    if not m:
        raise ValueError("time must look like HH:MM in 24-hour format")
    return datetime.time(int(m.group(1)), int(m.group(2)))


def fmt_toman(n: int) -> str:
    return f"{n:,} تومان"


# ---------------------------------------------------------------- validation + confirmation state
STATE = {"conv": "cli", "turn": {}, "pending": {}, "messages": None}      # per-conversation turn counter / pending booking
PLACEHOLDER_NAMES = {"نام", "مشتری", "name", "customer", "customer_name", "unknown", "نامشخص", "نام و نام خانوادگی"}


def check_name(name: str):
    """Return (clean_name, problem). problem is a short English reason or None."""
    clean = re.sub(r"\s+", " ", str(name)).strip()
    letters = re.findall(r"[A-Za-z\u0600-\u06FF]", clean)
    if clean.lower() in PLACEHOLDER_NAMES or not clean:
        return clean, "no real name was given"
    if re.search(r"[0-9\u06F0-\u06F9\u0660-\u0669]", clean):
        return clean, "it contains digits"
    if re.search(r"[^A-Za-z\u0600-\u06FF\u200c .\-']", clean):
        return clean, "it contains strange symbols"
    if len(letters) < 2:
        return clean, "it is too short"
    return clean, None


def check_phone(phone: str):
    """Return (phone_09xxxxxxxxx, error). error is a dict or None. Strict on purpose:
    a number with a missing/extra digit must be rejected, not silently 'fixed'."""
    digits = re.sub(r"\D", "", norm(phone))
    if len(digits) < 5:
        return None, {"kind": "none"}
    if digits.startswith("0098") and len(digits) == 14:       # explicit country code only
        digits = "0" + digits[4:]
    elif digits.startswith("98") and len(digits) == 12:
        digits = "0" + digits[2:]
    if len(digits) != 11:
        return None, {"kind": "count", "n": len(digits)}
    if not digits.startswith("09"):
        return None, {"kind": "prefix", "start": digits[:2]}
    return digits, None


FINAL = "FINAL::"     # a tool result starting with this is sent to the customer as-is (the model is skipped)
_FA = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def fa(text) -> str:
    return str(text).translate(_FA)


def pick(key: str, options: list) -> str:
    """Rotate through wordings so the same sentence is not sent twice in a row."""
    k = (STATE["conv"], key)
    i = STATE.setdefault("variant", {}).get(k, 0)
    STATE["variant"][k] = i + 1
    return options[i % len(options)]


def name_error_text(reason: str) -> str:
    if reason == "no real name was given":
        return pick("name_missing", [
            "برای ثبت نوبت لطفاً نام و نام خانوادگی‌تان را بفرستید.",
            "نام و نام خانوادگی شما را هم لازم دارم؛ لطفاً بنویسید."])
    if reason == "not what the customer wrote":
        return pick("name_mismatch", [
            "برای اینکه اسم شما دقیقاً درست ثبت شود، لطفاً نام و نام خانوادگی‌تان را یک‌بار دیگر بنویسید.",
            "می‌خواهم مطمئن شوم اسمتان را درست ثبت می‌کنم؛ لطفاً نام و نام خانوادگی را دوباره بنویسید."])
    return pick("name_bad", [
        "اسمی که نوشتید برایم واضح نبود. لطفاً نام و نام خانوادگی‌تان را دوباره بنویسید.",
        "متأسفانه نتوانستم اسم شما را درست تشخیص بدهم؛ لطفاً نام و نام خانوادگی را یک‌بار دیگر بفرستید.",
        "اسم قبلی خوانا نبود. لطفاً نام و نام خانوادگی را دوباره بنویسید."])


def phone_error_text(err: dict) -> str:
    kind = err["kind"]
    if kind == "none":
        return pick("phone_none", [
            "برای ثبت نوبت شماره موبایل شما را هم لازم دارم؛ لطفاً بفرستید.",
            "لطفاً شماره موبایل‌تان را هم بنویسید تا نوبت را ثبت کنم."])
    if kind == "count":
        n = fa(err["n"])
        return pick("phone_count", [
            f"شماره‌ای که وارد کردید صحیح نیست؛ {n} رقم نوشته‌اید، ولی شماره موبایل باید ۱۱ رقم باشد و با ۰۹ شروع شود. لطفاً دوباره بفرستید.",
            f"این شماره صحیح به نظر نمی‌رسد ({n} رقم). شماره موبایل ۱۱ رقم دارد و با ۰۹ شروع می‌شود؛ لطفاً دوباره بنویسید.",
            f"شماره‌تان درست نیست ({n} رقم وارد شده). لطفاً یک‌بار دیگر شماره موبایل ۱۱ رقمی را بفرستید."])
    start = fa(err["start"])
    return pick("phone_prefix", [
        f"شماره‌ای که وارد کردید صحیح نیست؛ شماره موبایل باید با ۰۹ شروع شود ولی شماره‌ی شما با {start} شروع می‌شود. لطفاً دوباره بفرستید.",
        f"این شماره درست نیست، چون با {start} شروع شده و شماره موبایل با ۰۹ شروع می‌شود. لطفاً دوباره بنویسید."])


def phone_display(phone: str) -> str:
    return fa(f"{phone[:4]} {phone[4:7]} {phone[7:]}")


def fmt_duration(minutes) -> str:
    h, m = divmod(int(minutes), 60)
    parts = ([f"{h} ساعت"] if h else []) + ([f"{m} دقیقه"] if m else [])
    return fa(" و ".join(parts) or "0 دقیقه")


def confirm_text(names, when, minutes, total, deposit, name, phone) -> str:
    intro = pick("confirm_intro", [
        "لطفاً اطلاعات نوبت را یک‌بار بررسی کنید (هنوز ثبت نشده است):",
        "قبل از ثبت نهایی، اطلاعات نوبت را ببینید (هنوز ثبت نشده):",
        "اطلاعات نوبت به این صورت است و هنوز ثبت نشده:"])
    if deposit > 0:
        outro = pick("confirm_outro_pay", [
            "اگر همه‌چیز درست است بنویسید «بله» تا لینک پرداخت بیعانه را برایتان بفرستم؛ در غیر این صورت بگویید کدام مورد اصلاح شود.",
            "درست است؟ با «بله» لینک پرداخت بیعانه را می‌فرستم، یا بگویید چه چیزی را عوض کنم.",
            "اگر اطلاعات درست است «بله» بفرستید تا لینک پرداخت بیعانه آماده شود، وگرنه مورد اشتباه را بنویسید."])
    else:
        outro = pick("confirm_outro", [
            "اگر همه‌چیز درست است بنویسید «بله» تا نوبت را ثبت کنم؛ در غیر این صورت بگویید کدام مورد اصلاح شود.",
            "درست است؟ با «بله» ثبت می‌کنم، یا بگویید چه چیزی را عوض کنم.",
            "اگر اطلاعات درست است «بله» بفرستید تا ثبت شود، وگرنه مورد اشتباه را بنویسید."])
    lines = [intro, f"• خدمات: {names}", f"• زمان: {fa(when)}", f"• مدت: {fmt_duration(minutes)}",
             f"• هزینه کل: {fa(fmt_toman(total))}"]
    if deposit > 0:
        lines.append(f"• بیعانه‌ی لازم برای قطعی شدن نوبت: {fa(fmt_toman(deposit))}")
    lines += [f"• نام: {name}", f"• شماره موبایل: {phone_display(phone)}", outro]
    return "\n".join(lines)


def booked_text(names, when, minutes, total, paid=None, bank_ref=None) -> str:
    intro = pick("booked_intro", ["نوبت شما ثبت شد ✅", "نوبتتان با موفقیت ثبت شد ✅"])
    lines = [intro, f"• خدمات: {names}", f"• زمان: {fa(when)}", f"• مدت: {fmt_duration(minutes)}",
             f"• هزینه کل: {fa(fmt_toman(total))}"]
    if paid:
        lines.append(f"• بیعانه‌ی پرداخت‌شده: {fa(fmt_toman(paid))}")
        if bank_ref:
            lines.append(f"• کد پیگیری: {bank_ref}")
        if total - paid > 0:
            lines.append(f"• باقی‌مانده برای پرداخت در سالن: {fa(fmt_toman(total - paid))}")
    lines += [fa(SALON["cancellation_policy"]), "منتظر شما هستیم."]
    return "\n".join(lines)


def payment_request_text(booking, payment) -> str:
    until = datetime.datetime.fromisoformat(booking["hold_until"]).strftime("%H:%M")
    return (f"نوبت شما تا ساعت {fa(until)} برایتان نگه داشته شده است.\n"
            f"برای قطعی شدن، بیعانه‌ی {fa(fmt_toman(payment['amount_toman']))} را از لینک زیر پرداخت کنید:\n"
            f"{payment['pay_url']}\n"
            f"بعد از پرداخت، تأیید نوبت را همین‌جا برایتان می‌فرستم. اگر پرداخت انجام نشود، نوبت آزاد می‌شود.")


# ---------------------------------------------------------------- facts must come from the customer's own words
def customer_text() -> str:
    """Everything the customer typed in this conversation (None-safe for unit tests)."""
    msgs = STATE.get("messages")
    if msgs is None:
        return ""
    return "\n".join(m.get("content", "") for m in msgs if m.get("role") == "user")


def customer_phone_runs():
    """Phone-like digit strings in the customer's messages, oldest first.
    Returns None when the conversation is not available (unit tests)."""
    if STATE.get("messages") is None:
        return None
    text = norm(customer_text())
    text = re.sub(r"(\+?98)[ -]?(9\d{2})[ -](\d{3})[ -](\d{4})", r"0\2\3\4", text)   # +98 912 345 6789
    text = re.sub(r"(09\d{2})[ -](\d{3})[ -](\d{4})", r"\1\2\3", text)               # 0912 345 6789
    return [r for r in re.findall(r"\d+", text) if len(r) >= 7 and r[0] in "09"]


def name_in_customer_words(name: str) -> bool:
    """The model may not 'fix' a name: every word of it must appear in what the customer wrote."""
    if STATE.get("messages") is None:
        return True
    text = norm(customer_text())
    tokens = [t for t in norm(name).split() if len(t) >= 2]
    return bool(tokens) and all(t in text for t in tokens)


def slot_ranges(slots: list) -> str:
    """['10:00','10:30','11:00','15:00'] -> '10:00 to 11:00, 15:00 to 15:00' (consecutive start times merged)."""
    out, first, prev = [], None, None
    for s in slots:
        t = datetime.datetime.strptime(s, "%H:%M")
        if prev is not None and t - prev == datetime.timedelta(minutes=SLOT_STEP_MIN):
            prev = t
            continue
        if first is not None:
            out.append((first, prev))
        first = prev = t
    if first is not None:
        out.append((first, prev))
    return fa(", ".join(f"{a:%H:%M}" if a == b else f"{a:%H:%M} تا {b:%H:%M}" for a, b in out))


AFFIRM = {"بله", "بلی", "آره", "اره", "اوکی", "اوکیه", "ok", "okay", "yes", "درسته", "درست", "تایید", "تأیید",
          "باشه", "موافقم", "صحیحه", "صحیح", "خوبه", "عالیه", "همینه", "ثبت", "ثبتش", "بفرست", "لینک"}
NEGATE = {"نه", "نخیر", "نیست", "اشتباه", "غلط", "عوض", "تغییر", "فقط", "ولی", "اما", "لغو", "کنسل", "نمیخوام"}


def customer_said_yes() -> bool:
    """A booking is only finalized when the customer's LATEST message is a clear yes
    (no 'but', 'no', 'change'...). The model's own tool call is not enough."""
    msgs = STATE.get("messages")
    if msgs is None:                                            # unit tests without a conversation
        return True
    last = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"), "")
    text = norm(last)
    words = set(re.findall(r"[^\W\d_]+", text))
    if words & NEGATE or "نمی خوام" in text:
        return False
    return bool(words & AFFIRM)


def fix_afternoon(t: datetime.time, d: datetime.date) -> datetime.time:
    """The salon never opens before 10, so '01:00'..'09:59' really means the afternoon (+12h)."""
    hours = day_hours(d)
    if hours and t.hour < 12 and t < hours[0]:
        shifted = t.replace(hour=t.hour + 12)
        if hours[0] <= shifted < hours[1]:
            return shifted
    return t


# ---------------------------------------------------------------- catalog search
STOPWORDS = {"قیمت", "چنده", "چقدره", "چقدر", "هزینه", "برای", "میخوام", "خوام", "می", "من",
             "یه", "یک", "لطفا", "سلام", "نوبت", "تومان", "چند", "است", "هست"}


def _haystack(svc) -> str:
    return norm(svc["name"] + " " + " ".join(svc["keywords"]))


GENERIC = {"مو", "ناخن", "صورت", "دست", "پا", "خدمات"}   # words shared by many services: weak signal


def _word_hit(token: str, hay: str) -> bool:
    words = hay.split()
    if len(token) <= 2:                       # very short tokens: whole-word match only
        return token in words
    return token in hay


def find_services(query: str):
    """All services the query mentions (so 'کوتاهی و مش' returns both).
    Strong words (رنگ, مش, کراتین...) decide; generic words (مو, ناخن) only matter
    when no strong word matched anything (e.g. the query is just 'مو')."""
    q = norm(query)
    tokens = [t for t in q.split() if len(t) >= 2 and t not in STOPWORDS]
    strong = [t for t in tokens if t not in GENERIC]
    weak = [t for t in tokens if t in GENERIC]

    def collect(words):
        found = []
        for svc in SALON["services"]:
            hay = _haystack(svc)
            score = sum(1 for t in words if _word_hit(t, hay))
            if norm(svc["name"]) in q:
                score += 2
            if score:
                found.append((score, svc))
        found.sort(key=lambda x: -x[0])
        return [svc for _, svc in found]

    hits = collect(strong)
    return hits if hits else collect(weak)


def resolve_services(value):
    """Turn the model's 'services' argument (list or text) into exactly-matched services."""
    if isinstance(value, str):
        value = [p for p in re.split(r"\s*(?:[+،,]|\sو\s)\s*", value) if p.strip()]
    if not value:
        raise ValueError("no service given; ask the customer which service they want")
    chosen = []
    for item in value:
        hits = find_services(str(item))
        if len(hits) != 1:
            names = ", ".join(s["name"] for s in hits) or "none"
            raise ValueError(f"service '{item}' is not exact (matches: {names}); ask the customer to choose")
        if hits[0] not in chosen:
            chosen.append(hits[0])
    return chosen


def service_view(svc) -> dict:
    return {"name": svc["name"], "price_text": fa(fmt_toman(svc["price_toman"])),
            "duration_text": fmt_duration(svc["duration_min"])}


def day_hours(d: datetime.date):
    h = SALON["hours"].get(WEEKDAYS_FA[d.weekday()])
    if not h:
        return None
    return parse_time(h[0]), parse_time(h[1])


def is_free(start: datetime.datetime, minutes: int) -> bool:
    return db.count_overlapping(start, minutes, now()) < SALON["capacity"]


def slot_problem(start: datetime.datetime, minutes: int):
    """Return None if the slot is bookable, else a short reason string."""
    hours = day_hours(start.date())
    if hours is None:
        return f"The salon is closed on {WEEKDAYS_FA[start.weekday()]}."
    open_t, close_t = hours
    end = start + datetime.timedelta(minutes=minutes)
    if start.time() < open_t or end.time() > close_t or end.date() != start.date():
        return (f"Outside working hours ({open_t:%H:%M}-{close_t:%H:%M}); "
                f"the service takes {minutes} minutes.")
    if start < now() + datetime.timedelta(minutes=30):
        return "That time is in the past or too soon."
    if not is_free(start, minutes):
        return "That time is already fully booked."
    return None


# ---------------------------------------------------------------- tools
def search_services(query: str) -> str:
    """Find salon services and their price and duration."""
    hits = find_services(query)
    if not hits:
        return "NO_MATCH: no such service. Offer the customer the list of services or hand off."
    out = {"services": [service_view(s) for s in hits]}
    if len(hits) >= 2:
        out["total_if_all_booked_together"] = {
            "price_text": fa(fmt_toman(sum(x["price_toman"] for x in hits))),
            "duration_text": fmt_duration(sum(x["duration_min"] for x in hits))}
    return json.dumps(out, ensure_ascii=False)


PERIODS = {"morning": (0, 12), "afternoon": (12, 17), "evening": (17, 24)}
PERIOD_ALIASES = {"صبح": "morning", "ظهر": "afternoon", "بعدازظهر": "afternoon", "بعد از ظهر": "afternoon",
                  "عصر": "evening", "شب": "evening"}


def get_free_slots(date: str, services, period: str = "") -> str:
    """List free start times for one visit (one or several services) on a date, optionally for a part of the day."""
    d = parse_date(date)
    chosen = resolve_services(services)
    minutes = sum(s["duration_min"] for s in chosen)
    hours = day_hours(d)
    label = jalali_label(d)
    if hours is None:
        return f"CLOSED on {label}."
    slots, cur = [], datetime.datetime.combine(d, hours[0])
    while cur.time() <= hours[1]:
        if slot_problem(cur, minutes) is None:
            slots.append(cur.strftime("%H:%M"))
        cur += datetime.timedelta(minutes=SLOT_STEP_MIN)
    head = f"{label} (visit length {fmt_duration(minutes)})"
    if not slots:
        return f"{head}: no free time that day."
    key = PERIOD_ALIASES.get(norm(period), norm(period))
    if key in PERIODS:
        lo, hi = PERIODS[key]
        part = [t for t in slots if lo <= int(t[:2]) < hi]
        if part:
            return f"{head}: free START times in the {key} (every {SLOT_STEP_MIN} min): {slot_ranges(part)}"
        return (f"{head}: nothing free in the {key}. Other free START times that day: {slot_ranges(slots)}")
    return f"{head}: free START times (every {SLOT_STEP_MIN} min): {slot_ranges(slots)}"


# ---------------------------------------------------------------- booking + payment flow
OUTBOX = []          # messages pushed to customers outside a normal reply (e.g. "payment received")


def push_to_customer(conv_id: str, text: str) -> None:
    """CLI: print it. The Bale adapter will replace this with a real 'send message' call."""
    OUTBOX.append((conv_id, text))
    print(f"\n[PUSH -> {conv_id}]\n{text}\n")


def deposit_amount(total: int) -> int:
    pct = SALON.get("deposit_percent", 0)
    if pct <= 0:
        return 0
    return min(total, math.ceil(total * pct / 100 / 1000) * 1000)       # round up to 1,000 toman


def create_payment_for_booking(booking):
    """Ask the payment provider for a new payment link for this booking and store it."""
    provider = payments.get_provider()
    req = provider.create(booking_id=booking["id"], amount_toman=booking["deposit_toman"],
                          description=f"{SALON['name']} - {booking['services']}",
                          callback_url=payments.CALLBACK_BASE_URL)
    db.add_payment(booking_id=booking["id"], provider=provider.name, provider_ref=req.provider_ref,
                   amount=booking["deposit_toman"], pay_url=req.pay_url, now=now())
    return db.payment_for_booking(booking["id"])


def booking_when(b) -> str:
    start = datetime.datetime.fromisoformat(b["start"])
    return f"{jalali_label(start.date())} ساعت {start:%H:%M}"


def booking_card(b) -> str:
    pay = db.payment_for_booking(b["id"])
    paid = pay["amount_toman"] if pay and pay["status"] == "paid" else None
    return booked_text(b["services"], booking_when(b), b["duration_min"], b["total_toman"],
                       paid=paid, bank_ref=pay["bank_ref"] if paid else None)


def finalize_booking(conv, name, phone, names, start, minutes, total, deposit) -> str:
    """The customer confirmed the card. Hold the slot and (if a deposit is required) request payment."""
    existing = db.find_active_pending(conv, start, now())
    if existing:                                          # same request again: do not create a second booking
        pay = db.payment_for_booking(existing["id"])
        if not pay or pay["status"] != "created":
            pay = create_payment_for_booking(existing)
        return payment_request_text(existing, pay)
    status = "pending_payment" if deposit > 0 else "booked"
    bid = db.create_booking(conv_id=conv, name=name, phone=phone, services=names, start=start,
                            minutes=minutes, total=total, deposit=deposit, capacity=SALON["capacity"],
                            status=status, hold_minutes=SALON.get("hold_minutes", 15), now=now())
    if bid is None:                                       # someone took the last place a moment ago
        return "متأسفانه همین الان این زمان توسط مشتری دیگری رزرو شد. می‌خواهید زمان‌های خالی دیگر را بررسی کنم؟"
    booking = db.get_booking(bid)
    if status == "booked":
        notify_owner(f"نوبت جدید: {name} ({phone}) - {names} - {booking_when(booking)}")
        return booking_card(booking)
    try:
        pay = create_payment_for_booking(booking)
    except Exception as e:                                # gateway down: do not keep the slot blocked
        db.set_booking_status(bid, "cancelled")
        notify_owner(f"خطا در ساخت لینک پرداخت (نوبت {bid}): {e}")
        return "متأسفانه الان امکان ساخت لینک پرداخت نیست. همکاران ما به‌زودی با شما تماس می‌گیرند."
    notify_owner(f"نوبت در انتظار پرداخت: {name} ({phone}) - {names} - {booking_when(booking)}")
    return payment_request_text(booking, pay)


def book_appointment(customer_name: str, phone: str, services, date: str, time: str) -> str:
    """Book ONE visit (several services allowed). Confirmation card first, then payment link."""
    parts = []
    name_clean, name_problem = check_name(customer_name)
    if not name_problem and not name_in_customer_words(name_clean):
        name_problem = "not what the customer wrote"           # the model changed the name: do not trust it
    if name_problem:
        parts.append(name_error_text(name_problem))
    runs = customer_phone_runs()
    if runs is None:                                            # no conversation available (unit tests)
        phone_n, phone_err = check_phone(phone)
    elif runs:                                                  # the customer's own latest number wins over the model's copy
        phone_n, phone_err = check_phone(runs[-1])
    else:
        phone_n, phone_err = None, {"kind": "none"}
    if phone_err:
        parts.append(phone_error_text(phone_err))
    if parts:
        return FINAL + "\n".join(parts)

    chosen = resolve_services(services)
    minutes = sum(s["duration_min"] for s in chosen)
    total = sum(s["price_toman"] for s in chosen)
    names = " + ".join(s["name"] for s in chosen)
    d = parse_date(date)
    start = datetime.datetime.combine(d, fix_afternoon(parse_time(time), d))
    problem = slot_problem(start, minutes)
    if problem:
        return f"ERROR: {problem} Use get_free_slots to offer other times."
    deposit = deposit_amount(total)

    # Step 1: the customer sees every detail and says yes (in a later turn). Step 2: hold slot + payment.
    conv = STATE["conv"]
    turn = STATE["turn"].get(conv, 0)
    key = (names, start.isoformat(timespec="minutes"), phone_n, name_clean)
    pending = STATE["pending"].get(conv)
    confirmed = bool(pending and pending["key"] == key and pending["turn"] < turn and customer_said_yes())
    if not confirmed:
        STATE["pending"][conv] = {"key": key, "turn": turn}
        return FINAL + confirm_text(names, f"{jalali_label(start.date())} ساعت {start:%H:%M}",
                                    minutes, total, deposit, name_clean, phone_n)
    STATE["pending"].pop(conv, None)
    return FINAL + finalize_booking(conv, name_clean, phone_n, names, start, minutes, total, deposit)


def my_booking_status() -> str:
    """Status of this customer's latest booking in this chat (paid? confirmed? send the link again)."""
    conv = STATE["conv"]
    b = db.latest_booking_for_conv(conv)
    if not b:
        return "NO_BOOKING: this customer has no booking in this chat yet."
    if b["status"] == "booked":
        return FINAL + booking_card(b)
    if b["status"] == "pending_payment" and b["hold_until"] > db.ts(now()):
        pay = db.payment_for_booking(b["id"])
        if not pay or pay["status"] != "created":         # earlier attempt failed: make a fresh link
            pay = create_payment_for_booking(b)
        return FINAL + payment_request_text(b, pay)
    return (f"NOT_ACTIVE: the customer's last booking is '{b['status']}' (the payment time ran out or it was "
            f"cancelled). Tell the customer and offer to book again with get_free_slots.")


def on_payment_callback(provider_ref: str, params: dict) -> dict:
    """Called by the web endpoint when the customer returns from the payment page.
    Verifies with the provider, checks the amount, confirms the booking, tells the customer."""
    pay = db.get_payment_by_ref(provider_ref or "")
    if not pay:
        return {"ok": False, "text": "پرداخت پیدا نشد."}
    booking = db.get_booking(pay["booking_id"])
    conv = booking["conv_id"]
    if pay["status"] == "paid":                                   # callback arrived twice
        return {"ok": True, "text": "این پرداخت قبلاً ثبت شده است."}

    try:
        result = payments.get_provider().verify(provider_ref=provider_ref,
                                                expected_amount_toman=pay["amount_toman"], callback_params=params)
    except Exception as e:
        notify_owner(f"خطا در تأیید پرداخت {provider_ref}: {e}")
        return {"ok": False, "text": "تأیید پرداخت با خطا روبه‌رو شد؛ همکاران ما پیگیری می‌کنند."}

    if not result.paid:
        db.set_payment_status(pay["id"], "failed")
        text = ("پرداخت انجام نشد یا لغو شد. اگر می‌خواهید دوباره تلاش کنید، همین‌جا بنویسید "
                "تا لینک جدید برایتان بفرستم (نوبت تا پایان مهلت نگه داشته می‌شود).")
        push_to_customer(conv, text)
        return {"ok": False, "text": "پرداخت انجام نشد."}

    if result.amount_toman is not None and result.amount_toman != pay["amount_toman"]:
        db.set_payment_status(pay["id"], "failed")
        notify_owner(f"مبلغ پرداخت‌شده ({result.amount_toman}) با مبلغ ثبت‌شده ({pay['amount_toman']}) نمی‌خواند: "
                     f"{provider_ref} - بررسی دستی لازم است")
        return {"ok": False, "text": "مبلغ پرداخت تأیید نشد؛ همکاران ما پیگیری می‌کنند."}

    if not db.claim_payment_paid(pay["id"], result.bank_ref, now()):
        return {"ok": True, "text": "این پرداخت قبلاً ثبت شده است."}

    b = db.get_booking(booking["id"])
    ok = False
    if b["status"] == "pending_payment":
        db.set_booking_status(b["id"], "booked")
        ok = True
    elif b["status"] == "expired":                                # paid after the hold ran out
        start = datetime.datetime.fromisoformat(b["start"])
        if db.count_overlapping(start, b["duration_min"], now()) < SALON["capacity"]:
            db.set_booking_status(b["id"], "booked")
            ok = True
    if ok:
        notify_owner(f"نوبت قطعی شد (بیعانه پرداخت شد): {b['customer_name']} ({b['phone']}) - "
                     f"{b['services']} - {booking_when(b)}")
        text = booked_text(b["services"], booking_when(b), b["duration_min"], b["total_toman"],
                           paid=pay["amount_toman"], bank_ref=result.bank_ref)
    else:
        db.set_booking_status(b["id"], "cancelled")
        notify_owner(f"بازگشت وجه لازم است: پرداخت {provider_ref} ({pay['amount_toman']} تومان) انجام شد ولی "
                     f"زمان دیگر خالی نیست - {b['customer_name']} ({b['phone']}) - {booking_when(b)}")
        text = ("پرداخت شما دریافت شد، اما متأسفانه آن زمان دیگر خالی نیست. همکاران ما برای تعیین زمان جدید "
                "یا بازگشت وجه به‌زودی با شما تماس می‌گیرند.")
    push_to_customer(conv, text)
    return {"ok": ok, "text": "پرداخت انجام شد." if ok else "پرداخت دریافت شد، نیاز به پیگیری."}


def handoff_to_human(reason: str, summary: str) -> str:
    """Escalate to the salon staff."""
    notify_owner(f"نیاز به پیگیری انسانی: {reason} | {summary}")
    return "DONE: staff will contact the customer. Tell the customer that."


def notify_owner(text: str) -> None:
    """For now: print + save in the database. Later this will send a Bale message to the owner."""
    db.add_notification(text, now())
    print(f"\n[OWNER] {text}\n")


TOOLS = {f.__name__: f for f in (search_services, get_free_slots, book_appointment,
                                 my_booking_status, handoff_to_human)}

_S = {"type": "string"}
_SERVICES = {"type": "array", "items": {"type": "string"},
             "description": "exact service names for ONE visit, one entry per service, e.g. [\"کوتاهی مو\", \"مش و هایلایت\"]"}


def _spec(name, desc, props, optional=()):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": [k for k in props if k not in optional]}}}


TOOL_SPECS = [
    _spec("search_services",
          "Find salon services with exact price and duration. If the customer mentions several "
          "services, put all of them in the query; every matching service is returned.",
          {"query": {**_S, "description": "service names or keywords, Persian"}}),
    _spec("get_free_slots", "List free start times for one visit (one or more services) on a date.",
          {"date": {**_S, "description": "YYYY-MM-DD (Gregorian)"}, "services": _SERVICES,
           "period": {**_S, "description": "morning | afternoon | evening - only if the customer named a part of the day"}},
          optional=("period",)),
    _spec("book_appointment",
          "Book ONE visit (one or more services). Only call when you have name, phone, services, date and time.",
          {"customer_name": _S, "phone": {**_S, "description": "Iranian mobile, 09xxxxxxxxx"},
           "services": _SERVICES, "date": {**_S, "description": "YYYY-MM-DD (Gregorian)"},
           "time": {**_S, "description": "HH:MM, 24-hour"}}),
    _spec("my_booking_status",
          "Status of this customer's latest booking in this chat: payment received? confirmed? Also re-sends the payment link.",
          {}),
    _spec("handoff_to_human", "Pass the conversation to salon staff.",
          {"reason": _S, "summary": {**_S, "description": "short summary in Persian"}}),
]


# ---------------------------------------------------------------- prompt / LLM loop
def hours_text() -> str:
    parts = []
    for day in ["شنبه", "یکشنبه", "دوشنبه", "سه‌شنبه", "چهارشنبه", "پنجشنبه", "جمعه"]:
        h = SALON["hours"].get(day)
        parts.append(f"{day}: {h[0]}-{h[1]}" if h else f"{day}: تعطیل")
    return "; ".join(parts)


def upcoming_days_text() -> str:
    t = now().date()
    names = {0: "امروز", 1: "فردا", 2: "پس‌فردا"}
    lines = []
    for i in range(8):
        d = t + datetime.timedelta(days=i)
        lines.append(f"- {names.get(i, '')} {jalali_label(d)} = {d:%Y-%m-%d}".replace("-  ", "- "))
    return "\n".join(lines)


def system_prompt() -> str:
    t = now()
    return f"""You are the Persian-language assistant of the beauty salon "{SALON['name']}".
Today is {jalali_label(t.date())} ({t:%Y-%m-%d}), current time {t:%H:%M}.
Upcoming days (copy the Persian label and the ISO date from here, never work out weekdays yourself):
{upcoming_days_text()}
Working hours: {hours_text()}. Address: {SALON['address']}. Phone: {SALON['phone']}.
Cancellation policy: {SALON['cancellation_policy']}

Rules:
1. Reply in Persian, polite and short (1-3 sentences). Use "شما". If the customer writes English, reply in English.
2. Prices and durations come ONLY from tools. Quote price_text exactly, in full digits (e.g. 2,500,000 تومان); never convert it to «هزار» or «میلیون» and never add words like "حدود". Never invent services.
2b. If the customer mentions several services, your answer MUST cover ALL of them (price and duration of each). Never silently drop one. If search_services returned fewer services than the customer asked about, say which one you could not find.
2d. Quote duration_text exactly as given; never do time arithmetic yourself.
2c. Several services in one visit are booked together as ONE appointment (list every service in "services"); the visit length is the sum of their durations.
3. To book you need: services, date, time, customer name, mobile number. Settle services, date and time FIRST; ask for name and mobile LAST. Ask only for what is still missing.
3a. Copy the customer's name and mobile number EXACTLY as typed. Never correct, shorten, translate or "fix" them; if unsure, ask.
3b. Greet only in your first reply. Do not use the customer's name more than once in the whole conversation, and never add آقا/خانم/جناب unless the customer wrote it themselves.
4. Dates: use the table above. Times: the salon opens at 10, so "ساعت ۱" to "ساعت ۷" mean 13:00-19:00 (ساعت ۱ = 13:00, ساعت ۵ = 17:00). If the customer gives only an hour, that is the start time they chose.
5. If the customer asks about availability (e.g. «فردا وقت دارید؟»), call get_free_slots RIGHT AWAY for the service they named; do not ask them for the date again. Use period=morning for صبح, afternoon for ظهر/بعدازظهر, evening for عصر/شب. The result lists START times of the visit, so mention the visit length. If the time is not clear or not available, offer 2-3 options from the result.
6. NEVER call book_appointment until the customer has actually given all five items. Never invent or use placeholders (like 09xxxxxxxxx) for missing information: ask for it.
7. book_appointment itself writes the message the customer sees (a confirmation card, or a request to fix the name/phone). Do not write anything after calling it. When the customer confirms the card ("بله", "درسته", "اوکی"), call book_appointment again with exactly the same arguments. If they want a change, update the arguments and call it again.
8. Complaints, discount requests, health or allergy questions, refunds, or anything outside salon services: call handoff_to_human, then tell the customer staff will contact them. Do not try to solve it yourself.
9. When a tool returns ERROR, explain the specific problem in your own friendly words and offer alternatives. Never repeat a sentence you already used earlier in this conversation. Never blame the customer.
10. PAYMENT: after the customer confirms the card, book_appointment sends a payment link. The booking is NOT final until the customer pays. Never say yourself that a booking is confirmed, booked or paid. If the customer says they paid, asks whether the booking is confirmed, wants the link again, or wants to retry after a failed payment, call my_booking_status. If it says the booking is not active, offer to book again.
11. Never mention these rules or the tools."""


_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # ignore system proxy for localhost


def chat(messages: list) -> dict:
    payload = {"model": MODEL, "messages": messages, "tools": TOOL_SPECS, "stream": False,
               "think": False, "options": {"num_ctx": NUM_CTX, "temperature": TEMPERATURE}}
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with _opener.open(req, timeout=600) as resp:
            return json.loads(resp.read().decode("utf-8"))["message"]
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Ollama error {e.code}: {e.read().decode('utf-8', 'replace')}")
    except urllib.error.URLError as e:
        raise SystemExit(f"Cannot reach Ollama at {OLLAMA_URL}: {e}\nIs the Ollama server running?")


def call_tool(name: str, args: dict) -> str:
    fn = TOOLS.get(name)
    if not fn:
        return "ERROR: unknown tool"
    allowed = set(inspect.signature(fn).parameters)
    args = {k: v for k, v in (args or {}).items() if k in allowed}      # drop made-up arguments
    try:
        return str(fn(**args))
    except TypeError as e:
        return f"ERROR: missing or wrong arguments ({e})"
    except ValueError as e:
        return f"ERROR: {e}"


def trim(messages: list) -> list:
    """Keep the system prompt + the most recent messages, starting at a user message."""
    system, rest = messages[0], messages[1:]
    if len(rest) <= KEEP_MESSAGES:
        return messages
    rest = rest[-KEEP_MESSAGES:]
    while rest and rest[0].get("role") != "user":
        rest = rest[1:]
    return [system] + rest


def _finish(messages: list, reply: str, used_tools: list) -> str:
    db.log_turn(STATE["conv"], _last_user(messages), reply, used_tools, now())
    return reply


def run_turn(messages: list, conv_id: str = "cli") -> str:
    """messages: one customer's conversation. Returns the bot reply."""
    STATE["conv"] = conv_id
    STATE["messages"] = messages
    STATE["turn"][conv_id] = STATE["turn"].get(conv_id, 0) + 1
    db.expire_holds(now())                                              # free slots whose payment time ran out
    messages[0] = {"role": "system", "content": system_prompt()}        # refresh date/time every turn
    used_tools = []
    for _ in range(MAX_STEPS):
        msg = chat(trim(messages))
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            reply = msg.get("content", "").strip()
            if any(m.get("role") == "assistant" and m.get("content") for m in messages[:-1]):   # not the first reply
                reply = re.sub(r"^\s*(سلام|درود)[\s!،,.؟]*", "", reply).strip() or reply
            # Safety net: the model must never claim "booked/paid" while the booking is not really final.
            claims = any(w in reply for w in ("ثبت شد", "قطعی شد", "پرداخت شد", "تأیید شد")) and "نوبت" in reply
            if claims and "book_appointment" not in used_tools and "my_booking_status" not in used_tools:
                last = db.latest_booking_for_conv(conv_id)
                if STATE["pending"].get(conv_id):
                    reply = "نوبت هنوز ثبت نشده است؛ لطفاً با «بله» تأیید کنید تا ادامه دهم."
                elif last and last["status"] == "pending_payment":
                    reply = "نوبت شما تا پرداخت بیعانه قطعی نمی‌شود. اگر پرداخت کرده‌اید، بنویسید «پرداخت کردم» تا وضعیت را بررسی کنم."
                if reply != msg.get("content", "").strip():
                    messages[-1] = {"role": "assistant", "content": reply}
            return _finish(messages, reply, used_tools)
        for call in calls:
            name = call["function"]["name"]
            args = call["function"].get("arguments") or {}
            result = call_tool(name, args)
            used_tools.append(name)
            print(f"  [tool] {name}({args}) -> {result[:100]}")
            if result.startswith(FINAL):                      # ready-made reply: send it, skip the model
                text = result[len(FINAL):]
                messages.append({"role": "tool", "tool_name": name, "content": "Message shown to the customer."})
                messages.append({"role": "assistant", "content": text})
                return _finish(messages, text, used_tools)
            messages.append({"role": "tool", "tool_name": name, "content": result})
    handoff_to_human("agent step limit reached", "The assistant could not finish this conversation.")
    return "ببخشید، همکارم به‌زودی با شما تماس می‌گیرد."


def _last_user(messages: list) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def main() -> None:
    messages = [{"role": "system", "content": system_prompt()}]
    server = None
    if os.environ.get("PAYMENT_PROVIDER", "fake").lower() == "fake":
        try:
            import pay_server
            server = pay_server.start(sys.modules[__name__])
            print(f"[dev] fake payment page: {payments.CALLBACK_BASE_URL}  (or type /pay or /fail here)")
        except OSError as e:
            print(f"[dev] could not start the fake payment web page ({e}); use /pay or /fail instead")
    print(f"Salon agent ready ({SALON['name']}). Type a message, Ctrl+C to quit.\n"
          f"Dev commands: /pay (simulate successful payment)  /fail (failed payment)  /status (database)\n")
    while True:
        user = input("you> ").strip()
        while OUTBOX:                                           # tell the model what was pushed to the customer
            c, t = OUTBOX.pop(0)
            if c == "cli":
                messages.append({"role": "assistant", "content": t})
        if not user:
            continue
        if user in ("/pay", "/fail"):
            pay = db.latest_open_payment_for_conv("cli")
            if not pay:
                print("[dev] no open payment for this chat\n")
                continue
            on_payment_callback(pay["provider_ref"], {"status": "OK" if user == "/pay" else "NOK"})
            continue
        if user == "/status":
            for r in db.get_con().execute("SELECT id,status,customer_name,services,start,deposit_toman,hold_until "
                                          "FROM bookings ORDER BY id DESC LIMIT 10"):
                print(dict(r))
            print()
            continue
        messages.append({"role": "user", "content": user})
        print("bot>", run_turn(messages), "\n")


if __name__ == "__main__":
    main()