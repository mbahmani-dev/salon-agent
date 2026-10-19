"""
Customer memory for the salon agent (standard library only).

Four layers, each stored and used differently:

  WORKING     the current conversation (messages) + the last few turns restored from the log if the
              customer wrote within the last day. Lives in RAM / conversation_log.
  EPISODIC    things that happened: confirmed bookings, hand-offs to staff, automatic summaries.
              Table `episodes`. Written by code at the moment the event happens.
  SEMANTIC    what we know about the customer: name and mobile on file, usual services, preferred part
              of the day (derived from bookings by code), plus short notes the customer asked us to
              remember (stated). Tables `customers`, `facts`.
  PROCEDURAL  HOW to handle situations (bridal makeup, complaints, cancellations). Plain text files in
              the skills/ folder that the salon owner can edit; the matching ones are added to the
              prompt for that turn only.

  DISTILLATION  conversation turns older than N days (default 30) are condensed: facts are recomputed
              from bookings (exact, no AI), an AI summary of the old chat is stored as one short
              unverified episode, and the raw turns are deleted. Run with distill.py.

Safety rules built in
  * Memory belongs to ONE chat identity (conv_id). Nothing is ever looked up by a phone number typed
    in chat, so nobody can read another person's data by typing their number.
  * Exact facts (bookings, counts, dates) are computed from the database, never written by the model.
  * Health, religion, politics, money, family trouble and similar topics are never stored as notes or
    summaries (a keyword filter: crude, a backstop, not a guarantee).
  * forget() deletes the profile, facts, episodes and raw turns. Bookings/payments stay (business records).
"""
import datetime
import hashlib
import re
from collections import Counter
from pathlib import Path

import db

BASE = Path(__file__).resolve().parent
SKILLS_DIR = BASE / "skills"
CONTEXT_CHARS = 700          # budget for the memory block in the prompt (small 4096-token context!)
SKILLS_CHARS = 800
MAX_NOTES = 8
RESTORE_HOURS = 24
RESTORE_TURNS = 4

_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def _norm(text) -> str:
    t = str(text).translate(_DIGITS).replace("ي", "ی").replace("ك", "ک").replace("\u200c", " ")
    return re.sub(r"\s+", " ", t).strip().lower()


# Topics we never keep in notes or summaries (normalized stems). Crude on purpose.
SENSITIVE_WORDS = [
    "حساسیت", "آلرژی", "بیمار", "باردار", "حامله", "شیردهی", "دارو", "قرص", "دیابت", "سرطان", "افسردگی",
    "اضطراب", "روانپزشک", "جراحی", "مذهب", "سیاسی", "حزب", "بدهی", "قرض", "وام", "طلاق", "خشونت", "کتک",
    "پلیس", "دادگاه", "زندان", "allerg", "pregnan", "diabet", "cancer", "religio", "politic", "debt",
    "police", "court",
]


def is_sensitive(text: str) -> bool:
    n = _norm(text)
    return any(w in n for w in SENSITIVE_WORDS)


def has_long_number(text: str) -> bool:
    return bool(re.search(r"\d{7,}", _norm(text)))


# ---------------------------------------------------------------- identity / semantic memory
def touch(conv_id: str, now):
    return db.upsert_customer(conv_id, now)


def profile(conv_id: str):
    return db.get_customer(conv_id)


def set_profile(conv_id: str, name: str, phone: str, now) -> None:
    db.upsert_customer(conv_id, now)
    db.set_customer_profile(conv_id, name, phone)


def remember_note(conv_id: str, note: str, now):
    """Store a short note the customer asked us to remember. Returns (ok, reason)."""
    n = re.sub(r"\s+", " ", str(note)).strip()
    if len(n) < 3 or len(n) > 120:
        return False, "length"
    if is_sensitive(n):
        return False, "sensitive"
    if has_long_number(n) or "http" in n.lower():
        return False, "number_or_link"
    c = db.upsert_customer(conv_id, now)
    key = "note:" + hashlib.sha1(_norm(n).encode("utf-8")).hexdigest()[:8]
    notes = [f for f in db.get_facts(c["id"]) if f["key"].startswith("note:")]
    if len(notes) >= MAX_NOTES and key not in {f["key"] for f in notes}:
        db.delete_fact(c["id"], notes[0]["key"])                      # drop the oldest note
    db.set_fact(c["id"], key, n, "stated", now)
    return True, ""


def _period_of(hour: int) -> str:
    return "morning" if hour < 12 else ("afternoon" if hour < 17 else "evening")


PERIOD_FA = {"morning": "صبح", "afternoon": "بعدازظهر", "evening": "عصر"}


def derive_facts(conv_id: str, now) -> dict:
    """Recompute facts from the bookings table. Exact and deterministic: no AI involved."""
    c = db.get_customer(conv_id)
    rows = db.bookings_of_conv(conv_id) if c else []
    if not rows:
        return {}
    starts = [datetime.datetime.fromisoformat(r["start"]) for r in rows]
    svc = Counter(s for r in rows for s in r["services"].split(" + "))
    facts = {"confirmed_bookings": str(len(rows)),
             "last_booking_date": max(starts).date().isoformat(),
             "last_services": rows[-1]["services"]}
    usual = [s for s, n in svc.most_common(2) if n >= 2]
    if usual:
        facts["usual_services"] = " + ".join(usual)
    if len(rows) >= 3:
        period, n = Counter(_period_of(t.hour) for t in starts).most_common(1)[0]
        if n / len(rows) >= 0.6:
            facts["preferred_period"] = period
    for key in ("usual_services", "preferred_period"):                # remove derived facts that no longer hold
        if key not in facts:
            db.delete_fact(c["id"], key)
    for k, v in facts.items():
        db.set_fact(c["id"], k, v, "derived", now)
    return facts


# ---------------------------------------------------------------- episodic memory
def record_booking(conv_id: str, services: str, when_text: str, now) -> None:
    c = db.upsert_customer(conv_id, now)
    db.add_episode(c["id"], "booking", f"نوبت قطعی: {services} - {when_text}", None, now)


def record_handoff(conv_id: str, reason: str, now) -> None:
    c = db.upsert_customer(conv_id, now)
    text, r = "ارجاع به کارمند", (reason or "").strip()[:60]
    if r and not is_sensitive(r) and not has_long_number(r):
        text += f": {r}"
    db.add_episode(c["id"], "handoff", text, None, now)


# ---------------------------------------------------------------- what goes into the prompt
def context_for(conv_id: str, now, fmt_date) -> str:
    """Compact memory block for the prompt ('' for a customer we know nothing about)."""
    c = db.get_customer(conv_id)
    if not c:
        return ""
    facts = {f["key"]: f["value"] for f in db.get_facts(c["id"])}
    lines = []
    if c["name"] or c["phone"]:
        lines.append(f"- On file: name {c['name'] or '?'}; mobile {c['phone'] or '?'}")
    if "confirmed_bookings" in facts:
        last = fmt_date(datetime.date.fromisoformat(facts["last_booking_date"]))
        lines.append(f"- Confirmed bookings so far: {facts['confirmed_bookings']}; last on {last} ({facts['last_services']})")
    if "usual_services" in facts:
        lines.append(f"- Usual services: {facts['usual_services']}")
    if "preferred_period" in facts:
        lines.append(f"- Usually books in the {PERIOD_FA[facts['preferred_period']]}")
    notes = [v for k, v in facts.items() if k.startswith("note:")]
    if notes:
        lines.append("- Customer asked us to remember: " + "; ".join(notes))
    summ = db.recent_episodes(c["id"], limit=1, kinds=["summary"])
    if summ:
        lines.append("- Older chats (automatic summary, unverified): " + summ[0]["text"])
    for e in db.recent_episodes(c["id"], limit=2, kinds=["handoff"]):
        lines.append(f"- {e['ts'][:10]}: {e['text']}")
    if not lines:
        return ""
    head = "Returning customer. Memory about THIS chat's customer only:"
    tail = ("Use it only when it helps (offer the same service; reuse name/mobile after the customer confirms them). "
            "Never recite it unprompted and never mention it to anyone else.")
    kept, size = [], len(head) + len(tail)
    for ln in lines:                                                   # priority order; drop what does not fit
        if size + len(ln) + 1 > CONTEXT_CHARS:
            break
        kept.append(ln)
        size += len(ln) + 1
    return "\n".join([head] + kept + [tail]) if kept else ""


# ---------------------------------------------------------------- procedural memory (skills)
def load_skills() -> list:
    out = []
    if not SKILLS_DIR.exists():
        return out
    for p in sorted(SKILLS_DIR.glob("*.md")):
        m = re.match(r"---\r?\n(.*?)\r?\n---\r?\n(.*)", p.read_text(encoding="utf-8"), re.S)
        if not m:
            continue
        header = {}
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                header[k.strip()] = v.strip()
        triggers = [_norm(t) for t in re.split(r"[,،]", header.get("triggers", "")) if t.strip()]
        out.append({"name": header.get("name", p.stem), "triggers": triggers, "body": m.group(2).strip()})
    return out


def select_skills(user_text: str, max_n: int = 2) -> list:
    t = _norm(user_text)
    scored = [(sum(1 for tr in s["triggers"] if tr in t), s) for s in load_skills()]
    scored = [x for x in scored if x[0] > 0]
    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored[:max_n]]


def skills_block(user_text: str) -> str:
    chosen = select_skills(user_text)
    if not chosen:
        return ""
    out, size = "Procedures for this situation (follow them):", 0
    for s in chosen:
        piece = f"\n[{s['name']}]\n{s['body']}"
        if size + len(piece) > SKILLS_CHARS:
            break
        out += piece
        size += len(piece)
    return out if size else ""


def build_extra(conv_id: str, now, fmt_date, user_text: str) -> str:
    return "\n\n".join(x for x in (context_for(conv_id, now, fmt_date), skills_block(user_text)) if x)


# ---------------------------------------------------------------- working memory
def restore_turns(conv_id: str, now) -> list:
    """Last few turns if the customer wrote within RESTORE_HOURS (so a restart does not lose the thread)."""
    since = db.ts(now - datetime.timedelta(hours=RESTORE_HOURS))
    msgs = []
    for r in db.recent_turns(conv_id, since, RESTORE_TURNS):
        if r["user_text"]:
            msgs.append({"role": "user", "content": r["user_text"]})
        if r["bot_text"]:
            msgs.append({"role": "assistant", "content": r["bot_text"]})
    return msgs


# ---------------------------------------------------------------- forgetting and distillation
def forget(conv_id: str) -> dict:
    return db.forget_conversation(conv_id)


def distill(conv_id: str, now, summarizer, days: int = 30, apply: bool = False) -> dict:
    """Condense raw turns older than `days`. summarizer(transcript) -> short Persian text or 'NONE'.
    Without apply=True nothing is changed (dry run)."""
    cutoff = db.ts(now - datetime.timedelta(days=days))
    turns = db.turns_before(conv_id, cutoff)
    report = {"conv_id": conv_id, "old_turns": len(turns), "summary": None, "status": "", "applied": apply}
    if not turns:
        report["status"] = "nothing to distill"
        return report
    transcript = "\n".join(f"Customer: {t['user_text']}\nAssistant: {t['bot_text']}" for t in turns)[-4000:]
    try:
        raw = (summarizer(transcript) or "").strip()
    except Exception as e:                                             # model down: keep everything
        report["status"] = f"summarizer failed ({e}); raw turns kept"
        return report
    if not raw or raw.upper().startswith("NONE"):
        report["status"] = "nothing durable to keep"
    elif is_sensitive(raw) or has_long_number(raw) or len(raw) > 400:
        report["status"] = "summary rejected by the privacy filter"
    else:
        report["summary"], report["status"] = raw, "ok"
    if apply:
        c = db.upsert_customer(conv_id, now)
        derive_facts(conv_id, now)
        if report["summary"]:
            db.add_episode(c["id"], "summary", report["summary"], None, now)
        report["deleted_turns"] = db.delete_turns_before(conv_id, cutoff)
    return report


def dump(conv_id: str) -> str:
    """Human-readable view of everything remembered about a chat (for the owner / debugging)."""
    c = db.get_customer(conv_id)
    if not c:
        return f"(nothing stored for '{conv_id}')"
    out = [f"customer #{c['id']}  conv_id={c['conv_id']}  name={c['name']}  phone={c['phone']}  last_seen={c['last_seen']}",
           "facts:"]
    out += [f"  [{f['source']}] {f['key']} = {f['value']}" for f in db.get_facts(c["id"])] or ["  (none)"]
    out.append("episodes (latest 8):")
    out += [f"  {e['ts'][:16]}  [{e['kind']}] {e['text']}" for e in db.recent_episodes(c["id"], limit=8)] or ["  (none)"]
    return "\n".join(out)
