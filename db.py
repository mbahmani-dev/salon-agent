"""
SQLite storage for the salon agent (standard library only).

Tables
  bookings          one row per visit. status: pending_payment -> booked | expired | cancelled
  payments          one row per payment attempt for a booking
  notifications     messages meant for the salon owner
  conversation_log  every customer turn (to measure the pilot)

Slots are protected inside a transaction (BEGIN IMMEDIATE), so two customers can never
take the last free place at the same moment. A booking that waits for payment HOLDS its
slot until `hold_until`; after that the hold is released (status 'expired').
"""
import datetime
import json
import os
import sqlite3
import threading
from pathlib import Path

DB_PATH = Path(os.environ.get("SALON_DB", Path(__file__).resolve().parent / "salon.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS bookings(
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at     TEXT NOT NULL,
  conv_id        TEXT NOT NULL,
  customer_name  TEXT NOT NULL,
  phone          TEXT NOT NULL,
  services       TEXT NOT NULL,
  start          TEXT NOT NULL,
  duration_min   INTEGER NOT NULL,
  total_toman    INTEGER NOT NULL,
  deposit_toman  INTEGER NOT NULL,
  status         TEXT NOT NULL CHECK (status IN ('pending_payment','booked','expired','cancelled')),
  hold_until     TEXT
);
CREATE INDEX IF NOT EXISTS idx_bookings_start ON bookings(start);
CREATE INDEX IF NOT EXISTS idx_bookings_conv  ON bookings(conv_id);

CREATE TABLE IF NOT EXISTS payments(
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  booking_id     INTEGER NOT NULL REFERENCES bookings(id),
  provider       TEXT NOT NULL,
  provider_ref   TEXT NOT NULL UNIQUE,
  amount_toman   INTEGER NOT NULL,
  status         TEXT NOT NULL CHECK (status IN ('created','paid','failed','expired')),
  pay_url        TEXT,
  bank_ref       TEXT,
  created_at     TEXT NOT NULL,
  paid_at        TEXT
);

CREATE TABLE IF NOT EXISTS notifications(
  id    INTEGER PRIMARY KEY AUTOINCREMENT,
  ts    TEXT NOT NULL,
  text  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversation_log(
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  ts        TEXT NOT NULL,
  conv_id   TEXT,
  user_text TEXT,
  bot_text  TEXT,
  tools     TEXT
);
"""

_local = threading.local()          # one connection per thread (the payment callback runs in another thread)


def get_con() -> sqlite3.Connection:
    con = getattr(_local, "con", None)
    if con is None or getattr(_local, "path", None) != str(DB_PATH):
        con = sqlite3.connect(DB_PATH, isolation_level=None, timeout=10)   # autocommit; we use BEGIN explicitly
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA journal_mode=WAL")
        con.executescript(SCHEMA)
        _local.con, _local.path = con, str(DB_PATH)
    return con


def ts(dt: datetime.datetime) -> str:
    return dt.isoformat(timespec="seconds")


def minutes_iso(dt: datetime.datetime) -> str:
    return dt.isoformat(timespec="minutes")


# ---------------------------------------------------------------- availability
def _is_active(row, now) -> bool:
    return row["status"] == "booked" or (row["status"] == "pending_payment" and row["hold_until"] > ts(now))


def count_overlapping(start: datetime.datetime, minutes: int, now: datetime.datetime) -> int:
    end = start + datetime.timedelta(minutes=minutes)
    rows = get_con().execute(
        "SELECT start, duration_min, status, hold_until FROM bookings "
        "WHERE status IN ('booked','pending_payment') AND substr(start,1,10)=?", (start.date().isoformat(),)
    ).fetchall()
    n = 0
    for r in rows:
        if not _is_active(r, now):
            continue
        bs = datetime.datetime.fromisoformat(r["start"])
        be = bs + datetime.timedelta(minutes=r["duration_min"])
        if bs < end and start < be:
            n += 1
    return n


def expire_holds(now: datetime.datetime) -> int:
    """Release slots whose payment window has passed."""
    con = get_con()
    con.execute("BEGIN IMMEDIATE")
    try:
        ids = [r["id"] for r in con.execute(
            "SELECT id FROM bookings WHERE status='pending_payment' AND hold_until <= ?", (ts(now),))]
        for i in ids:
            con.execute("UPDATE bookings SET status='expired' WHERE id=?", (i,))
            con.execute("UPDATE payments SET status='expired' WHERE booking_id=? AND status='created'", (i,))
        con.execute("COMMIT")
        return len(ids)
    except Exception:
        con.execute("ROLLBACK")
        raise


# ---------------------------------------------------------------- bookings
def create_booking(*, conv_id, name, phone, services, start, minutes, total, deposit, capacity,
                   status, hold_minutes, now):
    """Insert a booking if the slot still has room. Returns the booking id, or None if the slot is full."""
    con = get_con()
    con.execute("BEGIN IMMEDIATE")                       # lock: check + insert are one atomic step
    try:
        if count_overlapping(start, minutes, now) >= capacity:
            con.execute("ROLLBACK")
            return None
        hold = ts(now + datetime.timedelta(minutes=hold_minutes)) if status == "pending_payment" else None
        cur = con.execute(
            "INSERT INTO bookings(created_at,conv_id,customer_name,phone,services,start,duration_min,"
            "total_toman,deposit_toman,status,hold_until) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (ts(now), conv_id, name, phone, services, minutes_iso(start), minutes, total, deposit, status, hold))
        con.execute("COMMIT")
        return cur.lastrowid
    except Exception:
        con.execute("ROLLBACK")
        raise


def get_booking(booking_id: int):
    return get_con().execute("SELECT * FROM bookings WHERE id=?", (booking_id,)).fetchone()


def latest_booking_for_conv(conv_id: str):
    return get_con().execute(
        "SELECT * FROM bookings WHERE conv_id=? ORDER BY id DESC LIMIT 1", (conv_id,)).fetchone()


def find_active_pending(conv_id: str, start: datetime.datetime, now: datetime.datetime):
    return get_con().execute(
        "SELECT * FROM bookings WHERE conv_id=? AND start=? AND status='pending_payment' AND hold_until>? "
        "ORDER BY id DESC LIMIT 1", (conv_id, minutes_iso(start), ts(now))).fetchone()


def set_booking_status(booking_id: int, status: str) -> None:
    get_con().execute("UPDATE bookings SET status=? WHERE id=?", (status, booking_id))


# ---------------------------------------------------------------- payments
def add_payment(*, booking_id, provider, provider_ref, amount, pay_url, now) -> int:
    cur = get_con().execute(
        "INSERT INTO payments(booking_id,provider,provider_ref,amount_toman,status,pay_url,created_at) "
        "VALUES(?,?,?,?, 'created', ?, ?)", (booking_id, provider, provider_ref, amount, pay_url, ts(now)))
    return cur.lastrowid


def get_payment_by_ref(provider_ref: str):
    return get_con().execute("SELECT * FROM payments WHERE provider_ref=?", (provider_ref,)).fetchone()


def payment_for_booking(booking_id: int):
    return get_con().execute(
        "SELECT * FROM payments WHERE booking_id=? ORDER BY id DESC LIMIT 1", (booking_id,)).fetchone()


def latest_open_payment_for_conv(conv_id: str):
    return get_con().execute(
        "SELECT p.* FROM payments p JOIN bookings b ON b.id=p.booking_id "
        "WHERE b.conv_id=? AND p.status='created' ORDER BY p.id DESC LIMIT 1", (conv_id,)).fetchone()


def claim_payment_paid(payment_id: int, bank_ref, now) -> bool:
    """Mark a payment as paid exactly once (a gateway may call the callback twice)."""
    cur = get_con().execute(
        "UPDATE payments SET status='paid', bank_ref=?, paid_at=? WHERE id=? AND status IN ('created','expired')",
        (bank_ref, ts(now), payment_id))
    return cur.rowcount == 1


def set_payment_status(payment_id: int, status: str) -> None:
    get_con().execute("UPDATE payments SET status=? WHERE id=?", (status, payment_id))


# ---------------------------------------------------------------- notifications / log
def add_notification(text: str, now) -> None:
    get_con().execute("INSERT INTO notifications(ts,text) VALUES(?,?)", (ts(now), text))


def log_turn(conv_id, user_text, bot_text, tools, now) -> None:
    get_con().execute(
        "INSERT INTO conversation_log(ts,conv_id,user_text,bot_text,tools) VALUES(?,?,?,?,?)",
        (ts(now), conv_id, user_text, bot_text, json.dumps(tools, ensure_ascii=False)))
