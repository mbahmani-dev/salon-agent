"""
Look inside salon.db without any extra software.

    py show_db.py          latest bookings (with their payments) and owner notifications
    py show_db.py --csv    also write bookings.csv (opens in Excel)
"""
import csv
import sys
from pathlib import Path

import db

STATUS_FA = {"booked": "قطعی", "pending_payment": "منتظر پرداخت", "expired": "منقضی", "cancelled": "لغو"}


def main() -> None:
    con = db.get_con()
    print(f"database: {db.DB_PATH}\n")

    rows = con.execute(
        "SELECT b.*, p.status AS pay_status, p.amount_toman AS pay_amount, p.bank_ref AS bank_ref "
        "FROM bookings b LEFT JOIN payments p ON p.id = (SELECT MAX(id) FROM payments WHERE booking_id = b.id) "
        "ORDER BY b.id DESC LIMIT 20").fetchall()
    print(f"=== bookings (latest {len(rows)}) ===")
    if not rows:
        print("(none yet)")
    for r in rows:
        print(f"#{r['id']}  status={r['status']} ({STATUS_FA.get(r['status'], '')})")
        print(f"    customer: {r['customer_name']}  {r['phone']}")
        print(f"    services: {r['services']}")
        print(f"    start:    {r['start']}   duration: {r['duration_min']} min")
        print(f"    total:    {r['total_toman']:,}   deposit: {r['deposit_toman']:,}")
        pay = (f"{r['pay_status']}, {r['pay_amount']:,}, ref={r['bank_ref']}" if r["pay_status"] else "no payment record")
        print(f"    payment:  {pay}")
        if r["status"] == "pending_payment":
            print(f"    hold until: {r['hold_until']}")
        print()

    print("=== owner notifications (latest 10) ===")
    for r in con.execute("SELECT ts, text FROM notifications ORDER BY id DESC LIMIT 10"):
        print(f"{r['ts']}  {r['text']}")
    print()
    n = con.execute("SELECT COUNT(*) AS n FROM conversation_log").fetchone()["n"]
    print(f"conversation turns logged: {n}")

    if "--csv" in sys.argv:
        out = Path(db.DB_PATH).with_name("bookings.csv")
        data = con.execute("SELECT * FROM bookings ORDER BY id").fetchall()
        with open(out, "w", newline="", encoding="utf-8-sig") as f:      # utf-8-sig: Excel shows Persian correctly
            w = csv.writer(f)
            w.writerow(data[0].keys() if data else ["id"])
            for r in data:
                w.writerow(list(r))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
