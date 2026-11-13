#!/usr/bin/env python3
"""
Oracle predictor for the Lexsi Track 3 entry.

An INDEPENDENT re-implementation of the Recovery Copilot rule engine, written
from the shipped JavaScript (index.html: FAILURE_RULES, TEMPLATES, hashString,
generateMessage). It reads only the scenario fields of an eval JSONL and derives
the recovery tool calls itself — it never reads the dataset's expected_calls.

Purpose: cross-check. If this second implementation and the dataset generator
agree on every held-out case, the Python port of the engine is consistent.
Scored with evaluate.py, the oracle must reach perfect scores; anything less
means a porting bug somewhere, which is exactly what this check exists to catch.

Usage:
    python3 oracle.py --gold eval.jsonl --out predictions_oracle.jsonl
    python3 oracle.py --gold eval.jsonl --out predictions_corrupt.jsonl --corrupt
"""

import argparse
import ctypes
import json

# --- Independent port of the engine's data (from index.html) ---

RULES = {  # reason -> (channel, delay_minutes, plan shape)
    "insufficient_funds": ("SMS", 180, "full"),
    "otp_timeout": ("SMS", 2, "retry_then_message"),
    "bank_server_error": ("System", 30, "silent"),
    "card_declined": ("WhatsApp", 15, "message_then_offer"),
    "network_drop": ("SMS", 1, "retry_then_message"),
    "wrong_cvv": ("Email", 10, "message_only"),
    "expired_card": ("Email", 20, "message_then_offer"),
    "daily_limit_exceeded": ("WhatsApp", 240, "message_then_offer"),
}

SMS_T = [
    "Hi {f}, your payment of ₹{a} didn't go through. Tap to retry with {m} — takes 10 seconds.",
    "{f}, we noticed your ₹{a} payment failed. No charge was made. Retry anytime here.",
    "Quick nudge — your order for ₹{a} is on hold. Complete payment via {m} to confirm it.",
]
WHATSAPP_T = [
    "Hey {f} — your ₹{a} payment via {m} didn't complete. Want to try UPI instead? One tap below.",
    "{f}, looks like {m} declined your ₹{a} payment. Happens sometimes — try a different method here.",
    "Your cart is still saved. Payment of ₹{a} didn't clear. Switch payment method and finish in seconds.",
]
EMAIL_T = [
    "Hi {n}, we couldn't process your payment of ₹{a} using {m}. Please review your card details and try again.",
    "Hello {n}, your recent transaction of ₹{a} was not completed. Update your payment method to complete the order.",
    "{n}, your {m} appears to have expired or been declined for ₹{a}. Update it here to avoid losing your order.",
]
SYSTEM_T = [
    "Automated retry scheduled — no customer action required.",
    "Silent recovery attempt queued via payment gateway failover.",
    "Backend retry in progress, customer not notified.",
]
TEMPLATES = {"SMS": SMS_T, "WhatsApp": WHATSAPP_T, "Email": EMAIL_T, "System": SYSTEM_T}


def h32(s):
    h = 0
    for ch in s:
        h = ctypes.c_int32((h << 5) - h + ord(ch)).value
    return abs(h)


def inr(n):
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    groups = []
    while head:
        groups.insert(0, head[-2:] if len(head) > 2 else head)
        head = head[:-2] if len(head) > 2 else ""
        if len(head) <= 2:
            if head:
                groups.insert(0, head)
            break
    return ",".join(groups + [tail])


def message(txn_id, customer, amount, method, channel):
    tpl = TEMPLATES[channel][h32(txn_id) % 3]
    return tpl.format(f=customer.split(" ")[0], n=customer, a=inr(amount), m=method)


def predict(scn):
    channel, delay, shape = RULES[scn["failure_reason"]]
    tid, method = scn["txn_id"], scn["payment_method"]
    calls = []
    if shape in ("retry_then_message", "full"):
        calls.append({"tool": "schedule_retry",
                      "arguments": {"txn_id": tid, "delay_minutes": delay, "payment_method": method}})
    if shape == "silent":
        calls.append({"tool": "silent_retry",
                      "arguments": {"txn_id": tid, "delay_minutes": delay}})
        return calls
    calls.append({"tool": "send_customer_message",
                  "arguments": {"txn_id": tid, "channel": channel, "delay_minutes": delay,
                                "message": message(tid, scn["customer"], scn["amount_inr"], method, channel)}})
    if shape in ("message_then_offer", "full"):
        calls.append({"tool": "offer_alternate_method",
                      "arguments": {"txn_id": tid,
                                    "alternate_method": "UPI" if method != "UPI" else "Wallet"}})
    return calls


def corrupt(calls, scn):
    """Deliberate errors for the negative control: wrong channel on OTP cases,
    and contacting the customer on bank-server-error cases (a safety violation
    the audit must catch)."""
    out = []
    for c in calls:
        c = {"tool": c["tool"], "arguments": dict(c["arguments"])}
        if scn["failure_reason"] == "otp_timeout" and c["tool"] == "send_customer_message":
            c["arguments"]["channel"] = "Email"
        out.append(c)
    if scn["failure_reason"] == "bank_server_error":
        out.append({"tool": "send_customer_message",
                    "arguments": {"txn_id": scn["txn_id"], "channel": "SMS", "delay_minutes": 0,
                                  "message": "Your payment failed, please retry."}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corrupt", action="store_true")
    args = ap.parse_args()
    with open(args.gold) as f, open(args.out, "w") as out:
        for line in f:
            row = json.loads(line)
            calls = predict(row["scenario"])
            if args.corrupt:
                calls = corrupt(calls, row["scenario"])
            out.write(json.dumps({"id": row["id"], "predicted_calls": calls}, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
