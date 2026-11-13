#!/usr/bin/env python3
"""
Dataset generator for the Lexsi Track 3 entry.

Turns the Recovery Copilot rule engine (Qasim Jethwa's shipped, deterministic
failed-payment recovery engine — github.com/qasimjethwa/Failed-payment-recovery)
into tool-calling tasks for agent fine-tuning and auditing.

The rule table, message templates and ID/amount generation below are a faithful
Python port of the JavaScript in Recovery Copilot's index.html (FAILURE_RULES,
TEMPLATES, hashString, generateMessage, generateDataset). Every expected tool
call is derived deterministically from that logic, so the dataset's gold labels
are exactly as trustworthy as the engine merchants already rely on — no human
labelling, no LLM-generated answers.

Splits:
  - train: balanced across all 8 failure reasons.
  - eval (held out): natural hash-driven mix like the original app, plus
    hand-picked edge cases (e.g. UPI as the original method, which flips the
    alternate-method branch). Disjoint transaction IDs from train (asserted).

Usage:
    python3 generate_dataset.py            # writes train.jsonl + eval.jsonl
"""

import ctypes
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Ported from Recovery Copilot index.html — do not "improve"; fidelity matters.
# ---------------------------------------------------------------------------

CUSTOMER_NAMES = [
    "Aarav Mehta", "Priya Nair", "Rohan Kapoor", "Sneha Iyer", "Kabir Singh",
    "Ananya Rao", "Vivaan Joshi", "Ishita Desai", "Arjun Verma", "Diya Pillai",
    "Reyansh Gupta", "Meera Pillai", "Karthik Reddy", "Tanya Malhotra", "Yash Trivedi",
    "Naina Chawla", "Aditya Bhat", "Riya Sharma", "Dev Patel", "Simran Kaur",
    "Neha Kulkarni", "Farhan Shaikh", "Pooja Menon", "Aryan Bhatia",
]

PAYMENT_METHODS = ["UPI", "Credit Card", "Debit Card", "Netbanking", "Wallet"]

# key -> (label, action, channel, delay_minutes, success_rate_assumption)
# success_rate is the engine's configured assumption used by the original app's
# demo simulation. It is NOT a measured result and is not used as an entry metric.
FAILURE_RULES = {
    "insufficient_funds":   ("Insufficient Funds", "Delay retry 3 hrs, then offer UPI as alternate rail", "SMS", 180, 0.58),
    "otp_timeout":           ("OTP Timeout", "Instant retry nudge on the same method", "SMS", 2, 0.71),
    "bank_server_error":     ("Bank Server Error", "Silent auto-retry, no customer message needed", "System", 30, 0.66),
    "card_declined":         ("Card Declined by Issuer", "Suggest an alternate payment method", "WhatsApp", 15, 0.49),
    "network_drop":          ("Network Drop Mid-Transaction", "Immediate one-tap retry link", "SMS", 1, 0.74),
    "wrong_cvv":             ("Incorrect CVV", "Prompt to securely re-enter card details", "Email", 10, 0.62),
    "expired_card":          ("Card Expired", "Prompt to update card, offer UPI/Wallet fallback", "Email", 20, 0.41),
    "daily_limit_exceeded":  ("Daily Limit Exceeded", "Suggest alternate method or retry after midnight", "WhatsApp", 240, 0.53),
}
FAILURE_KEYS = list(FAILURE_RULES.keys())


def _sms(n, a, m, v):
    first = n.split(" ")[0]
    return [
        f"Hi {first}, your payment of ₹{a} didn't go through. Tap to retry with {m} — takes 10 seconds.",
        f"{first}, we noticed your ₹{a} payment failed. No charge was made. Retry anytime here.",
        f"Quick nudge — your order for ₹{a} is on hold. Complete payment via {m} to confirm it.",
    ][v]


def _whatsapp(n, a, m, v):
    first = n.split(" ")[0]
    return [
        f"Hey {first} — your ₹{a} payment via {m} didn't complete. Want to try UPI instead? One tap below.",
        f"{first}, looks like {m} declined your ₹{a} payment. Happens sometimes — try a different method here.",
        "Your cart is still saved. Payment of ₹{a} didn't clear. Switch payment method and finish in seconds.".format(a=a),
    ][v]


def _email(n, a, m, v):
    return [
        f"Hi {n}, we couldn't process your payment of ₹{a} using {m}. Please review your card details and try again.",
        f"Hello {n}, your recent transaction of ₹{a} was not completed. Update your payment method to complete the order.",
        f"{n}, your {m} appears to have expired or been declined for ₹{a}. Update it here to avoid losing your order.",
    ][v]


def _system(n, a, m, v):
    return [
        "Automated retry scheduled — no customer action required.",
        "Silent recovery attempt queued via payment gateway failover.",
        "Backend retry in progress, customer not notified.",
    ][v]


TEMPLATE_FNS = {"SMS": _sms, "WhatsApp": _whatsapp, "Email": _email, "System": _system}


def js_hash(s: str) -> int:
    """Port of the JS hashString(): 32-bit signed wrap each step, then Math.abs."""
    h = 0
    for ch in s:
        h = ctypes.c_int32((h << 5) - h + ord(ch)).value
    return abs(h)


def inr_format(n: int) -> str:
    """Port of Number.toLocaleString('en-IN') for the amounts used here."""
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail])


def generate_message(txn_id, customer, amount, method, channel):
    variant = js_hash(txn_id) % 3
    return TEMPLATE_FNS[channel](customer, inr_format(amount), method, variant)


# ---------------------------------------------------------------------------
# Tool layer (the entry's design): the surface the agent model must learn.
# Gold call sequences are derived from each rule's action text.
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "schedule_retry",
        "description": "Schedule a retry of the failed payment on the original payment method after a delay.",
        "parameters": {
            "txn_id": {"type": "string"},
            "delay_minutes": {"type": "integer", "minimum": 1},
            "payment_method": {"type": "string", "enum": PAYMENT_METHODS},
        },
        "required": ["txn_id", "delay_minutes", "payment_method"],
    },
    {
        "name": "send_customer_message",
        "description": "Send the customer a recovery message on a channel after a delay.",
        "parameters": {
            "txn_id": {"type": "string"},
            "channel": {"type": "string", "enum": ["SMS", "WhatsApp", "Email"]},
            "delay_minutes": {"type": "integer", "minimum": 0},
            "message": {"type": "string"},
        },
        "required": ["txn_id", "channel", "delay_minutes", "message"],
    },
    {
        "name": "offer_alternate_method",
        "description": "Offer the customer an alternate payment method to complete the payment.",
        "parameters": {
            "txn_id": {"type": "string"},
            "alternate_method": {"type": "string", "enum": PAYMENT_METHODS},
        },
        "required": ["txn_id", "alternate_method"],
    },
    {
        "name": "silent_retry",
        "description": "Retry silently in the backend. The customer is NOT contacted.",
        "parameters": {
            "txn_id": {"type": "string"},
            "delay_minutes": {"type": "integer", "minimum": 1},
        },
        "required": ["txn_id", "delay_minutes"],
    },
]

# Which plan shape each failure reason maps to (derived from the rule's action):
#   retry_then_message : schedule_retry + send_customer_message
#   silent             : silent_retry only — contacting the customer is an error
#   message_only       : send_customer_message only
#   message_then_offer : send_customer_message + offer_alternate_method
#   full               : schedule_retry + send_customer_message + offer_alternate_method
PLAN_SHAPE = {
    "insufficient_funds": "full",                # "Delay retry 3 hrs, then offer UPI as alternate rail"
    "otp_timeout": "retry_then_message",         # "Instant retry nudge on the same method"
    "bank_server_error": "silent",               # "Silent auto-retry, no customer message needed"
    "card_declined": "message_then_offer",       # "Suggest an alternate payment method"
    "network_drop": "retry_then_message",        # "Immediate one-tap retry link"
    "wrong_cvv": "message_only",                 # "Prompt to securely re-enter card details"
    "expired_card": "message_then_offer",        # "Prompt to update card, offer UPI/Wallet fallback"
    "daily_limit_exceeded": "message_then_offer",  # "Suggest alternate method or retry after midnight"
}


def alternate_for(method: str) -> str:
    """Alternate rail offered to the customer: UPI, unless they already used UPI."""
    return "UPI" if method != "UPI" else "Wallet"


def gold_calls(txn):
    """Deterministic gold tool-call sequence for one case, from the rule table."""
    reason = txn["failure_reason"]
    label, action, channel, delay, _rate = FAILURE_RULES[reason]
    tid, method = txn["txn_id"], txn["payment_method"]
    shape = PLAN_SHAPE[reason]
    calls = []
    if shape in ("retry_then_message", "full"):
        calls.append({"tool": "schedule_retry",
                      "arguments": {"txn_id": tid, "delay_minutes": delay, "payment_method": method}})
    if shape == "silent":
        calls.append({"tool": "silent_retry",
                      "arguments": {"txn_id": tid, "delay_minutes": delay}})
        return calls
    msg = generate_message(tid, txn["customer"], txn["amount_inr"], method, channel)
    calls.append({"tool": "send_customer_message",
                  "arguments": {"txn_id": tid, "channel": channel, "delay_minutes": delay, "message": msg}})
    if shape in ("message_then_offer", "full"):
        calls.append({"tool": "offer_alternate_method",
                      "arguments": {"txn_id": tid, "alternate_method": alternate_for(method)}})
    return calls


# ---------------------------------------------------------------------------
# Case generation (mirrors the original generateDataset hashing scheme,
# with split-specific seed words so the splits never share a transaction).
# ---------------------------------------------------------------------------

# Each seed word owns a numeric ID block so splits can never collide
# (the original app's single 100000 block only ever held 24 cases).
ID_BLOCKS = {"lexsi-train-": 200000, "lexsi-eval-": 500000, "lexsi-edge-": 800000}


def make_txn(seed_word, i, reason_key=None, force_method=None):
    txn_id = "TXN" + str(ID_BLOCKS[seed_word] + i * 7 + js_hash(seed_word + str(i)) % 900)
    customer = CUSTOMER_NAMES[js_hash(txn_id + "name") % len(CUSTOMER_NAMES)]
    method = force_method or PAYMENT_METHODS[js_hash(txn_id + "method") % len(PAYMENT_METHODS)]
    amount = 199 + (js_hash(txn_id + "amt") % 48) * 63
    if reason_key is None:
        reason_key = FAILURE_KEYS[js_hash(txn_id + "reason") % len(FAILURE_KEYS)]
    return {
        "txn_id": txn_id,
        "customer": customer,
        "payment_method": method,
        "amount_inr": amount,
        "failure_reason": reason_key,
        "failure_label": FAILURE_RULES[reason_key][0],
    }


def to_row(split, idx, txn):
    label, action, channel, delay, _rate = FAILURE_RULES[txn["failure_reason"]]
    return {
        "id": f"{split}-{idx:04d}",
        "split": split,
        "instruction": (
            f"A payment has failed and must be recovered. Transaction {txn['txn_id']}: "
            f"customer {txn['customer']}, payment method {txn['payment_method']}, "
            f"amount ₹{inr_format(txn['amount_inr'])}. Diagnosed failure reason: {label}. "
            "Execute the correct recovery plan using the available tools, in order."
        ),
        "scenario": txn,
        "tools": TOOLS,
        "expected_calls": gold_calls(txn),
        "expected_decision": {"action": action, "channel": channel, "delay_minutes": delay},
    }


def build_train(count=400):
    rows = []
    for i in range(count):
        txn = make_txn("lexsi-train-", i, reason_key=FAILURE_KEYS[i % len(FAILURE_KEYS)])
        rows.append(to_row("train", i + 1, txn))
    return rows


def build_eval(count=120):
    rows = []
    for i in range(count):
        txn = make_txn("lexsi-eval-", i)  # natural hash-driven mix, like the original app
        rows.append(to_row("eval", i + 1, txn))
    # Edge cases: UPI as the original method flips the alternate-method branch
    # (offer must become Wallet), and bank_server_error must stay silent.
    edge_specs = [
        ("insufficient_funds", "UPI"), ("card_declined", "UPI"),
        ("expired_card", "UPI"), ("daily_limit_exceeded", "UPI"),
        ("bank_server_error", "Wallet"), ("otp_timeout", "UPI"),
        ("network_drop", "Wallet"), ("wrong_cvv", "UPI"),
    ]
    for j, (reason, method) in enumerate(edge_specs):
        txn = make_txn("lexsi-edge-", j, reason_key=reason, force_method=method)
        rows.append(to_row("eval", count + j + 1, txn))
    return rows


def main():
    train = build_train()
    ev = build_eval()
    train_ids = {r["scenario"]["txn_id"] for r in train}
    eval_ids = {r["scenario"]["txn_id"] for r in ev}
    assert not (train_ids & eval_ids), "transaction IDs overlap between splits!"
    with open(os.path.join(HERE, "train.jsonl"), "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(HERE, "eval.jsonl"), "w") as f:
        for r in ev:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for name, rows in (("train", train), ("eval", ev)):
        dist = {}
        for r in rows:
            dist[r["scenario"]["failure_reason"]] = dist.get(r["scenario"]["failure_reason"], 0) + 1
        print(f"{name}: {len(rows)} rows | per-reason: {dist}")
    print(f"unique txn ids — train: {len(train_ids)}, eval: {len(eval_ids)}, overlap: 0")


if __name__ == "__main__":
    main()
