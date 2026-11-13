# Recovery Copilot

**Track 3: AI Revenue Recovery — Razorpay AI Builder Internship 2026**

A dashboard that watches failed payments, diagnoses *why* each one failed, and automatically queues the right recovery action and outreach message — turning checkout drop-offs back into revenue.

Live logic only, no external APIs, no build step. Open `index.html` in a browser and it runs.

---

## Project Objective — What does it solve?

Every payment gateway loses a meaningful slice of revenue not to fraud or bad products, but to **failed transactions that were never retried the right way**. A payment can fail for a dozen different reasons — insufficient funds, a timed-out OTP, a flaky network, an expired card — and each of those reasons calls for a *different* recovery strategy and a *different* message. Most merchants treat every failure the same way (one generic "your payment failed" email, if anything at all), which is why so much recoverable revenue is simply left on the table.

Recovery Copilot solves this by treating payment recovery as a **triage problem, not a broadcast problem**:

1. **Classify** — every failed transaction is read for its specific failure reason.
2. **Decide** — a rule engine maps that failure reason to a recovery action: which channel to use (SMS, WhatsApp, Email, or a silent backend retry), how long to wait before retrying, and whether to nudge the customer toward an alternate payment method.
3. **Act** — a personalized recovery message is generated automatically for that specific customer, amount, and payment method — no manual copywriting per case.
4. **Measure** — the dashboard tracks recovery rate and revenue recovered in real time, so the system's effectiveness is visible, not assumed.

This is the same shape as real fintech recovery infrastructure (used by Stripe, Razorpay's own smart-retry systems, and PayPal), rebuilt here at prototype scale to demonstrate the reasoning, not to reinvent the payments engineering underneath it.

**Why I built this specifically:** I run an e-commerce brand (Kemono, a D2C belt label) on Shopify and have watched real checkout failures cost real sales. This isn't a hypothetical problem for me — it's one I've seen from the merchant side.

---

## Build Challenges & Technical Obstacles

**1. Avoiding a fake "AI" that's actually a black box.**
The obvious shortcut was to bolt on a call to a language model and call it a day. I deliberately didn't — a hackathon judge (or a real merchant) can't audit a black-box output on a payments flow. Instead I built an explicit, inspectable rule engine: every recovery decision traces back to a named failure reason with a stated channel, delay, and success assumption. It's less flashy than "AI generated," but it's the version I could actually explain and defend, and it's the version that doesn't quietly recommend a wrong action.

**2. Making the "AI-generated message" feel personalized without an API.**
Without a live model call, the risk was every message reading like an obvious mail-merge. I solved this with a **deterministic template bank per channel** (3 tone variants each) selected by a hash of the transaction ID, so the same transaction always produces the same message on reload — it reads as intentional personalization rather than random noise, while staying fully self-contained and deployable with zero setup or API key.

**3. Representing "recovery" visually without misleading stats.**
A revenue-recovery dashboard invites vanity metrics. I constrained the stats strip to four numbers that are all directly traceable to the visible table rows (failed count, attempts queued, amount recovered, recovery rate) rather than an aggregate "AI confidence score" that would have been unfounded. The animated pulse line only spikes when a real row in the table flips to "Recovered," so the visual never claims more than the data underneath it.

**4. Keeping it deployable in a single evening with zero dependencies.**
Given the timeline, I ruled out any framework, build step, or backend. The whole thing is vanilla HTML/CSS/JS — three files, no `npm install`, works instantly on GitHub Pages. The trade-off is no persistent storage or real payment gateway integration, which I've noted below as the clear next step rather than pretending it's out of scope.

---

## What's next (if this moved past prototype)

- Replace the synthetic dataset with real webhook data from Razorpay's payment failure events.
- A/B test template variants against actual recovery rate instead of assumed success rates.
- Add a merchant-facing settings panel to customize delay windows and channel priority per business type.

---

## Tech

Vanilla HTML / CSS / JavaScript. No frameworks, no build tools, no external API calls. Fonts loaded from Google Fonts (Space Grotesk, Inter, JetBrains Mono).

## Run it

Clone the repo and open `index.html` in any browser — that's the entire setup.
