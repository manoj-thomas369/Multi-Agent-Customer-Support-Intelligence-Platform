"""Synthesizes a labeled e-commerce support ticket dataset and FAQ knowledge
base. No proprietary dataset was provided for this project, so tickets are
generated from category-specific templates with randomized entities and
sentiment-bearing phrasing, then labeled using the same rule-of-thumb logic
(category + delay + sentiment) a real support team would use for priority.
"""

import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.nlp.entities import PRODUCT_VOCAB
from src.nlp.sentiment import analyze_sentiment

random.seed(42)

TICKETS_PER_CATEGORY = 220

TEMPLATES: dict[str, list[str]] = {
    "Delivery": [
        "My order #{order_id} for the {product} hasn't been delivered yet, it's been {days} days.",
        "The delivery for order #{order_id} is delayed and I really need the {product} soon.",
        "Order #{order_id} shows as delivered but I never actually received my {product}.",
        "Can you track my shipment for order #{order_id}? The {product} seems to be missing.",
        "I've been waiting {days} days for order #{order_id} with no update on the {product} shipment.",
    ],
    "Refund": [
        "I returned my {product} from order #{order_id} over a week ago but still haven't received my refund.",
        "Requesting a refund for order #{order_id} since the {product} arrived defective.",
        "The refund for order #{order_id} was promised within 5 business days, it's been {days} days now.",
        "I cancelled order #{order_id} but the refund for the {product} hasn't been processed.",
        "Refund status for order #{order_id} still shows pending after {days} days.",
    ],
    "Payment": [
        "I was charged twice for order #{order_id}, please refund the extra amount for the {product}.",
        "Payment failed for order #{order_id} but the amount was still deducted from my account.",
        "I'm unable to complete payment for the {product}, I keep getting an error at checkout.",
        "My card was charged for order #{order_id} but the order never actually went through.",
        "There's an incorrect charge on my account related to order #{order_id} for the {product}.",
    ],
    "Product Issue": [
        "The {product} I received in order #{order_id} arrived damaged.",
        "My {product} from order #{order_id} stopped working after just two days.",
        "I ordered a {product} but received a completely different item in order #{order_id}.",
        "The {product} from order #{order_id} is missing parts and accessories.",
        "The quality of the {product} in order #{order_id} is much worse than what was described.",
    ],
    "Account": [
        "I can't log into my account, the password reset email never arrives.",
        "My account got locked after a few failed login attempts.",
        "I'm unable to update my shipping address on my account profile.",
        "My account shows someone else's order history, this looks like a mix-up.",
        "I want to delete my account but can't find the option anywhere.",
    ],
}

SUFFIXES = {
    "angry": [
        " This is completely unacceptable.",
        " I am extremely frustrated with this service.",
        " This is honestly the worst experience I've had shopping online.",
        " I expect this to be fixed immediately.",
    ],
    "neutral": [
        " Please help resolve this.",
        " Could you please assist me with this?",
        " Let me know the next steps.",
        "",
    ],
    "polite": [
        " I'd really appreciate your help with this.",
        " Thanks in advance for looking into it.",
        " Hoping to get this sorted soon, thank you.",
    ],
}

SENTIMENT_WEIGHTS = [("angry", 0.3), ("neutral", 0.4), ("polite", 0.3)]


def _weighted_choice(weights: list[tuple[str, float]]) -> str:
    labels, probs = zip(*weights)
    return random.choices(labels, weights=probs, k=1)[0]


def _random_order_id() -> str:
    return "ORD" + "".join(random.choices("0123456789", k=5))


def determine_priority(category: str, sentiment_score: float, days: int) -> str:
    score = 0
    if category in ("Payment", "Refund"):
        score += 1
    if category == "Delivery" and days >= 7:
        score += 1
    if sentiment_score <= config.NEGATIVE_SENTIMENT_THRESHOLD:
        score += 1
    if score >= 2:
        return "High"
    if score == 1:
        return "Medium"
    return "Low"


def generate_tickets() -> pd.DataFrame:
    rows = []
    seen_texts: set[str] = set()

    for category, templates in TEMPLATES.items():
        generated = 0
        attempts = 0
        while generated < TICKETS_PER_CATEGORY and attempts < TICKETS_PER_CATEGORY * 10:
            attempts += 1
            template = random.choice(templates)
            days = random.randint(1, 14)
            text = template.format(
                order_id=_random_order_id(),
                product=random.choice(PRODUCT_VOCAB),
                days=days,
            )
            tone = _weighted_choice(SENTIMENT_WEIGHTS)
            text += random.choice(SUFFIXES[tone])
            text = text.strip()

            if text in seen_texts:
                continue
            seen_texts.add(text)
            generated += 1

            _, sentiment_score = analyze_sentiment(text)
            priority = determine_priority(category, sentiment_score, days)
            rows.append({"text": text, "category": category, "priority": priority})

    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)


FAQ_ENTRIES: dict[str, list[tuple[str, str]]] = {
    "Delivery": [
        (
            "My order hasn't arrived yet, what should I do?",
            "We're sorry for the delay. Please allow up to 2 extra business days for the courier's "
            "final-mile update. If it still doesn't arrive by then, we'll dispatch a free replacement "
            "or issue a full refund, whichever you prefer.",
        ),
        (
            "How can I track my order?",
            "You can track your order in real time from the 'My Orders' section of your account. If "
            "tracking hasn't updated in 48 hours, let us know and we'll chase it up with the courier directly.",
        ),
        (
            "The tracking says delivered but I never received my package.",
            "This can happen with a courier scanning error. We'll open an investigation with the courier "
            "and, if it isn't located within 48 hours, we'll send a replacement or refund at no cost to you.",
        ),
    ],
    "Refund": [
        (
            "How long does a refund take to process?",
            "Refunds are processed within 5-7 business days of us receiving the returned item, and the "
            "money typically appears on your statement 2-3 business days after that.",
        ),
        (
            "I returned an item but haven't received my refund.",
            "We're sorry for the wait. Please share your return tracking number so we can confirm receipt "
            "and expedite the refund; if it's already past 7 business days we'll issue it immediately.",
        ),
        (
            "Can I get a refund without returning the item?",
            "For damaged or defective items we can issue a refund without requiring a return in most cases "
            "- just share a photo of the issue and we'll process it right away.",
        ),
    ],
    "Payment": [
        (
            "I was charged twice for the same order.",
            "Sorry about that - this is usually a temporary authorization hold rather than a duplicate "
            "charge. If both charges are still showing after 3 business days, we'll refund the duplicate "
            "immediately.",
        ),
        (
            "My payment failed but the amount was deducted.",
            "That's typically a pending authorization from your bank that will automatically release within "
            "3-5 business days. If it doesn't, share your bank statement reference and we'll refund it directly.",
        ),
        (
            "I'm getting an error when trying to pay at checkout.",
            "Please try an alternate payment method or clearing your browser cache. If the error persists, "
            "send us a screenshot of the error and we'll escalate it to our payments team.",
        ),
    ],
    "Product Issue": [
        (
            "The product I received is damaged.",
            "We're sorry to hear that. Please share a photo of the damage and we'll send a free replacement "
            "right away, no return needed for clearly damaged items.",
        ),
        (
            "I received the wrong item.",
            "Apologies for the mix-up. We'll ship the correct item immediately and send a prepaid return "
            "label for the incorrect one - no charge to you either way.",
        ),
        (
            "The product stopped working after a few days.",
            "That's covered under our replacement guarantee. We'll send a free replacement, and if it "
            "happens again we'll offer a full refund instead.",
        ),
    ],
    "Account": [
        (
            "I can't log into my account.",
            "Please try the 'Forgot password' link on the sign-in page; reset emails can take a few minutes "
            "and may land in spam. If you still can't get in, we can manually verify your identity and reset "
            "access.",
        ),
        (
            "My account got locked after failed login attempts.",
            "For security, accounts lock temporarily after repeated failed attempts. It unlocks automatically "
            "after 30 minutes, or we can verify your identity and unlock it immediately.",
        ),
        (
            "How do I update my account details?",
            "You can update your address, email and phone number from Account Settings. If a field won't "
            "save, let us know which one and we'll update it for you directly.",
        ),
    ],
}


def generate_faq() -> pd.DataFrame:
    rows = [
        {"category": category, "question": question, "resolution": resolution}
        for category, entries in FAQ_ENTRIES.items()
        for question, resolution in entries
    ]
    return pd.DataFrame(rows)


def main() -> None:
    tickets_df = generate_tickets()
    tickets_df.to_csv(config.TICKETS_DATASET_PATH, index=False)
    print(f"Wrote {len(tickets_df)} tickets to {config.TICKETS_DATASET_PATH}")

    faq_df = generate_faq()
    faq_df.to_csv(config.FAQ_DATASET_PATH, index=False)
    print(f"Wrote {len(faq_df)} FAQ entries to {config.FAQ_DATASET_PATH}")


if __name__ == "__main__":
    main()
