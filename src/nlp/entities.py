import re

PRODUCT_VOCAB = [
    "wireless earbuds",
    "bluetooth speaker",
    "smartwatch",
    "laptop stand",
    "phone case",
    "running shoes",
    "office chair",
    "desk lamp",
    "backpack",
    "coffee maker",
    "air fryer",
    "yoga mat",
    "gaming mouse",
    "mechanical keyboard",
    "power bank",
    "hdmi cable",
    "wall charger",
    "sunglasses",
    "electric kettle",
    "water bottle",
]

_INTENT_KEYWORDS: dict[str, list[str]] = {
    "delivery_delay": ["not delivered", "delayed", "delivery", "shipment", "tracking", "hasn't arrived", "not arrived"],
    "refund_request": ["refund", "money back", "return"],
    "payment_issue": ["charged twice", "payment failed", "double charged", "deducted", "checkout", "payment"],
    "product_defect": ["damaged", "defective", "broken", "stopped working", "wrong item"],
    "account_access": ["log in", "login", "locked", "password", "account"],
}

_ORDER_ID_PATTERN = re.compile(r"#\s?([A-Za-z0-9-]{4,12})")


def extract_order_id(text: str) -> str | None:
    match = _ORDER_ID_PATTERN.search(text)
    return match.group(1) if match else None


def extract_product(text: str) -> str | None:
    lowered = text.lower()
    for product in PRODUCT_VOCAB:
        if product in lowered:
            return product
    return None


def extract_entities(text: str) -> dict[str, str]:
    entities: dict[str, str] = {}
    order_id = extract_order_id(text)
    if order_id:
        entities["order_id"] = order_id
    product = extract_product(text)
    if product:
        entities["product"] = product
    return entities


def extract_intent(text: str) -> str:
    lowered = text.lower()
    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return intent
    return "general_inquiry"
