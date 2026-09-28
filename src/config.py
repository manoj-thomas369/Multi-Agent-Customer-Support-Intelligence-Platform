from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
VECTOR_INDEX_DIR = BASE_DIR / "vector_index"
DB_PATH = BASE_DIR / "support.db"

TICKETS_DATASET_PATH = DATA_DIR / "tickets_dataset.csv"
FAQ_DATASET_PATH = DATA_DIR / "faq_knowledge_base.csv"

CATEGORY_MODEL_PATH = MODELS_DIR / "category_classifier.joblib"
PRIORITY_MODEL_PATH = MODELS_DIR / "priority_classifier.joblib"
TRAINING_REPORT_PATH = MODELS_DIR / "training_report.json"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = VECTOR_INDEX_DIR / "knowledge_base.index"
FAISS_METADATA_PATH = VECTOR_INDEX_DIR / "knowledge_base_meta.json"
LEARNED_CORPUS_PATH = VECTOR_INDEX_DIR / "learned_resolutions.jsonl"

CATEGORIES = ["Delivery", "Refund", "Payment", "Product Issue", "Account"]
PRIORITIES = ["Low", "Medium", "High"]

# Below this predicted-category confidence, a ticket is routed to a human.
ESCALATION_CONFIDENCE_THRESHOLD = 0.55
# VADER compound score at/below this is treated as strongly negative sentiment.
NEGATIVE_SENTIMENT_THRESHOLD = -0.4
# Below this similarity score, retrieved knowledge is considered a weak match.
RETRIEVAL_RELEVANCE_THRESHOLD = 0.35

RETRIEVAL_TOP_K = 3
# Confident, non-escalated resolutions at/above this bar get folded back into
# the retrieval corpus so future similar tickets benefit from them.
LEARNING_CONFIDENCE_THRESHOLD = 0.75

for _dir in (DATA_DIR, MODELS_DIR, VECTOR_INDEX_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
