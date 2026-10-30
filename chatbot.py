import json
import random
import re
from pathlib import Path

from nltk.stem import PorterStemmer
from nltk.tokenize import wordpunct_tokenize

BASE_DIR = Path(__file__).resolve().parent
with (BASE_DIR / "intents.json").open(encoding="utf-8") as file:
    INTENTS = json.load(file)["intents"]

stemmer = PorterStemmer()
STOP_WORDS = {
    "a", "an", "the", "is", "are", "am", "was", "were", "be", "been",
    "to", "of", "in", "on", "at", "for", "and", "or", "but", "with",
    "i", "you", "he", "she", "it", "we", "they", "me", "my", "your",
    "this", "that", "what", "how", "do", "does", "can", "could", "would"
}

def preprocess(text):
    """Lowercase, tokenize, remove stop words and stem tokens."""
    text = text.lower()
    tokens = wordpunct_tokenize(text)
    cleaned = []
    for token in tokens:
        if re.fullmatch(r"[a-z0-9']+", token) and token not in STOP_WORDS:
            cleaned.append(stemmer.stem(token))
    return cleaned

def similarity(user_tokens, pattern_tokens):
    """Simple token-overlap score used for intent matching."""
    if not user_tokens or not pattern_tokens:
        return 0.0
    user_set = set(user_tokens)
    pattern_set = set(pattern_tokens)
    overlap = len(user_set & pattern_set)
    return overlap / len(pattern_set)

def get_response(message):
    user_tokens = preprocess(message)
    best_intent = None
    best_score = 0.0

    for intent in INTENTS:
        for pattern in intent["patterns"]:
            pattern_tokens = preprocess(pattern)
            score = similarity(user_tokens, pattern_tokens)
            if score > best_score:
                best_score = score
                best_intent = intent

    # A threshold avoids confidently answering unrelated questions.
    if best_intent is not None and best_score >= 0.35:
        return random.choice(best_intent["responses"])

    fallback = next(
        (item for item in INTENTS if item["tag"] == "fallback"),
        None
    )
    if fallback:
        return random.choice(fallback["responses"])
    return "I'm not sure I understood. Could you rephrase that?"
