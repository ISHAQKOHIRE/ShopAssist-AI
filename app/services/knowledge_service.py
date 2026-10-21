import json
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
KNOWLEDGE_FILE = BASE_DIR / "data" / "knowledge_base.json"

STOP_WORDS = {
    "what", "is", "the", "of", "a", "an", "my", "your",
    "i", "you", "do", "does", "how", "can", "me", "tell",
    "please", "about", "for", "to", "and", "in", "on",
    "are", "with", "it", "we", "our", "this", "that"
}


def load_knowledge_base():
    with KNOWLEDGE_FILE.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def tokenize(text):
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    return {word for word in words if word not in STOP_WORDS and len(word) > 1}


def find_relevant_information(question, knowledge=None):
    if knowledge is None:
        knowledge = load_knowledge_base()

    question_lower = question.lower()
    question_words = tokenize(question)
    matches = []

    for product in knowledge.get("products", []):
        name = product.get("name", "")
        description = product.get("description", "")
        product_id = product.get("id", "")

        name_words = tokenize(name)
        description_words = tokenize(description)
        id_words = tokenize(product_id)

        overlap = question_words.intersection(
            name_words | description_words | id_words
        )

        score = len(overlap)

        if name.lower() in question_lower:
            score += 10

        if product_id.lower() in question_lower:
            score += 10

        if score > 0:
            matches.append({
                "type": "product",
                "relevance": score,
                "data": product,
            })

    for faq in knowledge.get("faqs", []):
        faq_question = faq.get("question", "")
        faq_answer = faq.get("answer", "")
        faq_words = tokenize(faq_question) | tokenize(faq_answer)

        overlap = question_words.intersection(faq_words)
        score = len(overlap)

        if question_words and score > 0:
            matches.append({
                "type": "faq",
                "relevance": score,
                "data": faq,
            })

    matches.sort(
        key=lambda item: (item["relevance"], item["type"] == "product"),
        reverse=True,
    )

    if not matches:
        return []

    best_score = matches[0]["relevance"]

    return [
        item for item in matches
        if item["relevance"] == best_score
    ][:3]
