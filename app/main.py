import json
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.services.knowledge_service import (
    find_relevant_information,
    load_knowledge_base,
)

app = FastAPI(title="ShopAssist AI", version="1.0.0")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"

BASE_DIR = Path(__file__).resolve().parent.parent
HTML_FILE = BASE_DIR / "app" / "static" / "index.html"


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    if not HTML_FILE.is_file():
        raise HTTPException(status_code=500, detail="Chat interface not found.")
    return FileResponse(HTML_FILE)


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/knowledge-base")
def knowledge_base_status():
    try:
        knowledge = load_knowledge_base()
        return {
            "status": "loaded",
            "product_count": len(knowledge.get("products", [])),
            "faq_count": len(knowledge.get("faqs", [])),
        }
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not load the product and FAQ knowledge base.",
        ) from exc


@app.post("/chat")
def chat(request: ChatRequest):
    question = request.message.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        knowledge = load_knowledge_base()
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not load the product and FAQ knowledge base.",
        ) from exc

    relevant_information = find_relevant_information(question, knowledge)
    context = json.dumps(relevant_information, ensure_ascii=False)

    prompt = (
        "You are ShopAssist AI, a customer support assistant for a demo store.\n"
        "Use the supplied knowledge-base context to answer store-specific questions.\n"
        "Only state prices, product availability, delivery rules, or policies "
        "when supported by the context.\n"
        "If the context does not contain the requested information, say you "
        "cannot confirm it from the available store information.\n"
        "Never invent order details or policies.\n\n"
        f"Knowledge-base context: {context}\n\n"
        f"Customer question: {question}\n\n"
        "Answer clearly and helpfully."
    )

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
            },
            timeout=180,
        )
        response.raise_for_status()
        result = response.json()

        return {
            "question": question,
            "answer": result["response"],
            "model": MODEL_NAME,
            "knowledge_matches": len(relevant_information),
        }

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not connect to the local AI model. Check Ollama.",
        ) from exc

@app.get("/products")
def get_products():
    try:
        knowledge = load_knowledge_base()
        return {"products": knowledge.get("products", [])}
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not load products."
        ) from exc
