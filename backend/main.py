from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import time
import os
import json
import hashlib
from openai import OpenAI

app = FastAPI(title="AI Book Concierge API", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRojoKX5x12MGB5PbwNE2qTErL_HjpDUOupVIkXQtRrLabnXx4O1FZKKjetkU6r8AfJQfhDanuWQ1qh/pub?output=csv"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
PENDING_LED_COMMAND = None


class RecommendRequest(BaseModel):
    query: str = ""
    max_price: Optional[float] = None
    genre: Optional[str] = None
    language: Optional[str] = None
    limit: int = 300


class ChatRequest(BaseModel):
    query: str
    language: Optional[str] = None
    limit: int = 300


class SetLedRequest(BaseModel):
    book_title: str
    color: str


class LedAckRequest(BaseModel):
    command_id: str


ALLOWED_COLORS = [
    {"key": "blue", "emoji": "🔵", "en": "Blue", "ru": "синий", "he": "כחול", "rgb": [0, 80, 255]},
    {"key": "purple", "emoji": "🟣", "en": "Purple", "ru": "фиолетовый", "he": "סגול", "rgb": [140, 0, 255]},
    {"key": "red", "emoji": "🔴", "en": "Red", "ru": "красный", "he": "אדום", "rgb": [255, 0, 0]},
    {"key": "orange", "emoji": "🟠", "en": "Orange", "ru": "оранжевый", "he": "כתום", "rgb": [255, 120, 0]},
    {"key": "yellow", "emoji": "🟡", "en": "Yellow", "ru": "жёлтый", "he": "צהוב", "rgb": [255, 220, 0]},
    {"key": "green", "emoji": "🟢", "en": "Green", "ru": "зелёный", "he": "ירוק", "rgb": [0, 255, 80]},
    {"key": "cyan", "emoji": "🔷", "en": "Cyan", "ru": "голубой", "he": "טורקיז", "rgb": [0, 220, 255]},
    {"key": "white", "emoji": "⚪", "en": "White", "ru": "белый", "he": "לבן", "rgb": [255, 255, 255]},
]


def detect_language(text: str) -> str:
    text = text or ""
    if any("\u0590" <= ch <= "\u05FF" for ch in text):
        return "he"
    if any("\u0400" <= ch <= "\u04FF" for ch in text):
        return "ru"
    return "en"


def stable_color_for_title(title: str):
    normalized = (title or "").strip().lower()
    digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()
    index = int(digest, 16) % len(ALLOWED_COLORS)
    return ALLOWED_COLORS[index]


def normalize_color_key(color: str) -> str:
    value = (color or "").strip().lower()
    mapping = {
        "blue": "blue", "синий": "blue", "כחול": "blue",
        "purple": "purple", "фиолетовый": "purple", "סגול": "purple",
        "red": "red", "красный": "red", "אדום": "red",
        "orange": "orange", "оранжевый": "orange", "כתום": "orange",
        "yellow": "yellow", "жёлтый": "yellow", "желтый": "yellow", "צהוב": "yellow",
        "green": "green", "зелёный": "green", "зеленый": "green", "ירוק": "green",
        "cyan": "cyan", "голубой": "cyan", "טורקיז": "cyan",
        "white": "white", "белый": "white", "לבן": "white",
    }
    return mapping.get(value, value)


def color_by_key(color_key: str):
    normalized = normalize_color_key(color_key)
    for color in ALLOWED_COLORS:
        if color["key"] == normalized:
            return color
    return ALLOWED_COLORS[0]


def parse_led_range(value: str):
    text = str(value or "").strip()
    if not text or "-" not in text:
        return None

    parts = text.split("-")
    try:
        start = int(parts[0].strip())
        end = int(parts[1].strip())
    except Exception:
        return None

    if start < 0 or end < start:
        return None

    return {
        "start": start,
        "end": end,
        "stop": end + 1
    }


def normalize_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df


def read_inventory():
    url = f"{SHEET_URL}&t={int(time.time())}"
    df = pd.read_csv(url)
    df = normalize_columns(df)

    required_columns = [
        "title",
        "author",
        "price",
        "currency",
        "category",
        "description",
        "in_stock",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in Google Sheet: {missing}")

    df["in_stock"] = (
        df["in_stock"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "yes", "1", "available", "in stock", "в наличии", "במלאי"])
    )

    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace("₪", "", regex=False)
        .str.replace("nis", "", case=False, regex=False)
        .str.replace("ils", "", case=False, regex=False)
        .str.replace(",", ".", regex=False)
        .str.extract(r"(\d+\.?\d*)")[0]
    )

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[df["in_stock"] == True]

    results = []

    for _, row in df.head(300).iterrows():
        price_value = row.get("price", 0)
        if pd.isna(price_value):
            price_value = 0

        title = str(row.get("title", "")).strip()
        color = stable_color_for_title(title)

        image_url = ""
        if "image_url" in df.columns:
            image_url = str(row.get("image_url", "") or "").strip()

        led_range = ""
        if "led_range" in df.columns:
            led_range = str(row.get("led_range", "") or "").strip()

        results.append({
            "title": title,
            "author": str(row.get("author", "")).strip(),
            "price": float(price_value),
            "currency": "NIS",
            "category": str(row.get("category", "")).strip(),
            "description": str(row.get("description", "")).strip(),
            "in_stock": True,
            "image_url": image_url,
            "led_range": led_range,
            "color_key": color["key"],
            "color_emoji": color["emoji"],
            "color_en": color["en"],
            "color_ru": color["ru"],
            "color_he": color["he"],
        })

    return results


def find_book_by_title(title: str):
    inventory = read_inventory()
    target = (title or "").strip().lower()

    for book in inventory:
        if book["title"].strip().lower() == target:
            return book

    for book in inventory:
        if target and target in book["title"].strip().lower():
            return book

    return None


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "AI Book Concierge API",
        "version": "1.2.0",
    }


@app.get("/debug")
def debug():
    try:
        inventory = read_inventory()
        return {
            "status": "ok",
            "service": "AI Book Concierge API",
            "total_available": len(inventory),
            "sample": inventory[:10],
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


@app.get("/debug-all")
def debug_all():
    try:
        inventory = read_inventory()
        return {
            "status": "ok",
            "service": "AI Book Concierge API",
            "total_available": len(inventory),
            "data": inventory,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


@app.post("/recommend")
def recommend_books(request: RecommendRequest):
    try:
        inventory = read_inventory()
        return {
            "results": inventory[:300],
            "total_available": len(inventory),
            "message": "Returned all available books.",
        }
    except Exception as e:
        return {
            "results": [],
            "total_available": 0,
            "message": f"Server error: {str(e)}",
        }


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        if client is None:
            return {
                "recommendations": [],
                "message": "OpenAI API key is not configured on the server.",
            }

        inventory = read_inventory()
        user_language = request.language or detect_language(request.query)

        system_instructions = """
You are AI Book Concierge, a smart bookstore assistant.

You receive a live inventory list from the backend.
Recommend books ONLY from this inventory.

Rules:
- Never invent books.
- Never use books outside the provided inventory.
- Choose the best 3-5 books unless the user asks for all.
- Respect genre, price, age, mood, author, and language.
- Always use NIS.
- Do not use book emojis.
- Use only the color data attached to each book.
- Return JSON only.
"""

        output_schema = """
Return JSON only:
{
  "recommendations": [
    {
      "title": "string",
      "author": "string",
      "price": 89,
      "currency": "NIS",
      "category": "string",
      "description": "short description in the user's language",
      "image_url": "string",
      "led_range": "30-40",
      "color_key": "blue",
      "color_emoji": "🔵",
      "color_label": "Blue",
      "display_line": "Title — Author — 89 NIS — 🔵 Blue",
      "reason": "short reason in user's language"
    }
  ],
  "message": "short message in user's language"
}
"""

        input_payload = {
            "user_request": request.query,
            "user_language": user_language,
            "inventory": inventory[:300],
            "format_rules": {
                "price": "Always write price as 89 NIS",
                "display_line": "Title — Author — Price NIS — color emoji + color name",
                "no_led_word": "Do not write the word LED",
                "no_book_icons": "Do not use book emojis",
            },
        }

        response = client.responses.create(
            model=OPENAI_MODEL,
            instructions=system_instructions,
            input=f"{json.dumps(input_payload, ensure_ascii=False)}\n\n{output_schema}",
        )

        raw_text = response.output_text.strip()

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            return {
                "recommendations": [],
                "message": "The AI response could not be parsed as JSON.",
                "raw_response": raw_text,
            }

    except Exception as e:
        return {
            "recommendations": [],
            "message": f"Server error: {str(e)}",
        }


@app.post("/set-led")
def set_led(request: SetLedRequest):
    global PENDING_LED_COMMAND

    try:
        book = find_book_by_title(request.book_title)

        if not book:
            return {
                "status": "error",
                "message": f"Book not found in inventory: {request.book_title}",
            }

        led_range = parse_led_range(book.get("led_range", ""))

        if not led_range:
            return {
                "status": "error",
                "message": f"No valid led_range found for book: {request.book_title}",
                "book_title": request.book_title,
                "led_range": book.get("led_range", ""),
            }

        selected_color = color_by_key(request.color)
        command_id = f"cmd_{int(time.time() * 1000)}"

        PENDING_LED_COMMAND = {
            "command_id": command_id,
            "book_title": book["title"],
            "color": selected_color["key"],
            "rgb": selected_color["rgb"],
            "led_range": book.get("led_range", ""),
            "start": led_range["start"],
            "end": led_range["end"],
            "stop": led_range["stop"],
            "duration_seconds": 10,
            "effect": "breathing",
            "created_at": int(time.time()),
        }

        return {
            "status": "ok",
            "mode": "bridge",
            "message": "LED range command queued for local bridge.",
            "command": PENDING_LED_COMMAND,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


@app.get("/led-command")
def get_led_command():
    if PENDING_LED_COMMAND is None:
        return {
            "has_command": False,
            "command": None,
        }

    return {
        "has_command": True,
        "command": PENDING_LED_COMMAND,
    }


@app.post("/led-command/ack")
def acknowledge_led_command(request: LedAckRequest):
    global PENDING_LED_COMMAND

    if PENDING_LED_COMMAND and PENDING_LED_COMMAND["command_id"] == request.command_id:
        PENDING_LED_COMMAND = None
        return {
            "status": "ok",
            "message": "LED command acknowledged and cleared.",
        }

    return {
        "status": "ignored",
        "message": "No matching command to acknowledge.",
    }
