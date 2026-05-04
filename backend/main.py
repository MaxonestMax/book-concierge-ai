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

app = FastAPI(title="AI Book Concierge API", version="1.4.0")

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

PENDING_LED_COMMANDS = []


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
    {"key": "blue",   "emoji": "🔵", "en": "Blue",   "ru": "синий",       "he": "כחול",   "rgb": [0, 0, 255]},
    {"key": "purple", "emoji": "🟣", "en": "Purple", "ru": "фиолетовый",  "he": "סגול",   "rgb": [180, 0, 255]},
    {"key": "red",    "emoji": "🔴", "en": "Red",    "ru": "красный",     "he": "אדום",   "rgb": [255, 0, 0]},
    {"key": "yellow", "emoji": "🟡", "en": "Yellow", "ru": "жёлтый",      "he": "צהוב",   "rgb": [255, 200, 0]},
    {"key": "green",  "emoji": "🟢", "en": "Green",  "ru": "зелёный",     "he": "ירוק",   "rgb": [0, 255, 0]},
    {"key": "cyan",   "emoji": "🔷", "en": "Cyan",   "ru": "голубой",     "he": "טורקיז", "rgb": [0, 255, 255]},
    {"key": "white",  "emoji": "⚪", "en": "White",  "ru": "белый",       "he": "לבן",   "rgb": [255, 255, 255]},
]


def normalize_bool(value) -> bool:
    text = str(value or "").strip().lower()
    return text in ["true", "yes", "1", "available", "in stock", "в наличии", "במלאי"]


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
        "stop": end + 1,
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

    df["in_stock"] = df["in_stock"].apply(normalize_bool)

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

        sale = False
        if "sale" in df.columns:
            sale = normalize_bool(row.get("sale", ""))

        more_info = ""
        if "more_info" in df.columns:
            more_info = str(row.get("more_info", "") or "").strip()

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
            "sale": sale,
            "more_info": more_info,
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
        "version": "1.4.0",
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
            "sale_count": len([book for book in inventory if book.get("sale") is True]),
            "pending_led_commands": len(PENDING_LED_COMMANDS),
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
            "sale_count": len([book for book in inventory if book.get("sale") is True]),
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
            "sale_count": len([book for book in inventory if book.get("sale") is True]),
            "message": "Returned all available books.",
        }

    except Exception as e:
        return {
            "results": [],
            "total_available": 0,
            "sale_count": 0,
            "message": f"Server error: {str(e)}",
        }


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        if client is None:
            return {
                "recommendations": [],
                "sale_picks": [],
                "message": "OpenAI API key is not configured on the server.",
            }

        inventory = read_inventory()
        user_language = request.language or detect_language(request.query)

        sale_inventory = [
            book for book in inventory
            if book.get("sale") is True and book.get("in_stock") is True
        ]

        system_instructions = """
You are AI Book Concierge, a smart bookstore assistant.

You receive a live inventory list from the backend.
Recommend books ONLY from this inventory.

Rules:
- Never invent books.
- Never use books outside the provided inventory.
- Choose the best 3-5 recommendations unless the user asks for all.
- Respect genre, price, age, mood, author, and language.
- Always use NIS.
- Do not use book emojis.
- Use only the color data attached to each book.
- Return JSON only.
- Preserve sale, more_info, image_url, led_range, and color_key fields exactly from inventory.
- sale_picks must contain ONLY books where sale is true and in_stock is true.
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
      "more_info": "translated detailed info in user's language",
      "image_url": "string",
      "led_range": "30-40",
      "sale": true,
      "color_key": "blue",
      "color_emoji": "🔵",
      "color_label": "Blue",
      "display_line": "Title — Author — 89 NIS — 🔵 Blue",
      "reason": "short reason in user's language"
    }
  ],
  "sale_picks": [
    {
      "title": "string",
      "author": "string",
      "price": 89,
      "currency": "NIS",
      "category": "string",
      "description": "short description in the user's language",
      "more_info": "translated detailed info in user's language",
      "image_url": "string",
      "led_range": "30-40",
      "sale": true,
      "color_key": "blue",
      "color_emoji": "🔵",
      "color_label": "Blue",
      "display_line": "Title — Author — 89 NIS — 🔵 Blue",
      "reason": "short sale reason in user's language"
    }
  ],
  "message": "short message in user's language"
}
"""

        input_payload = {
            "user_request": request.query,
            "user_language": user_language,
            "inventory": inventory[:300],
            "sale_inventory": sale_inventory[:300],
            "format_rules": {
                "price": "Always write price as 89 NIS",
                "display_line": "Title — Author — Price NIS — color emoji + color name",
                "no_led_word": "Do not write the word LED",
                "no_book_icons": "Do not use book emojis",
                "sale_rules": [
                    "sale_picks must use only books where sale is true and in_stock is true",
                    "If there are 3 or more sale books, return exactly 3 sale_picks",
                    "If there are fewer than 3 sale books, return all available sale books",
                    "Do not mark non-sale books as sale"
                ]
            },
        }

        response = client.responses.create(
            model=OPENAI_MODEL,
            instructions=system_instructions,
            input=f"{json.dumps(input_payload, ensure_ascii=False)}\n\n{output_schema}",
        )

        raw_text = response.output_text.strip()

        try:
            parsed = json.loads(raw_text)

            if "recommendations" not in parsed:
                parsed["recommendations"] = []

            if "sale_picks" not in parsed:
                parsed["sale_picks"] = []

            parsed["sale_picks"] = [
                book for book in parsed["sale_picks"]
                if book.get("sale") is True
            ]

            return parsed

        except json.JSONDecodeError:
            return {
                "recommendations": [],
                "sale_picks": [],
                "message": "The AI response could not be parsed as JSON.",
                "raw_response": raw_text,
            }

    except Exception as e:
        return {
            "recommendations": [],
            "sale_picks": [],
            "message": f"Server error: {str(e)}",
        }


@app.post("/set-led")
def set_led(request: SetLedRequest):
    global PENDING_LED_COMMANDS

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

        command = {
            "command_id": command_id,
            "book_title": book["title"],
            "color": selected_color["key"],
            "rgb": selected_color["rgb"],
            "led_range": book.get("led_range", ""),
            "start": led_range["start"],
            "end": led_range["end"],
            "stop": led_range["stop"],
            "duration_seconds": 10,
            "effect": "layered_travel_breathing",
            "created_at": int(time.time()),
        }

        PENDING_LED_COMMANDS.append(command)
        PENDING_LED_COMMANDS = PENDING_LED_COMMANDS[-20:]

        return {
            "status": "ok",
            "mode": "bridge_queue",
            "message": "LED command added to queue for local bridge.",
            "command": command,
            "pending_count": len(PENDING_LED_COMMANDS),
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


@app.get("/led-command")
def get_led_command():
    if not PENDING_LED_COMMANDS:
        return {
            "has_command": False,
            "command": None,
        }

    return {
        "has_command": True,
        "command": PENDING_LED_COMMANDS[0],
    }


@app.get("/led-commands")
def get_led_commands():
    return {
        "has_commands": len(PENDING_LED_COMMANDS) > 0,
        "commands": PENDING_LED_COMMANDS,
        "count": len(PENDING_LED_COMMANDS),
    }


@app.post("/led-command/ack")
def acknowledge_led_command(request: LedAckRequest):
    global PENDING_LED_COMMANDS

    before = len(PENDING_LED_COMMANDS)

    PENDING_LED_COMMANDS = [
        command for command in PENDING_LED_COMMANDS
        if command["command_id"] != request.command_id
    ]

    after = len(PENDING_LED_COMMANDS)

    if before != after:
        return {
            "status": "ok",
            "message": "LED command acknowledged and removed.",
            "pending_count": after,
        }

    return {
        "status": "ignored",
        "message": "No matching command to acknowledge.",
        "pending_count": after,
    }
