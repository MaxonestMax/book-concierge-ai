from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import time
import os
import json
import hashlib
import threading
import uuid
import requests
from openai import OpenAI

app = FastAPI(
    title="AI Book Concierge API",
    version="1.0.0"
)

# ============================================================
# CONFIG
# ============================================================

# Set one of these in Render to the published Google Sheet CSV URL.
SHEET_URL = (
    os.environ.get("SHEET_URL")
    or os.environ.get("GOOGLE_SHEET_CSV_URL")
    or os.environ.get("GOOGLE_SHEET_URL")
    or "https://docs.google.com/spreadsheets/d/e/2PACX-1vRojoKX5x12MGB5PbwNE2qTErL_HjpDUOupVIkXQtRrLabnXx4O1FZKKjetkU6r8AfJQfhDanuWQ1qh/pub?output=csv"
)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.2")

# Optional. Потом можно добавить в Render Environment:
# WLED_URL = http://192.168.1.50/json/state
WLED_URL = os.environ.get("WLED_URL", "")
LED_COUNT = int(os.environ.get("LED_COUNT", "288"))

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

LED_COMMAND_QUEUE: List[Dict[str, Any]] = []
LED_QUEUE_LOCK = threading.Lock()
DEFAULT_LED_DURATION_SECONDS = 10.0
DEFAULT_LED_RANGE = [0, 4]


# ============================================================
# MODELS
# ============================================================

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
    start: Optional[int] = None
    stop: Optional[int] = None
    led_range: Optional[Any] = None
    duration_seconds: Optional[float] = None


# ============================================================
# COLOR LOGIC
# ============================================================

ALLOWED_COLORS = [
    {
        "key": "blue",
        "emoji": "🔵",
        "en": "Blue",
        "ru": "синий",
        "he": "כחול",
        "rgb": [0, 80, 255],
    },
    {
        "key": "purple",
        "emoji": "🟣",
        "en": "Purple",
        "ru": "фиолетовый",
        "he": "סגול",
        "rgb": [140, 0, 255],
    },
    {
        "key": "red",
        "emoji": "🔴",
        "en": "Red",
        "ru": "красный",
        "he": "אדום",
        "rgb": [255, 0, 0],
    },
    {
        "key": "orange",
        "emoji": "🟠",
        "en": "Orange",
        "ru": "оранжевый",
        "he": "כתום",
        "rgb": [255, 120, 0],
    },
    {
        "key": "yellow",
        "emoji": "🟡",
        "en": "Yellow",
        "ru": "жёлтый",
        "he": "צהוב",
        "rgb": [255, 220, 0],
    },
    {
        "key": "green",
        "emoji": "🟢",
        "en": "Green",
        "ru": "зелёный",
        "he": "ירוק",
        "rgb": [0, 255, 80],
    },
    {
        "key": "cyan",
        "emoji": "🔷",
        "en": "Cyan",
        "ru": "голубой",
        "he": "טורקיז",
        "rgb": [0, 220, 255],
    },
    {
        "key": "white",
        "emoji": "⚪",
        "en": "White",
        "ru": "белый",
        "he": "לבן",
        "rgb": [255, 255, 255],
    },
]


def detect_language(text: str) -> str:
    text = text or ""

    hebrew_chars = sum(1 for ch in text if "\u0590" <= ch <= "\u05FF")
    cyrillic_chars = sum(1 for ch in text if "\u0400" <= ch <= "\u04FF")

    if hebrew_chars > 0:
        return "he"

    if cyrillic_chars > 0:
        return "ru"

    return "en"


def stable_color_for_title(title: str) -> Dict[str, Any]:
    normalized = (title or "").strip().lower()
    digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()
    index = int(digest, 16) % len(ALLOWED_COLORS)
    return ALLOWED_COLORS[index]


def normalize_color_key(color: str) -> str:
    value = (color or "").strip().lower()

    mapping = {
        "blue": "blue",
        "синий": "blue",
        "כחול": "blue",

        "purple": "purple",
        "фиолетовый": "purple",
        "סגול": "purple",

        "red": "red",
        "красный": "red",
        "אדום": "red",

        "orange": "orange",
        "оранжевый": "orange",
        "כתום": "orange",

        "yellow": "yellow",
        "жёлтый": "yellow",
        "желтый": "yellow",
        "צהוב": "yellow",

        "green": "green",
        "зелёный": "green",
        "зеленый": "green",
        "ירוק": "green",

        "cyan": "cyan",
        "голубой": "cyan",
        "טורקיז": "cyan",

        "white": "white",
        "белый": "white",
        "לבן": "white",
    }

    return mapping.get(value, value)


def color_by_key(color_key: str) -> Dict[str, Any]:
    normalized = normalize_color_key(color_key)

    for color in ALLOWED_COLORS:
        if color["key"] == normalized:
            return color

    return ALLOWED_COLORS[0]


def normalize_title(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def parse_led_range(value: Any) -> Optional[List[int]]:
    if value is None:
        return None

    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return [int(value[0]), int(value[1])]

    if isinstance(value, dict):
        if "start" in value and "stop" in value:
            return [int(value["start"]), int(value["stop"])]

        if "from" in value and "to" in value:
            return [int(value["from"]), int(value["to"])]

    text = str(value).strip()

    if not text or text.lower() in ["nan", "none", "null"]:
        return None

    normalized = (
        text
        .replace("..", "-")
        .replace(":", "-")
        .replace("—", "-")
        .replace("–", "-")
    )
    parts = [part.strip() for part in normalized.split("-") if part.strip()]

    if len(parts) >= 2:
        return [int(float(parts[0])), int(float(parts[1]))]

    try:
        point = int(float(text))
        return [point, point + 1]
    except ValueError:
        pass

    return None


def clamp_led_range(led_range: List[int]) -> List[int]:
    start = max(0, min(LED_COUNT, int(led_range[0])))
    stop = max(0, min(LED_COUNT, int(led_range[1])))

    if stop <= start:
        stop = min(LED_COUNT, start + 1)

    return [start, stop]


def safe_sheet_url_preview() -> str:
    if "XXXX" in SHEET_URL:
        return SHEET_URL

    return SHEET_URL[:80] + ("..." if len(SHEET_URL) > 80 else "")


# ============================================================
# INVENTORY
# ============================================================

def read_inventory() -> List[Dict[str, Any]]:
    url = f"{SHEET_URL}&t={int(time.time())}"
    df = pd.read_csv(url)

    df.columns = df.columns.str.strip().str.lower()

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

        led_range = None

        for range_column in ["led_range", "led range", "ledrange", "range"]:
            if range_column in df.columns:
                led_range = parse_led_range(row.get(range_column))
                break

        if led_range is None:
            start_column = next(
                (col for col in ["led_start", "start", "led from", "led_from"] if col in df.columns),
                None,
            )
            stop_column = next(
                (col for col in ["led_stop", "stop", "led to", "led_to"] if col in df.columns),
                None,
            )

            if start_column and stop_column:
                led_range = parse_led_range([row.get(start_column), row.get(stop_column)])

        if led_range is not None:
            led_range = clamp_led_range(led_range)

        results.append({
            "title": title,
            "author": str(row.get("author", "")).strip(),
            "price": float(price_value),
            "currency": "NIS",
            "category": str(row.get("category", "")).strip(),
            "description": str(row.get("description", "")).strip(),
            "in_stock": True,
            "image_url": image_url,
            "color_key": color["key"],
            "color_emoji": color["emoji"],
            "color_en": color["en"],
            "color_ru": color["ru"],
            "color_he": color["he"],
            "led_range": led_range,
        })

    return results


def find_led_range_for_book(book_title: str) -> Optional[List[int]]:
    requested_title = normalize_title(book_title)

    try:
        inventory = read_inventory()

        for item in inventory:
            if normalize_title(item.get("title", "")) == requested_title:
                led_range = item.get("led_range")

                if led_range:
                    return clamp_led_range(led_range)

                break

        for item in inventory:
            inventory_title = normalize_title(item.get("title", ""))

            if requested_title and (
                requested_title in inventory_title
                or inventory_title in requested_title
            ):
                led_range = item.get("led_range")

                if led_range:
                    return clamp_led_range(led_range)

    except Exception as e:
        print(f"Could not read led_range from inventory: {e}")

    return None


def build_led_command(request: SetLedRequest) -> Dict[str, Any]:
    selected_color = color_by_key(request.color)
    led_range = find_led_range_for_book(request.book_title)
    led_range_source = "sheet"

    if led_range is None:
        if request.start is not None and request.stop is not None:
            led_range = clamp_led_range([request.start, request.stop])
            led_range_source = "request_start_stop"
        else:
            led_range = parse_led_range(request.led_range)

            if led_range is not None:
                led_range = clamp_led_range(led_range)
                led_range_source = "request_led_range"

    if led_range is None:
        print(f"LED range missing for '{request.book_title}', using fallback {DEFAULT_LED_RANGE}")
        led_range = clamp_led_range(DEFAULT_LED_RANGE)
        led_range_source = "fallback"

    return {
        "command_id": f"led_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
        "book_title": request.book_title,
        "color": selected_color["key"],
        "rgb": selected_color["rgb"],
        "start": led_range[0],
        "stop": led_range[1],
        "led_range": led_range,
        "led_range_source": led_range_source,
        "duration_seconds": request.duration_seconds or DEFAULT_LED_DURATION_SECONDS,
        "created_at": time.time(),
    }


def enqueue_led_command(command: Dict[str, Any]) -> None:
    with LED_QUEUE_LOCK:
        LED_COMMAND_QUEUE.append(command)

        if len(LED_COMMAND_QUEUE) > 20:
            del LED_COMMAND_QUEUE[0:len(LED_COMMAND_QUEUE) - 20]


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "AI Book Concierge API",
        "version": "1.0.0",
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


@app.get("/debug-sheet")
def debug_sheet():
    try:
        url = f"{SHEET_URL}&t={int(time.time())}"
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip().str.lower()

        preview_columns = ["title", "led_range"]
        available_preview_columns = [
            column for column in preview_columns if column in df.columns
        ]

        preview = []
        if available_preview_columns:
            preview = (
                df[available_preview_columns]
                .head(10)
                .fillna("")
                .to_dict(orient="records")
            )

        return {
            "status": "ok",
            "sheet_url_preview": safe_sheet_url_preview(),
            "columns": list(df.columns),
            "row_count": int(len(df)),
            "preview": preview,
        }

    except Exception as e:
        return {
            "status": "error",
            "sheet_url_preview": safe_sheet_url_preview(),
            "message": str(e),
        }


@app.get("/debug-led-range")
def debug_led_range(book_title: str):
    led_range = find_led_range_for_book(book_title)

    return {
        "status": "ok",
        "book_title": book_title,
        "led_range": led_range,
        "fallback": clamp_led_range(DEFAULT_LED_RANGE),
        "source": "sheet" if led_range else "missing",
    }


@app.get("/led-commands")
def get_led_commands():
    with LED_QUEUE_LOCK:
        commands = list(LED_COMMAND_QUEUE)

    return {
        "status": "ok",
        "commands": commands,
        "count": len(commands),
    }


@app.get("/led-command")
def get_led_command():
    with LED_QUEUE_LOCK:
        command = LED_COMMAND_QUEUE[0] if LED_COMMAND_QUEUE else None

    return {
        "status": "ok",
        "has_command": command is not None,
        "command": command,
    }


@app.post("/led-command/ack")
def acknowledge_led_command(payload: Dict[str, Any]):
    command_id = payload.get("command_id")

    with LED_QUEUE_LOCK:
        before = len(LED_COMMAND_QUEUE)
        LED_COMMAND_QUEUE[:] = [
            command
            for command in LED_COMMAND_QUEUE
            if command.get("command_id") != command_id
        ]
        removed = before - len(LED_COMMAND_QUEUE)

    return {
        "status": "ok",
        "command_id": command_id,
        "removed": removed,
        "remaining": len(LED_COMMAND_QUEUE),
    }


@app.post("/recommend")
def recommend_books(request: RecommendRequest):
    try:
        inventory = read_inventory()

        return {
            "results": inventory[:300],
            "total_available": len(inventory),
            "message": "Returned all available books. Client or AI should filter by genre, price, mood, and user request.",
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
You must recommend books ONLY from this inventory.

Rules:
- Never invent books.
- Never use books outside the provided inventory.
- Never recommend out-of-stock books.
- Choose the best 3-5 books unless the user explicitly asks for all.
- Respect user request: genre, price, age, mood, author, language.
- Always use NIS as currency.
- Do not use book emojis.
- Use only the color data already attached to each book.
- Do not use pink, brown, gray, black, beige, or dark colors.
- Keep the answer language the same as the user language.
- Return JSON only. No markdown. No comments.
"""

        output_schema = """
Return JSON only in this exact structure:
{
  "recommendations": [
    {
      "title": "string",
      "author": "string",
      "price": 89,
      "currency": "NIS",
      "category": "string",
      "description": "short description in the user's language, 1-2 sentences",
      "image_url": "string",
      "color_key": "blue",
      "color_emoji": "🔵",
      "color_label": "Blue",
      "display_line": "Title — Author — 89 NIS — 🔵 Blue",
      "reason": "short reason in the user's language"
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
                "colors": {
                    "en": {
                        "blue": "Blue",
                        "purple": "Purple",
                        "red": "Red",
                        "orange": "Orange",
                        "yellow": "Yellow",
                        "green": "Green",
                        "cyan": "Cyan",
                        "white": "White"
                    },
                    "ru": {
                        "blue": "синий",
                        "purple": "фиолетовый",
                        "red": "красный",
                        "orange": "оранжевый",
                        "yellow": "жёлтый",
                        "green": "зелёный",
                        "cyan": "голубой",
                        "white": "белый"
                    },
                    "he": {
                        "blue": "כחול",
                        "purple": "סגול",
                        "red": "אדום",
                        "orange": "כתום",
                        "yellow": "צהוב",
                        "green": "ירוק",
                        "cyan": "טורקיז",
                        "white": "לבן"
                    }
                }
            }
        }

        response = client.responses.create(
            model=OPENAI_MODEL,
            instructions=system_instructions,
            input=f"{json.dumps(input_payload, ensure_ascii=False)}\n\n{output_schema}",
        )

        raw_text = response.output_text.strip()

        try:
            parsed = json.loads(raw_text)
            return parsed

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
    try:
        command = build_led_command(request)
        enqueue_led_command(command)

        return {
            "status": "ok",
            "mode": "queue",
            "message": "LED command queued for bridge.",
            "command": command,
            "queue_size": len(LED_COMMAND_QUEUE),
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }
