"""
Prompt Engineer AI (UZ/RU) — profiles + cascade + native streaming
"""

import os
import sys
import re
import json
import time
import yaml
import requests
from dataclasses import dataclass
from typing import Literal, Dict, Optional

from langdetect import detect
from langchain_ollama import OllamaLLM

SUPPORTED_MODES = ("coding", "image", "video", "chatgpt")

# ---------------------------
# Config
# ---------------------------
@dataclass
class RefinerConfig:
    model: str = "gpt-oss:20b"
    temperature: float = 0.3
    top_p: float = 0.9
    num_ctx: int = 2048  # smaller default = faster
    max_bullets: int = 10
    base_url: Optional[str] = None  # OLLAMA_HOST
    keep_alive: str = "30m"         # keep model loaded (prewarm)

def load_config(path: str = "config.yaml") -> RefinerConfig:
    data = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    oll = data.get("ollama", {})
    ref = data.get("refiner", {})
    return RefinerConfig(
        model=str(oll.get("model", "gpt-oss:20b")),
        temperature=float(oll.get("temperature", 0.3)),
        top_p=float(oll.get("top_p", 0.9)),
        num_ctx=int(oll.get("num_ctx", 2048)),
        max_bullets=int(ref.get("max_bullets", 10)),
        base_url=os.getenv("OLLAMA_HOST") or None,
        keep_alive=str(oll.get("keep_alive", "30m")),
    )

# ---------------------------
# Helpers
# ---------------------------
def detect_lang(text: str) -> Literal["uz", "ru", "en"]:
    try:
        code = detect(text)
        if code.startswith("uz"):
            return "uz"
        if code.startswith("ru"):
            return "ru"
        return "en"
    except Exception:
        return "uz"

def pick_system(lang: str, systems: Dict[str, str]) -> str:
    return systems.get(lang, systems["uz"])

def build_instruction(mode: str, modes_spec: Dict) -> str:
    mode_spec = modes_spec.get(mode, modes_spec["chatgpt"])
    sections = mode_spec.get("sections", [])
    bullets = "\n".join([f"- {s}" for s in sections])
    return f"Bo'limlar / Разделы:\n{bullets}"

def score_output(mode: str, text: str) -> float:
    """Simple quality heuristic: required headers present + minimum length."""
    checks = {
        "image": [r"Maqsad", r"Kompozitsiya|Композиция", r"Mavzu|Детали", r"Uslub|Стиль", r"Kamera|Камера", r"Chiqish|Параметры", r"Nimalar|Исключить", r"Tekshirish|Уточняющий"],
        "video": [r"Maqsad|Цель", r"Syujet|Сцены", r"Kadr|Шот", r"Ovoz|Озвучка", r"Chiqish|Параметры", r"Tekshirish|Уточняющий"],
        "coding": [r"Maqsad|Цель", r"Kirish|Вход", r"Cheklov|Огранич", r"Chiqish|Формат", r"Qadam|Шаг", r"Test|Тест", r"Tekshirish|Уточняющий"],
        "chatgpt": [r"Maqsad|Цель", r"Rollar|Роли", r"Qadam|Шаг", r"Misol|Пример", r"Cheklov|Огранич", r"Tekshirish|Уточняющий"],
    }
    needed = checks.get(mode, [])
    hits = sum(1 for pat in needed if re.search(pat, text, re.I))
    length_ok = len(text.strip()) > 600  # ~600 chars ≈ decent structure
    base = hits / max(len(needed), 1)
    return base * (1.0 if length_ok else 0.8)

def preload_keepalive(base_url: str, model: str, keep_alive: str) -> None:
    """Preload and keep model in RAM using Ollama REST (no stream)."""
    try:
        url = (base_url or "http://localhost:11434").rstrip("/") + "/api/generate"
        requests.post(url, json={"model": model, "prompt": " ", "stream": False, "keep_alive": keep_alive}, timeout=2)
    except Exception:
        pass  # best-effort

def make_llm(model: str, base_url: Optional[str], temperature: float, top_p: float, num_ctx: int) -> OllamaLLM:
    return OllamaLLM(model=model, base_url=base_url, temperature=temperature, top_p=top_p, num_ctx=num_ctx)

def stream_text(llm: OllamaLLM, prompt: str) -> str:
    pieces = []
    for chunk in llm.stream(prompt):
        t = getattr(chunk, "text", str(chunk))
        pieces.append(t)
        sys.stdout.write(t)
        sys.stdout.flush()
    return "".join(pieces)

# ---------------------------
# Core
# ---------------------------
class PromptRefiner:
    def __init__(self, cfg: RefinerConfig, system_prompts: Dict[str, str], modes_spec: Dict):
        self.cfg = cfg
        self.system_prompts = system_prompts
        self.modes_spec = modes_spec

    def _make_prompt(self, raw_prompt: str, mode: str) -> str:
        lang = detect_lang(raw_prompt)
        system = pick_system(lang, self.system_prompts)
        instruction = build_instruction(mode, self.modes_spec)

        examples = {
            "uz": {
                "input": "rasm chiz: qizil mashina",
                "output": (
                    "Maqsad: qizil sport avtomobilining realistik 3/4 rakursdagi rasmi.\n"
                    "Kompozitsiya: bitta avtomobil, fon minimal, yo'l sirtida.\n"
                    "Mavzu detali: kupe, porloq qizil, qora disklar.\n"
                    "Uslub & Yoritish: kunduzgi diffuz yorug', yumshoq soyalar.\n"
                    "Kamera: 35mm, past rakurs.\n"
                    "Chiqish parametrlari: 1024x1024.\n"
                    "Nimalar kiritilmasin: odamlar, logotiplar.\n"
                    "Tekshirish savoli: Fon rangini xohlaysizmi? (oq/ko'k/asfalt)"
                ),
            },
            "ru": {
                "input": "сделай промпт для телеграм-бота",
                "output": (
                    "Цель: готовый промпт для чат-бота в Telegram.\n"
                    "Роли: бот/пользователь/система.\n"
                    "Шаги: классификация запроса → ответ → формат Markdown.\n"
                    "Примеры: приветствие, /help, погода.\n"
                    "Ограничения: 2000 символов, без ссылок, без ключей.\n"
                    "Уточняющий вопрос: основной функционал бота?"
                ),
            },
        }
        lang_ex = "uz" if "uz" in system else "ru"
        ex = examples.get(lang_ex)

        prompt = f"""
{system}

=== Yo'riqnoma / Инструкция ===
{instruction}

=== Misol / Пример ===
Kirish / Вход: {ex['input'] if ex else '-'}
Chiqish / Выход: {ex['output'] if ex else '-'}

=== Foydalanuvchi kirishi / Пользовательский ввод ===
{raw_prompt}

=== Vazifa / Задача ===
1) Kirishdan kelib chiqib, tanlangan rejimga mos ravishda to'liq TUZILGAN yakuniy PROMPT yozing.
2) Tilni saqlang (UZ yoki RU). Max {self.cfg.max_bullets} banddan oshirmang.
3) Agar kerak bo'lsa, "Nimalar kiritilmasin / Исключить" bandini qo'shing.
""".strip()
        return prompt

    def run_once(self, model: str, num_ctx: int, prompt: str, stream: bool) -> str:
        llm = make_llm(model, self.cfg.base_url, self.cfg.temperature, self.cfg.top_p, num_ctx)
        if stream:
            return stream_text(llm, prompt)
        return llm.invoke(prompt)

    def refine(self, raw_prompt: str, mode: Literal["coding","image","video","chatgpt"], model: str, num_ctx: int, stream: bool) -> str:
        preload_keepalive(self.cfg.base_url or "http://localhost:11434", model, self.cfg.keep_alive)
        prompt = self._make_prompt(raw_prompt, mode)
        return self.run_once(model, num_ctx, prompt, stream)

# ---------------------------
# Builder / CLI
# ---------------------------
def build_refiner() -> PromptRefiner:
    cfg = load_config()
    with open("templates/system_uz.txt", "r", encoding="utf-8") as f:
        uz = f.read()
    with open("templates/system_ru.txt", "r", encoding="utf-8") as f:
        ru = f.read()
    with open("templates/modes.yaml", "r", encoding="utf-8") as f:
        modes = yaml.safe_load(f)
    systems = {"uz": uz, "ru": ru}
    return PromptRefiner(cfg, systems, modes)

if __name__ == "__main__":
    import argparse
    cfg = load_config()

    parser = argparse.ArgumentParser(description="UZ/RU Prompt Engineer (profiles + cascade)")
    parser.add_argument("prompt", type=str)
    parser.add_argument("--mode", type=str, default="chatgpt", choices=SUPPORTED_MODES)

    # profiles
    parser.add_argument("--profile", type=str, default="speed", choices=["speed","balanced","max"])
    parser.add_argument("--cascade", action="store_true", help="Try fast model, auto-fallback to 20B if quality low")

    # manual overrides
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--num-ctx", type=int, default=None)
    parser.add_argument("--no-stream", action="store_true")

    args = parser.parse_args()
    stream = not args.no_stream

    # Profile presets (fastest first)
    presets = {
        "speed":    {"model": "qwen2.5:7b",   "num_ctx": 2048},
        "balanced": {"model": "mistral:latest","num_ctx": 4096},
        "max":      {"model": cfg.model,      "num_ctx": cfg.num_ctx},  # gpt-oss:20b from config
    }
    prof = presets[args.profile]

    # Apply manual overrides
    model = args.model or prof["model"]
    num_ctx = args.num_ctx or prof["num_ctx"]

    refiner = build_refiner()

    # Run (with optional cascade)
    prompt = refiner._make_prompt(args.prompt, args.mode)

    print(f"[Model: {model} | num_ctx: {num_ctx} | host: {cfg.base_url or 'http://localhost:11434'} | profile: {args.profile} | cascade: {args.cascade}]")
    print("=" * 80)

    text = refiner.run_once(model, num_ctx, prompt, stream=stream)

    if args.cascade:
        s = score_output(args.mode, text)
        if s < 0.8:
            print("\n[Auto-fallback → gpt-oss:20b for higher quality]\n")
            text = refiner.run_once("gpt-oss:20b", cfg.num_ctx, prompt, stream=stream)

    if args.no_stream:
        print(text)
    else:
        print("\n")
# ---------------------------
# FastAPI Bridge Function
# ---------------------------

def refine(text: str, mode: Literal["coding", "image", "video", "chatgpt"] = "chatgpt", profile: str = "balanced") -> str:
    """
    Entry point for external apps (e.g., FastAPI) to refine a prompt.
    Defaults: mode='chatgpt', profile='balanced'.
    """
    cfg = load_config()
    refiner = build_refiner()

    # Profile presets
    presets = {
        "speed":    {"model": "qwen2.5:7b",   "num_ctx": 2048},
        "balanced": {"model": "mistral:latest","num_ctx": 4096},
        "max":      {"model": cfg.model,      "num_ctx": cfg.num_ctx},
    }
    prof = presets.get(profile, presets["balanced"])

    # Generate
    result = refiner.refine(
        raw_prompt=text,
        mode=mode,
        model=prof["model"],
        num_ctx=prof["num_ctx"],
        stream=False
    )
    return result
