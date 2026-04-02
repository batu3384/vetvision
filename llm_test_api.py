"""Small Gemini smoke-test utility for VetVision."""

from __future__ import annotations

import os
import sys

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
MODEL_OPTIONS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-flash-latest",
]


def get_working_model():
    """Return the first available model that can generate content."""
    if not API_KEY:
        raise RuntimeError("GEMINI_API_KEY tanımlı değil.")

    genai.configure(api_key=API_KEY)

    for model_name in MODEL_OPTIONS:
        try:
            model = genai.GenerativeModel(
                model_name,
                generation_config={"temperature": 0.3},
            )
            model.generate_content("test")
            return model, model_name
        except Exception:
            continue

    raise RuntimeError("Kullanılabilir Gemini modeli bulunamadı.")


def build_prompt(breed_name: str) -> str:
    return f"""
Sen bir veteriner bilgi asistanısın.
{breed_name} ırkı için kısa ve güvenli bir ön bilgi hazırla.

Şu başlıkları kullan:
1. Genel profil
2. Dikkat edilmesi gereken sağlık noktaları
3. Beslenme ve egzersiz notu
4. Veterinere ne zaman başvurulmalı

Yanıt kısa, açık ve Türkçe olsun.
"""


def main() -> int:
    breed_name = sys.argv[1] if len(sys.argv) > 1 else "Golden Retriever"

    try:
        model, model_name = get_working_model()
        response = model.generate_content(build_prompt(breed_name))
        print(f"Model: {model_name}\n")
        print(response.text.strip())
        return 0
    except Exception as exc:
        print(f"Gemini testi başarısız: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
