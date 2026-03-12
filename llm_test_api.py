# Google Gemini API ile Veteriner Tavsiye Sistemi
# Kurulum: pip install google-generativeai
# Not: API anahtarını https://aistudio.google.com adresinden ücretsiz alabilirsiniz.

import os
import google.generativeai as genai

# API Key - environment variable kullanin (hardcode etmeyin!)
API_KEY = os.environ.get("GOOGLE_API_KEY", "")

genai.configure(api_key=API_KEY)

# Model isimleri - sırayla denenecek
MODEL_OPTIONS = [
    'gemini-2.5-flash',
    'gemini-2.0-flash',
    'gemini-2.0-flash-lite',
    'gemini-flash-latest'
]


def list_available_models():
    """Kullanılabilir modelleri listele."""
    print("\n📋 Mevcut Modeller Şunlardır:")
    print("-" * 50)
    try:
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                print(f"  ✅ {model.name}")
    except Exception as e:
        print(f"  ❌ Model listesi alınamadı: {e}")
    print("-" * 50)


def get_working_model():
    """Çalışan bir model bul."""
    for model_name in MODEL_OPTIONS:
        try:
            model = genai.GenerativeModel(
                model_name,
                generation_config={'temperature': 0.3}
            )
            # Test et
            model.generate_content("test")
            return model, model_name
        except Exception:
            continue
    return None, None


def veteriner_tavsiyesi(irk_ismi: str) -> str:
    """
    Verilen köpek ırkı için Gemini API ile detaylı bilgi kartı üretir.
    
    Args:
        irk_ismi: Köpek ırkının adı (örn: "Golden Retriever")
    
    Returns:
        Markdown formatında veteriner bilgi kartı
    """
    try:
        model, model_name = get_working_model()
        
