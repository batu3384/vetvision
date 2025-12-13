# Google Gemini API ile Veteriner Tavsiye Sistemi
# Kurulum: pip install google-generativeai
# Not: API anahtarını https://aistudio.google.com adresinden ücretsiz alabilirsiniz.

import google.generativeai as genai

# API Anahtarını buraya yapıştır
API_KEY = ""

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
        
        if model is None:
            list_available_models()
            return "❌ Hiçbir model çalışmıyor. Yukarıdaki listeye bakın."
        
        print(f"[Kullanılan Model: {model_name}]")
        
        system_prompt = """Sen bir Veteriner Tıbbi Asistanısın. Görevin, köpek ırkları hakkında ansiklopedik, doğru ve açıklayıcı teknik veriler sunmaktır. Asla sohbet etme, duygusal yorumlar yapma. Sadece veriyi sun."""

        prompt = f"""{system_prompt}

GÖREV: Aşağıdaki köpek ırkı için teknik bilgi kartını doldur.
IRK: {irk_ismi}

KURALLAR:
1. Sohbet ifadeleri (Merhaba, Saygılar vb.) KESİNLİKLE YASAK.
2. Sağlık kısmında hastalığı sadece listeleme, yanına ne olduğunu kısaca açıkla.
3. %100 Türkçe yaz (Tıbbi terimlerin orijinalleri parantez içinde kalabilir).

İSTENEN ÇIKTI FORMATI:

# 🐾 {irk_ismi}

## 🧠 KARAKTER VE MİZAÇ ANALİZİ
(Buraya ırkın zeka seviyesi, eğitilebilirliği, aile ve çocuklarla uyumu, yalnız kalma toleransı hakkında tek bir dolu paragraf yaz.)

## 🩺 GENETİK SAĞLIK RİSKLERİ
(Bu ırkta sık görülen hastalıkları şu formatta yaz):
* **[Hastalık Adı] ([Orijinal Tıbbi Terim]):** [Bu hastalık nedir? Belirtisi nedir? 1 cümle ile açıkla.]
* **[Hastalık Adı] ([Orijinal Tıbbi Terim]):** [Bu hastalık nedir? Belirtisi nedir? 1 cümle ile açıkla.]
* **[Hastalık Adı] ([Orijinal Tıbbi Terim]):** [Bu hastalık nedir? Belirtisi nedir? 1 cümle ile açıkla.]

## 🏠 BAKIM VE YAŞAM GEREKSİNİMLERİ
* **Egzersiz İhtiyacı:** (Günlük süre ve yoğunluk düzeyi)
* **Tüy Bakımı:** (Fırçalama sıklığı ve özel bakım notları)
* **Beslenme:** (Diyet türü ve dikkat edilmesi gereken noktalar)
* **Yaşam Alanı:** (Apartman/ev uygunluğu)

## 💡 VETERİNER NOTU
(Bu ırka sahip olmayı düşünenler için tek cümlelik, kritik ve hayati bir uyarı.)"""

        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        error_msg = str(e)
        error_lower = error_msg.lower()
        if "404" in error_msg:
            list_available_models()
            return f"❌ Model bulunamadı (404). Yukarıdaki listeye bakın."
        elif "quota" in error_lower or "limit" in error_lower:
            return "❌ API Kotası Doldu. Lütfen daha sonra tekrar deneyin."
        elif "connection" in error_lower or "network" in error_lower:
            return "❌ Bağlantı Hatası. İnternet bağlantınızı kontrol edin."
        else:
            return f"❌ Hata: {error_msg}"


if __name__ == "__main__":
    print("=" * 70)
    print("🐾 VetVision - Veteriner Bilgi Sistemi 🐾")
    print("=" * 70)
    print()
    
    # Test: Golden Retriever
    irk = "Golden Retriever"
    
    print(f"📋 Test ırkı: {irk}")
    print()
    print("⚡ Google Gemini devrede... (Otomatik Model Seçimi)")
    print("-" * 70)
    print()
    
    sonuc = veteriner_tavsiyesi(irk)
    print(sonuc)
    print()
    print("-" * 70)
    if not sonuc.startswith("❌"):
        print("✅ Teknik bilgi kartı başarıyla oluşturuldu!")
    else:
        print("💡 API anahtarınızı kontrol edin veya https://aistudio.google.com adresinden yeni bir anahtar alın.")
