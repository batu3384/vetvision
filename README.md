# VetVision - AI Destekli Veteriner Danismanlik Sistemi

Evcil hayvan sahipleri icin gelistirilmis, yapay zeka destekli veteriner danismanlik uygulamasi. Kopek fotograflarindan irk tespiti yapar ve Google Gemini ile saglik onerileri sunar.

## Ozellikler

- Goruntu Siniflandirma: EfficientNetB0 tabanli derin ogrenme modeli ile kopek irki tespiti
- - AI Danisman: Google Gemini API entegrasyonu ile irka ozel veteriner tavsiyeleri
  - - PDF Raporlama: Analiz sonuclarini profesyonel PDF formatinda disa aktarma
    - - Modern Arayuz: CustomTkinter ile sik masaustu uygulamasi
     
      - ## Teknolojiler
     
      - | Kategori | Teknoloji |
      - |----------|----------|
      - | Derin Ogrenme | TensorFlow, Keras, EfficientNetB0 |
      - | LLM | Google Gemini API |
      - | Arayuz | CustomTkinter |
      - | Goruntu Isleme | OpenCV, Pillow |
      - | Raporlama | ReportLab |
      - | Veri | Pandas, NumPy |
     
      - ## Proje Yapisi
     
      - ```
        vetvision/
        |-- app.py              # Ana uygulama ve arayuz
        |-- train_model.py      # Model egitim scripti
        |-- llm_test_api.py     # Gemini API test modulu
        |-- vetvision_model.h5  # Egitilmis model dosyasi
        |-- confusion_matrix.png # Model basarim grafigi
        ```

        ## Kurulum

        ```bash
        git clone https://github.com/batu3384/vetvision.git
        cd vetvision
        pip install tensorflow customtkinter google-generativeai reportlab opencv-python pillow pandas numpy
        python app.py
        ```

        ## Lisans

        MIT License
