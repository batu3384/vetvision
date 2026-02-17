# VetVision - AI-Powered Veterinary Advisor

An AI-powered veterinary advisory application for pet owners. Detects dog breeds from photos using deep learning and provides health recommendations via Google Gemini.

## Features

- **Breed Detection:** EfficientNetB0-based deep learning model for dog breed classification
- **AI Advisor:** Breed-specific veterinary recommendations powered by Google Gemini API
- **PDF Reports:** Export analysis results as professional PDF documents
- **Modern UI:** Sleek desktop application built with CustomTkinter

## Tech Stack

| Category | Technology |
|----------|-----------|
| Deep Learning | TensorFlow, Keras, EfficientNetB0 |
| LLM | Google Gemini API |
| UI | CustomTkinter |
| Image Processing | OpenCV, Pillow |
| Reporting | ReportLab |
| Data | Pandas, NumPy |

## Project Structure

```
vetvision/
├── app.py               # Main application & UI
├── train_model.py       # Model training script
├── llm_test_api.py      # Gemini API test module
├── vetvision_model.h5   # Trained model file
└── confusion_matrix.png # Model performance chart
```

## Getting Started

```bash
git clone https://github.com/batu3384/vetvision.git
cd vetvision
pip install tensorflow customtkinter google-generativeai reportlab opencv-python pillow pandas numpy
python app.py
```

## License

MIT License
