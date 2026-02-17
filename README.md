# 🐾 VetVision - AI-Powered Veterinary Advisor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![Gemini API](https://img.shields.io/badge/AI-Google%20Gemini-4285F4) ![License](https://img.shields.io/badge/License-MIT-green)

**VetVision** is an intelligent veterinary advisory system designed to assist pet owners. By leveraging deep learning for breed classification and Large Language Models (LLMs) for medical advice, it provides comprehensive health insights for dogs.

## 🚀 Key Features

- **🐶 Breed Detection:** Identifies dog breeds from images with high accuracy using an **EfficientNetB0** based deep learning model.
- **🩺 AI Veterinarian:** Integrates **Google Gemini API** to offer breed-specific health care tips, nutritional advice, and potential genetic risks.
- **📄 Smart Reporting:** Generates professional **PDF health reports** summarizing the analysis results.
- **💻 Modern UI:** Features a sleek, dark-themed desktop interface built with **CustomTkinter**.

## 🛠️ Technology Stack

| Component | Technology | Description |
|-----------|------------|-------------|
| **Core** | Python | Main programming language |
| **Deep Learning** | TensorFlow / Keras | EfficientNetB0 model for image classification |
| **LLM** | Google Gemini API | Generative AI for veterinary advice |
| **GUI** | CustomTkinter | Modern UI framework for Python |
| **Image Processing** | OpenCV & MRI | Image handling and preprocessing |
| **Reporting** | ReportLab | PDF generation engine |

## 📂 Project Structure

```bash
vetvision/
├── app.py               # Main application entry point & UI logic
├── train_model.py       # Script for training the breed classification model
├── llm_test_api.py      # Utility to test Gemini API integration
├── vetvision_model.h5   # Pre-trained deep learning model
├── confusion_matrix.png # Model performance visualization
├── labels.txt           # List of supported dog breeds
└── requirements.txt     # Python dependencies
```

## ⚡ Getting Started

### Prerequisites
- Python 3.8 or higher
- A Google Cloud API Key for Gemini

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/batu3384/vetvision.git
   cd vetvision
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *(If requirements.txt is missing, install manually: `pip install tensorflow customtkinter google-generativeai reportlab opencv-python pillow pandas numpy`)*

3. **Run the application**
   ```bash
   python app.py
   ```

## 📜 License
This project is licensed under the MIT License.
