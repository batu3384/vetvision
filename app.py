"""
VetVision Pro - Dog Breed Classifier
Clean & Minimal Light Theme UI
"""
import json
import threading
import os
from pathlib import Path
from datetime import datetime

import customtkinter as ctk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(2)
except:
    try:
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

import tensorflow as tf

# PDF için
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import HexColor
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Drag & Drop için
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False

# Gemini AI için
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
    GEMINI_AVAILABLE = bool(GEMINI_API_KEY)
    if GEMINI_AVAILABLE:
        genai.configure(api_key=GEMINI_API_KEY)
except ImportError:
    GEMINI_AVAILABLE = False
    GEMINI_API_KEY = ""

preprocess_input = tf.keras.applications.efficientnet.preprocess_input

MODEL_PATH = Path("artifacts/model/vetvision_model.h5")
LABELS_PATH = Path("artifacts/model/labels.txt")
IMG_SIZE = (224, 224)

# ═══════════════════════════════════════════════════════════════════════════════
# 🎨 MODERN LIGHT THEME
# ═══════════════════════════════════════════════════════════════════════════════
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

# Color Palette
COLORS = {
    "bg": "#f0f4f8",
    "card": "#ffffff",
    "card_alt": "#f8fafc",
    "primary": "#3b82f6",
    "primary_hover": "#2563eb",
    "secondary": "#64748b",
    "secondary_hover": "#475569",
    "success": "#10b981",
    "success_hover": "#059669",
    "accent": "#8b5cf6",
    "text": "#0f172a",
    "text_secondary": "#64748b",
    "border": "#e2e8f0",
    "border_focus": "#3b82f6",
}


class VetVisionApp(ctk.CTk):
    """Dog Breed Classifier - Modern Clean UI"""
    
    def __init__(self):
        super().__init__()
        
        self.title("VetVision Pro")
        self.geometry("1200x750")
        self.minsize(1000, 650)
        self.configure(fg_color=COLORS["bg"])
        
        # State
        self.model = None
        self.labels = None
        self.current_image = None
        self.current_image_path = None
        self.current_breed = None
        self.current_confidence = 0
        self.photo_ref = None
        
        self._build_ui()
        self._setup_drag_drop()
        self._load_model()
        
        # Center
        self.update_idletasks()
        x = (self.winfo_screenwidth() - 1200) // 2
        y = (self.winfo_screenheight() - 750) // 2
        self.geometry(f"+{x}+{y}")
    
    def _build_ui(self):
        """Build modern clean UI"""
        
        # Main container
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=40, pady=40)
        
        main.grid_columnconfigure(0, weight=0, minsize=380)
        main.grid_columnconfigure(1, weight=1)
        main.grid_rowconfigure(0, weight=1)
        
        # ═══════════════════════════════════════════════════════════════
        # LEFT PANEL - Image & Buttons
        # ═══════════════════════════════════════════════════════════════
        left = ctk.CTkFrame(main, corner_radius=20, fg_color=COLORS["card"])
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 24))
        
        left_inner = ctk.CTkFrame(left, fg_color="transparent")
        left_inner.pack(fill="both", expand=True, padx=28, pady=28)
        
        # Title with gradient-like effect
        title_frame = ctk.CTkFrame(left_inner, fg_color="transparent")
        title_frame.pack(fill="x", pady=(0, 24))
        
        ctk.CTkLabel(
            title_frame, text="🐕",
            font=("Segoe UI Emoji", 36)
        ).pack(side="left", padx=(0, 12))
        
        title_text = ctk.CTkFrame(title_frame, fg_color="transparent")
        title_text.pack(side="left")
        
        ctk.CTkLabel(
            title_text, text="VetVision",
            font=("Segoe UI", 28, "bold"), text_color=COLORS["primary"]
        ).pack(anchor="w")
        
        ctk.CTkLabel(
            title_text, text="Köpek Irk Tanıma",
            font=("Segoe UI", 12), text_color=COLORS["text_secondary"]
        ).pack(anchor="w")
        
        # Divider
        ctk.CTkFrame(left_inner, height=1, fg_color=COLORS["border"]).pack(fill="x", pady=(0, 24))
        
        # Image Container with modern border
        self.image_container = ctk.CTkFrame(
            left_inner, height=300, corner_radius=16,
            fg_color=COLORS["card_alt"], border_width=2, border_color=COLORS["border"]
        )
        self.image_container.pack(fill="x")
        self.image_container.pack_propagate(False)
        
        # Placeholder
        self.placeholder_frame = ctk.CTkFrame(self.image_container, fg_color="transparent")
        self.placeholder_frame.place(relx=0.5, rely=0.5, anchor="center")
        
        ctk.CTkLabel(
            self.placeholder_frame, text="📷",
            font=("Segoe UI Emoji", 56), text_color=COLORS["border"]
        ).pack()
        
        text = "Sürükle & Bırak" if DND_AVAILABLE else "Fotoğraf Seç"
        ctk.CTkLabel(
            self.placeholder_frame, text=text,
            font=("Segoe UI", 18, "bold"), text_color=COLORS["text"]
        ).pack(pady=(12, 4))
        
        ctk.CTkLabel(
            self.placeholder_frame, text="veya butonu kullanın",
            font=("Segoe UI", 12), text_color=COLORS["text_secondary"]
        ).pack()
        
        self.image_label = ctk.CTkLabel(self.image_container, text="")
        
        # Buttons
        btn_frame = ctk.CTkFrame(left_inner, fg_color="transparent")
        btn_frame.pack(fill="x", pady=(28, 0))
        
        self.select_btn = ctk.CTkButton(
            btn_frame, text="📁  Fotoğraf Seç",
            command=self._select_image,
            font=("Segoe UI", 14, "bold"),
            height=48, corner_radius=12,
            fg_color=COLORS["secondary"], hover_color=COLORS["secondary_hover"],
            text_color="white",
            state="disabled"
        )
        self.select_btn.pack(fill="x", pady=(0, 12))
        
        self.analyze_btn = ctk.CTkButton(
            btn_frame, text="⚡  Analizi Başlat",
            command=self._analyze,
            font=("Segoe UI", 15, "bold"),
            height=54, corner_radius=12,
            fg_color=COLORS["primary"], hover_color=COLORS["primary_hover"],
            text_color="white",
            state="disabled"
        )
        self.analyze_btn.pack(fill="x")
        
        # Status with icon
        status_frame = ctk.CTkFrame(left_inner, fg_color="transparent")
        status_frame.pack(fill="x", pady=(20, 0))
        
        self.status_icon = ctk.CTkLabel(
            status_frame, text="⏳",
            font=("Segoe UI Emoji", 14)
        )
        self.status_icon.pack(side="left", padx=(0, 8))
        
        self.status_label = ctk.CTkLabel(
            status_frame, text="Model yükleniyor...",
            font=("Segoe UI", 12), text_color=COLORS["text_secondary"]
        )
        self.status_label.pack(side="left")
        
        # ═══════════════════════════════════════════════════════════════
        # RIGHT PANEL - Results & Report
        # ═══════════════════════════════════════════════════════════════
        right = ctk.CTkFrame(main, corner_radius=20, fg_color=COLORS["card"])
        right.grid(row=0, column=1, sticky="nsew")
        
        right.grid_rowconfigure(0, weight=0)
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)
        
        # Results Section - Card style
        results_card = ctk.CTkFrame(
            right, corner_radius=16,
            fg_color=COLORS["card_alt"], border_width=1, border_color=COLORS["border"]
        )
        results_card.grid(row=0, column=0, sticky="ew", padx=28, pady=(28, 0))
        
        results = ctk.CTkFrame(results_card, fg_color="transparent")
        results.pack(fill="x", padx=24, pady=20)
        
        # Result header
        ctk.CTkLabel(
            results, text="Tespit Edilen Irk",
            font=("Segoe UI", 12), text_color=COLORS["text_secondary"]
        ).pack(anchor="w")
        
        self.breed_label = ctk.CTkLabel(
            results, text="Fotoğraf yükleyin",
            font=("Segoe UI", 32, "bold"), text_color=COLORS["text"]
        )
        self.breed_label.pack(anchor="w", pady=(4, 0))
        
        # Confidence row
        conf_row = ctk.CTkFrame(results, fg_color="transparent")
        conf_row.pack(fill="x", pady=(16, 0))
        
        self.conf_label = ctk.CTkLabel(
            conf_row, text="Güven Oranı",
            font=("Segoe UI", 12), text_color=COLORS["text_secondary"]
        )
        self.conf_label.pack(side="left")
        
        self.conf_percent = ctk.CTkLabel(
            conf_row, text="—",
            font=("Segoe UI", 14, "bold"), text_color=COLORS["primary"]
        )
        self.conf_percent.pack(side="right")
        
        self.conf_bar = ctk.CTkProgressBar(
            results, height=10,
            progress_color=COLORS["primary"], fg_color=COLORS["border"],
            corner_radius=5
        )
        self.conf_bar.pack(fill="x", pady=(8, 0))
        self.conf_bar.set(0)
        
        # Other Predictions Section
        other_preds_frame = ctk.CTkFrame(results, fg_color="transparent")
        other_preds_frame.pack(fill="x", pady=(16, 0))
        
        ctk.CTkLabel(
            other_preds_frame, text="Diğer Olası Irklar",
            font=("Segoe UI", 12), text_color=COLORS["text_secondary"]
        ).pack(anchor="w")
        
        self.other_preds_container = ctk.CTkFrame(other_preds_frame, fg_color="transparent")
        self.other_preds_container.pack(fill="x", pady=(8, 0))
        
        # Create 3 prediction labels
        self.other_pred_labels = []
        for i in range(3):
            pred_row = ctk.CTkFrame(self.other_preds_container, fg_color="transparent")
            pred_row.pack(fill="x", pady=2)
            
            name_label = ctk.CTkLabel(
                pred_row, text="—",
                font=("Segoe UI", 11), text_color=COLORS["text_secondary"]
            )
            name_label.pack(side="left")
            
            conf_label = ctk.CTkLabel(
                pred_row, text="",
                font=("Segoe UI", 11, "bold"), text_color=COLORS["text_secondary"]
            )
            conf_label.pack(side="right")
            
            self.other_pred_labels.append((name_label, conf_label))
        
        # AI Report Section
        report_frame = ctk.CTkFrame(right, fg_color="transparent")
        report_frame.grid(row=1, column=0, sticky="nsew", padx=28, pady=(20, 28))
        
        # Report Header
        report_header = ctk.CTkFrame(report_frame, fg_color="transparent")
        report_header.pack(fill="x", pady=(0, 12))
        
        title_row = ctk.CTkFrame(report_header, fg_color="transparent")
        title_row.pack(side="left")
        
        ctk.CTkLabel(
            title_row, text="🤖",
            font=("Segoe UI Emoji", 18)
        ).pack(side="left", padx=(0, 8))
        
        ctk.CTkLabel(
            title_row, text="Veteriner Raporu",
            font=("Segoe UI", 18, "bold"), text_color=COLORS["text"]
        ).pack(side="left")
        
        btn_box = ctk.CTkFrame(report_header, fg_color="transparent")
        btn_box.pack(side="right")
        
        self.info_btn = ctk.CTkButton(
            btn_box, text="✨ Rapor Oluştur",
            command=self._get_info,
            font=("Segoe UI", 12, "bold"),
            width=140, height=36, corner_radius=10,
            fg_color=COLORS["success"], hover_color=COLORS["success_hover"],
            text_color="white",
            state="disabled"
        )
        self.info_btn.pack(side="left", padx=(0, 10))
        
        self.copy_btn = ctk.CTkButton(
            btn_box, text="📋 Kopyala",
            command=self._copy_info,
            font=("Segoe UI", 12, "bold"),
            width=100, height=36, corner_radius=10,
            fg_color=COLORS["card_alt"], hover_color=COLORS["border"],
            text_color=COLORS["text"],
            border_width=1, border_color=COLORS["border"]
        )
        self.copy_btn.pack(side="left", padx=(0, 8))
        
        self.pdf_btn = ctk.CTkButton(
            btn_box, text="📄 PDF İndir",
            command=self._export_pdf,
            font=("Segoe UI", 12, "bold"),
            width=110, height=36, corner_radius=10,
            fg_color=COLORS["card_alt"], hover_color=COLORS["border"],
            text_color=COLORS["text"],
            border_width=1, border_color=COLORS["border"],
            state="disabled"
        )
        self.pdf_btn.pack(side="left")
        
        # Report Text with modern styling
        self.info_text = ctk.CTkTextbox(
            report_frame,
            font=("Segoe UI", 14),
            fg_color=COLORS["card_alt"],
            text_color=COLORS["text"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["border"],
            wrap="word"
        )
        self.info_text.pack(fill="both", expand=True)
        self.info_text.insert("1.0", "Fotoğraf yükleyip analiz yaptıktan sonra AI destekli veteriner raporu oluşturabilirsiniz.")
        self.info_text.configure(state="disabled")
    
    def _setup_drag_drop(self):
        if DND_AVAILABLE:
            try:
                self.image_container.drop_target_register(DND_FILES)
                self.image_container.dnd_bind('<<Drop>>', self._on_drop)
            except:
                pass
    
    def _on_drop(self, event):
        path = event.data
        if path.startswith('{') and path.endswith('}'):
            path = path[1:-1]
        if ' ' in path and not os.path.exists(path):
            path = path.split()[0]
        
        ext = Path(path).suffix.lower()
        if ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
            self._load_image(path)
    
    def _load_model(self):
        def load():
            self.model = tf.keras.models.load_model(MODEL_PATH)
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.labels = {int(v): k.replace("_", " ").title() for k, v in data.items()}
            self.after(0, self._on_model_loaded)
        
        threading.Thread(target=load, daemon=True).start()
    
    def _on_model_loaded(self):
        self.status_label.configure(text="Hazır")
        self.select_btn.configure(state="normal")
    
    def _select_image(self):
        path = filedialog.askopenfilename(
            title="Köpek Fotoğrafı Seç",
            filetypes=[("Resim", "*.jpg *.jpeg *.png *.webp *.bmp")]
        )
        if path:
            self._load_image(path)
    
    def _load_image(self, path):
        self.current_image = Image.open(path)
        self.current_image_path = path
        
        self.placeholder_frame.place_forget()
        
        # Get container dimensions
        container_w = self.image_container.winfo_width() - 20
        container_h = self.image_container.winfo_height() - 20
        
        # Use default if not yet rendered
        if container_w < 50:
            container_w = 320
        if container_h < 50:
            container_h = 280
        
        display = self.current_image.copy()
        w, h = display.size
        
        # Calculate ratio to fill the container while maintaining aspect ratio
        ratio = min(container_w / w, container_h / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        display = display.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        self.photo_ref = ctk.CTkImage(display, size=(new_w, new_h))
        self.image_label.configure(image=self.photo_ref, text="")
        self.image_label.place(relx=0.5, rely=0.5, anchor="center")
        
        self.analyze_btn.configure(state="normal")
        self._reset_results()
        self.status_label.configure(text=f"Yüklendi: {Path(path).name}")
    
    def _reset_results(self):
        self.breed_label.configure(text="Analiz bekleniyor...")
        self.conf_label.configure(text="Güven Oranı")
        self.conf_percent.configure(text="—")
        self.conf_bar.set(0)
        self.info_btn.configure(state="disabled")
        self.pdf_btn.configure(state="disabled")
        
        # Reset other predictions
        for name_lbl, conf_lbl in self.other_pred_labels:
            name_lbl.configure(text="—")
            conf_lbl.configure(text="")
        
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", "Analiz sonrası rapor oluşturabilirsiniz.")
        self.info_text.configure(state="disabled")
        
        self.current_breed = None
        self.current_confidence = 0
    
    def _analyze(self):
        if not self.current_image or not self.model:
            return
        
        self.analyze_btn.configure(state="disabled")
        self.status_label.configure(text="Analiz ediliyor...")
        
        def run():
            try:
                base_img = self.current_image.convert("RGB")
                
                augmented = [
                    base_img,
                    base_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
                    base_img.rotate(10, fillcolor=(128, 128, 128)),
                    base_img.rotate(-10, fillcolor=(128, 128, 128)),
                ]
                
                all_preds = []
                for aug in augmented:
                    img = aug.resize(IMG_SIZE)
                    arr = preprocess_input(np.expand_dims(np.array(img, dtype="float32"), 0))
                    preds = self.model.predict(arr, verbose=0)[0]
                    all_preds.append(preds)
                
                avg_preds = np.mean(all_preds, axis=0)
                
                # Get top 4 predictions
                top_indices = np.argsort(avg_preds)[-4:][::-1]
                top_results = []
                for idx in top_indices:
                    breed_name = self.labels[int(idx)]
                    confidence = float(avg_preds[idx]) * 100
                    top_results.append((breed_name, confidence))
                
                self.after(0, lambda: self._show_results(top_results))
            except Exception as e:
                self.after(0, lambda: self._on_error(str(e)))
        
        threading.Thread(target=run, daemon=True).start()
    
    def _on_error(self, error):
        self.analyze_btn.configure(state="normal")
        self.status_label.configure(text=f"Hata: {error}")
    
    def _show_results(self, top_results):
        # First result is the main prediction
        breed, conf = top_results[0]
        self.current_breed = breed
        self.current_confidence = conf
        
        if conf < 40:
            self.breed_label.configure(text="Tespit Edilemedi", text_color="#dc3545")
            self.conf_percent.configure(text=f"%{conf:.1f}")
        else:
            self.breed_label.configure(text=breed, text_color=COLORS["text"])
            self.conf_percent.configure(text=f"%{conf:.1f}")
            self.info_btn.configure(state="normal")
        
        self.conf_bar.set(conf / 100)
        
        # Show other predictions (2nd, 3rd, 4th)
        for i, (name_lbl, conf_lbl) in enumerate(self.other_pred_labels):
            if i + 1 < len(top_results):
                other_breed, other_conf = top_results[i + 1]
                name_lbl.configure(text=other_breed)
                conf_lbl.configure(text=f"%{other_conf:.1f}")
            else:
                name_lbl.configure(text="—")
                conf_lbl.configure(text="")
        
        self.analyze_btn.configure(state="normal")
        self.status_label.configure(text="Analiz tamamlandı")
    
    def _copy_info(self):
        self.info_text.configure(state="normal")
        text = self.info_text.get("1.0", "end-1c")
        self.info_text.configure(state="disabled")
        
        self.clipboard_clear()
        self.clipboard_append(text)
        self.status_label.configure(text="Kopyalandı!")
        self.after(2000, lambda: self.status_label.configure(text="Hazır"))
    
    def _export_pdf(self):
        if not PDF_AVAILABLE or not self.current_breed:
            return
        
        desktop = Path.home() / "Desktop"
        default_name = f"VetVision_{self.current_breed.replace(' ', '_')}.pdf"
        
        path = filedialog.asksaveasfilename(
            title="PDF Kaydet",
            initialdir=str(desktop),
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
            initialfile=default_name
        )
        
        if not path:
            return
        
        try:
            self._create_pdf(path)
            self.status_label.configure(text="PDF kaydedildi")
            if messagebox.askyesno("PDF", "Dosyayı açmak ister misiniz?"):
                os.startfile(path)
        except Exception as e:
            messagebox.showerror("Hata", str(e))
    
    def _create_pdf(self, path):
        fonts_dir = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')
        arial_path = os.path.join(fonts_dir, 'arial.ttf')
        arial_bold_path = os.path.join(fonts_dir, 'arialbd.ttf')
        
        use_turkish = False
        try:
            if os.path.exists(arial_path):
                pdfmetrics.registerFont(TTFont('Arial-TR', arial_path))
                pdfmetrics.registerFont(TTFont('Arial-TR-Bold', arial_bold_path if os.path.exists(arial_bold_path) else arial_path))
                use_turkish = True
        except:
            pass
        
        font_r = 'Arial-TR' if use_turkish else 'Helvetica'
        font_b = 'Arial-TR-Bold' if use_turkish else 'Helvetica-Bold'
        
        def clean(text):
            result = []
            for c in text:
                if ord(c) < 256 or c in 'ğĞüÜşŞıİöÖçÇ':
                    if c not in '•✓✗⚠⭐✨🐾🧠🩺🏠💡':
                        result.append(c)
            return ''.join(result)
        
        c = canvas.Canvas(path, pagesize=A4)
        w, h = A4
        
        c.setFont(font_b, 22)
        c.setFillColor(HexColor("#0d6efd"))
        c.drawString(2*cm, h - 2.5*cm, "VetVision Pro - Rapor")
        
        c.setFont(font_r, 10)
        c.setFillColor(HexColor("#6c757d"))
        c.drawString(2*cm, h - 3.2*cm, f"Tarih: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
        
        c.setStrokeColor(HexColor("#e9ecef"))
        c.line(2*cm, h - 3.6*cm, w - 2*cm, h - 3.6*cm)
        
        c.setFont(font_b, 14)
        c.setFillColor(HexColor("#1a1a2e"))
        c.drawString(2*cm, h - 4.6*cm, f"Irk: {self.current_breed}")
        
        c.setFont(font_r, 12)
        c.setFillColor(HexColor("#6c757d"))
        c.drawString(2*cm, h - 5.3*cm, f"Güven: %{self.current_confidence:.1f}")
        
        self.info_text.configure(state="normal")
        raw = self.info_text.get("1.0", "end-1c")
        self.info_text.configure(state="disabled")
        
        info = clean(raw).replace('**', '').replace('#', '')
        
        text_obj = c.beginText(2*cm, h - 6.5*cm)
        text_obj.setFont(font_r, 10)
        text_obj.setFillColor(HexColor("#495057"))
        
        for line in info.split('\n'):
            while len(line) > 90:
                pos = line.rfind(' ', 0, 90)
                if pos == -1: pos = 90
                text_obj.textLine(line[:pos])
                line = line[pos:].lstrip()
            text_obj.textLine(line)
            if text_obj.getY() < 3*cm:
                c.drawText(text_obj)
                c.showPage()
                text_obj = c.beginText(2*cm, h - 2*cm)
                text_obj.setFont(font_r, 10)
                text_obj.setFillColor(HexColor("#495057"))
        
        c.drawText(text_obj)
        c.save()
    
    def _get_info(self):
        if not self.current_breed or not GEMINI_AVAILABLE:
            return
        
        self.info_btn.configure(state="disabled")
        self.status_label.configure(text="Rapor hazırlanıyor...")
        
        def fetch():
            try:
                model = genai.GenerativeModel('gemini-2.5-flash',
                    generation_config={'temperature': 0.3})
                
                system_prompt = """Sen bir Veteriner Tıbbi Asistanısın. Görevin, köpek ırkları hakkında ansiklopedik, doğru ve açıklayıcı teknik veriler sunmaktır. Asla sohbet etme, duygusal yorumlar yapma. Sadece veriyi sun."""

                prompt = f"""{system_prompt}

GÖREV: Aşağıdaki köpek ırkı için teknik bilgi kartını doldur.
IRK: {self.current_breed}

KURALLAR:
1. Sohbet ifadeleri (Merhaba, Saygılar vb.) KESİNLİKLE YASAK.
2. Sağlık kısmında hastalığı sadece listeleme, yanına ne olduğunu kısaca açıkla.
3. %100 Türkçe yaz (Tıbbi terimlerin orijinalleri parantez içinde kalabilir).

İSTENEN ÇIKTI FORMATI:

🐾 {self.current_breed}

🧠 KARAKTER VE MİZAÇ ANALİZİ
(Buraya ırkın zeka seviyesi, eğitilebilirliği, aile ve çocuklarla uyumu, yalnız kalma toleransı hakkında tek bir dolu paragraf yaz.)

🩺 GENETİK SAĞLIK RİSKLERİ
(Bu ırkta sık görülen hastalıkları şu formatta yaz):
• [Hastalık Adı] ([Orijinal Tıbbi Terim]): [Bu hastalık nedir? Belirtisi nedir? 1 cümle ile açıkla.]
• [Hastalık Adı] ([Orijinal Tıbbi Terim]): [Bu hastalık nedir? Belirtisi nedir? 1 cümle ile açıkla.]
• [Hastalık Adı] ([Orijinal Tıbbi Terim]): [Bu hastalık nedir? Belirtisi nedir? 1 cümle ile açıkla.]

🏠 BAKIM VE YAŞAM GEREKSİNİMLERİ
• Egzersiz İhtiyacı: (Günlük süre ve yoğunluk düzeyi)
• Tüy Bakımı: (Fırçalama sıklığı ve özel bakım notları)
• Beslenme: (Diyet türü ve dikkat edilmesi gereken noktalar)
• Yaşam Alanı: (Apartman/ev uygunluğu)

💡 VETERİNER NOTU
(Bu ırka sahip olmayı düşünenler için tek cümlelik, kritik ve hayati bir uyarı.)"""

                response = model.generate_content(prompt)
                self.after(0, lambda: self._show_info(response.text))
            except Exception as e:
                self.after(0, lambda: self._show_info(f"Hata: {e}"))
        
        threading.Thread(target=fetch, daemon=True).start()
    
    def _show_info(self, text):
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", text)
        self.info_text.configure(state="disabled")
        
        self.info_btn.configure(state="normal")
        
        if not text.startswith("Hata"):
            self.pdf_btn.configure(state="normal")
        
        self.status_label.configure(text="Rapor hazır")


def main():
    if not MODEL_PATH.exists():
        print("Model bulunamadı!")
        return
    
    app = VetVisionApp()
    app.mainloop()


if __name__ == "__main__":
    main()
