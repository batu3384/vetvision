import os
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except Exception as e:
    raise ImportError("seaborn gerekli paketi bulunamadı. Lütfen: pip install seaborn") from e

try:
    import tensorflow as tf
    # Kısaltmalar
    load_model = tf.keras.models.load_model
    ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
    # EfficientNet preprocessing
    try:
        from tensorflow.keras.applications.efficientnet import preprocess_input
    except Exception as e:
        raise ImportError("EfficientNet preprocess_input bulunamadı. Lütfen 'tensorflow' sürümünüzü kontrol edin veya gerekli paketleri yükleyin.") from e
except Exception as e:
    raise ImportError("TensorFlow veya Keras yüklenemedi. Lütfen uygun ortamda 'pip install tensorflow' çalıştırın.") from e

from sklearn.metrics import confusion_matrix, classification_report

# 1. Stil ayarı
sns.set(style="darkgrid")

# 2. Model ve veri yükleme
MODEL_PATH = "vetvision_b3_model.keras" if os.path.exists("vetvision_b3_model.keras") else "vetvision_model.h5"
# Otomatik test dizini seçimi: tercih sırası
candidate_dirs = [
    "dataset/organized/val",
    "dataset/test",
    "dataset/val",
    "dataset/organized/test",
    "dataset/organized/validation",
]

def dir_has_images(path):
    if not os.path.isdir(path):
        return False
    # alt klasörlerde en az bir görüntü dosyası bulunup bulunmadığını kontrol et
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                return True
    return False

TEST_DIR = None
for d in candidate_dirs:
    if dir_has_images(d):
        TEST_DIR = d
        break
if TEST_DIR is None:
    raise FileNotFoundError(
        "Test/validation dizini bulunamadı veya içinde görüntü yok. Lütfen 'dataset/test' veya 'dataset/organized/val' gibi dizinlerden birini oluşturun ve içinde sınıf alt klasörlarıyla resimleri yerleştirin."
    )
# Modelin kullandığı boyut ve batch size
BATCH_SIZE = 32

# Model giriş boyutunu MODEL_PATH'e göre ayarla (B3 -> 300, B0/Başka -> 224)
if 'b3' in MODEL_PATH.lower():
    IMG_SIZE = (300, 300)
else:
    IMG_SIZE = (224, 224)

# Test verisini yükle (preprocessing fonksiyonu kritik)
try:
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
except NameError:
    # Eğer preprocess_input tanımlı değilse fallback olarak rescale kullan
    test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Sınıf isimleri (index sırasına göre)
class_indices = test_generator.class_indices
class_names = [None] * len(class_indices)
for name, idx in class_indices.items():
    class_names[idx] = name

# Modeli yükle (varlığını kontrol et ve compile=False kullan)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Lütfen model dosyasını yerleştirin veya MODEL_PATH'i güncelleyin.")
model = load_model(MODEL_PATH, compile=False)

# 3. Tahmin yap
# predict ile generator kullanırken adım sayısını belirtmek daha güvenlidir
steps = int(np.ceil(test_generator.samples / float(BATCH_SIZE)))
predictions = model.predict(test_generator, steps=steps, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# 4. Karmaşıklık Matrisi
cm = confusion_matrix(y_true, y_pred)

# En çok karıştırılan 10 ırkı bul
cm_sum = cm.sum(axis=1) + cm.sum(axis=0)
top10_idx = np.argsort(cm_sum)[-10:]
top10_labels = [class_names[i] for i in top10_idx]
cm_top10 = cm[np.ix_(top10_idx, top10_idx)]

plt.figure(figsize=(10,8))
sns.heatmap(cm_top10, annot=True, fmt="d", cmap="mako", xticklabels=top10_labels, yticklabels=top10_labels)
plt.title("En Çok Karıştırılan 10 Irk - Karmaşıklık Matrisi")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# 5. Classification Report
report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
print(report)
with open("report.txt", "w", encoding="utf-8") as f:
    f.write(report)

print("Karmaşıklık matrisi ve rapor başarıyla oluşturuldu.")
