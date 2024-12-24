import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# 1. CSV ve Görüntüleri Yükle
data_dir = "C:/Users/myazo/OneDrive/Masaüstü/train/train" # Görüntülerin olduğu dizin
csv_path = "C:/Users/myazo/OneDrive/Masaüstü/train_data.csv"  # Etiketlerin olduğu CSV dosyası

# CSV dosyasını pandas ile yükle
df = pd.read_csv(csv_path)

# Görüntü dosyalarının tam yollarını ekle
df['filepath'] = df['filename'].apply(lambda x: os.path.join(data_dir, x))

# 2. Eğitim ve Test Setlerini Ayır
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 3. Veri Setlerini TensorFlow Dataset Formatına Çevir
def process_image(file_path, label, img_size=(150, 150)):
    # Görüntüyü yükle ve yeniden boyutlandır
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = img / 255.0  # Normalize et
    return img, label

def create_dataset(df, label_map, img_size=(150, 150), batch_size=32):
    file_paths = df['filepath'].values
    labels = df['city'].map(label_map).values  # Etiketleri sayısal değerlere çevir
    labels = tf.keras.utils.to_categorical(labels, num_classes=len(label_map))  # One-hot encode
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(lambda x, y: process_image(x, y, img_size))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Sınıf etiketlerini sayısal değerlere dönüştürmek için bir haritalama oluştur
unique_labels = df['city'].unique()
label_map = {label: idx for idx, label in enumerate(unique_labels)}
print(label_map)

# Eğitim ve test veri setlerini oluştur
train_dataset = create_dataset(train_df, label_map)
test_dataset = create_dataset(test_df, label_map)

# 4. Modeli Oluştur
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(unique_labels), activation='softmax')  # Sınıf sayısına göre çıktı
])

# 5. Modeli Derleme
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 6. Modeli Eğitme
model.fit(train_dataset, epochs=10, validation_data=test_dataset)

# 7. Modeli Değerlendirme
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Doğruluğu: {test_accuracy:.2f}")

model.save('my_model2.h5')