from tensorflow.keras.models import load_model
import tensorflow as tf
import pandas as pd
import os
import numpy as np

# Load the trained model
model = load_model('my_model.h5')  # Use your saved model path

def preprocess_image(file_path, img_size=(150, 150)):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = img / 255.0  # Normalize to [0, 1]
    return img

# Directory containing new images
image_dir = "C:/Users/myazo/OneDrive/Masaüstü/test/test"

# List all image file paths
image_files = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.jpg')]

# Reverse the label map to get class names
reverse_label_map = {0: 'Istanbul',1: 'Ankara', 2: 'Izmir'}

# Prepare a list for predictions
predictions = []

for file_path in image_files:
    img = preprocess_image(file_path)
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)  # Get prediction
    predicted_label = np.argmax(prediction, axis=1)[0]  # Get the class index
    class_name = reverse_label_map[predicted_label]  # Convert to class name
    file_name = os.path.basename(file_path)  # Extract only the file name
    predictions.append((file_name, class_name))

# Convert to DataFrame
predictions_df = pd.DataFrame(predictions, columns=['filename', 'label'])

# Save to CSV
predictions_df.to_csv('predicted_labels.csv', index=False)

print("Predictions saved to predicted_labels.csv")