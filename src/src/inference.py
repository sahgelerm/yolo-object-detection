import numpy as np
import cv2
from tensorflow.keras.models import load_model

# загрузка модели
model = load_model("my_model.keras")

# классы (важно: порядок должен совпадать с обучением)
CLASS_NAMES = [
    "heiko_dachy_5",
    "kokutsu_dachy",
    " dzenkutsu_dachy"
]


def preprocess_image(image_path, img_size=(128, 64)):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    img = cv2.resize(img, img_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img


def predict(image_path):
    img = preprocess_image(image_path)
    predictions = model.predict(img)

    class_id = np.argmax(predictions)
    confidence = np.max(predictions)

    return CLASS_NAMES[class_id], confidence


if __name__ == "__main__":
    image_path = "test.jpg"  # путь к изображению

    label, confidence = predict(image_path)

    print(f"Предсказание: {label}")
    print(f"Уверенность: {confidence:.2f}")
