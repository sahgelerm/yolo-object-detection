from ultralytics import YOLO

def predict(image_path):
    model = YOLO("best.pt")
    results = model(image_path)
    results.show()

if __name__ == "__main__":
    predict("test.jpg")
