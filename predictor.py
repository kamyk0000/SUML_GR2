import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class FoodClassifier:
    def __init__(self, model_path="Model/food11.onnx", session=None):
        if session is not None:
            self.session = session
        else:
            self.session = ort.InferenceSession(model_path)

        self.input_name = self.session.get_inputs()[0].name

        self.class_names = [
            'Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food',
            'Meat', 'Noodles-Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable-Fruit'
        ]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image: Image.Image):
        img = self.transform(image).unsqueeze(0).numpy().astype(np.float32)
        pred = self.session.run(None, {self.input_name: img})[0]
        probs = np.exp(pred) / np.sum(np.exp(pred))
        top_idx = np.argmax(probs)
        return self.class_names[top_idx], float(probs[0][top_idx])
