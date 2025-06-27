import onnxruntime as ort
from PIL import Image
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from Train.predictor import FoodClassifier


# python -m mlflow server --host 127.0.0.1 --port 8080

class MLflowClient:
    def __init__(self):
        mlflow.set_tracking_uri("http://localhost:8080")
        self.client = MlflowClient()

    def get_latest_session(self, model_name="Food11ONNX"):
        model_uri = "models:/Food11ONNX/" + str(len(self.client.search_model_versions(f"name='{model_name}'")))
        onnx_model = mlflow.onnx.load_model(model_uri)
        session = ort.InferenceSession(onnx_model.SerializeToString(), providers=["CPUExecutionProvider"])
        return session

    def predict(self, image, model_name="Food11ONNX"):
        session = self.get_latest_session(model_name)
        classifier = FoodClassifier(session=session)
        label, confidence = classifier.predict(image)
        print(f"🍽️ Predicted Food Type: **{label}** ({confidence * 100:.2f}% confidence)")
        return label, confidence
