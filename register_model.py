import mlflow.onnx
import onnx

mlflow.set_tracking_uri("http://localhost:8080")

mlflow.set_experiment("food11_exp")

onnx_model_path = "Model/food11.onnx"
onnx_model = onnx.load_model(onnx_model_path)

with mlflow.start_run():
    mlflow.onnx.log_model(
        onnx_model=onnx_model,
        artifact_path="onnx_model",
        registered_model_name="Food11ONNX"
    )
