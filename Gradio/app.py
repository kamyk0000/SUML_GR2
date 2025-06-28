import gradio as gr
from Model.predictor import FoodClassifier


def classify_food(image):
    if image is None:
        return f"🍽️ Please input an image first!"
    label, confidence = classifier.predict(image)
    return f"🍽️ Predicted Food Type: *{label}* ({confidence * 100:.2f}% confidence)"


classifier = FoodClassifier("../Model/food11.onnx")

interface = gr.Interface(
    fn=classify_food,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(),
    title="SUML Food11 Classifier",
    description="Input an image of food below to get the classification.",
).launch()

