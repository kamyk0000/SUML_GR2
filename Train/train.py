# %%
import argparse
import os

import azureml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from azureml.core import Dataset, ScriptRunConfig, Workspace, Datastore, Run
from azureml.core.compute import AmlCompute
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import mlflow.pytorch

# %%
# Loading data
ws = Workspace.from_config()
datastore = Datastore.register_azure_blob_container(
    workspace=ws,
    datastore_name="food11",
    container_name="food11-data",
    account_name="food11",  # nazwa konta magazynu
    account_key="FzqK+ax1QXlmSW3lwnHVUC7e+8VjWm/c5h+dX8FZXUJ4Xdyauin6pn+mmnXcxkHJZZPc0sUTr3KM+ASttGJVaQ==",
    create_if_not_exists=False
)
datastore = Datastore.get(ws, "food11")
data_path_on_blob = [(datastore, 'food11-data/')]

food11_dataset = Dataset.File.from_files(path=data_path_on_blob)

food11_dataset = food11_dataset.register(workspace=ws,
                                         name='food11_dataset',
                                         description='Dataset obrazków Food11 z kontenera',
                                         create_new_version=True)

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default=None, help="Ścieżka do zamontowanego datasetu (opcjonalna lokalnie)")
args, _ = parser.parse_known_args()

# %%
# CONFIG
if args.data_path is None:
    DATA_DIR = "../Data/"
else:
    DATA_DIR = args.data_path
MODEL_PTH = "Model/food11_resnet18.pth"
MODEL_ONNX = "Model/food11_resnet18.onnx"
METRICS_PATH = "../Results/metrics.txt"
CONF_MATRIX_IMG = "Results/model_results.png"
BATCH_SIZE = 32
EPOCHS = 2
NUM_CLASSES = 11
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# TRANSFORMS
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# LOADERS
train_loader = DataLoader(
    datasets.ImageFolder(os.path.join(DATA_DIR, "training"), transform=transform),
    batch_size=BATCH_SIZE, shuffle=True)

val_loader = DataLoader(
    datasets.ImageFolder(os.path.join(DATA_DIR, "validation"), transform=transform),
    batch_size=BATCH_SIZE)

eval_loader = DataLoader(
    datasets.ImageFolder(os.path.join(DATA_DIR, "evaluation"), transform=transform),
    batch_size=BATCH_SIZE)

# %%
# MODEL
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# %%
# TRAIN
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{EPOCHS} complete.")


# %%
# EVALUATE FUNCTION
def evaluate(loader, label="Validation"):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"{label} → Accuracy: {acc * 100:.2f}%  F1: {f1:.2f}")
    return acc, f1, y_true, y_pred


# Run evaluation on validation set
val_acc, val_f1, y_true, y_pred = evaluate(val_loader, "Validation")

# Run evaluation on holdout test set
eval_acc, eval_f1, _, _ = evaluate(eval_loader, "Evaluation")

# %%
# SAVE METRICS
with open(METRICS_PATH, "w") as f:
    f.write(f"Validation Accuracy = {val_acc * 100:.2f}%, F1 = {val_f1:.2f}\n")
    f.write(f"Evaluation Accuracy = {eval_acc * 100:.2f}%, F1 = {eval_f1:.2f}")

# %%
# SAVE CONFUSION MATRIX
class_names = train_loader.dataset.classes
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(xticks_rotation=45)
plt.tight_layout()
plt.savefig(CONF_MATRIX_IMG, dpi=120)
plt.close()

# %%
# SAVE MODEL
torch.save(model.state_dict(), MODEL_PTH)
print("Saved PyTorch model to:", MODEL_PTH)

# %%
# EXPORT TO ONNX
dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
torch.onnx.export(
    model, dummy_input, MODEL_ONNX,
    input_names=["input"], output_names=["output"],
    export_params=True, opset_version=11
)
print("Exported ONNX model to:", MODEL_ONNX)

# %%
# LOG TO MLFLOW
dummy_input2 = torch.randn(1, 3, 224, 224).to(DEVICE).numpy()
with mlflow.start_run(run_name="resnet18-food11"):
    mlflow.pytorch.log_model(
        model,
        "../Model",
        registered_model_name="Food11ResNet",
        input_example=dummy_input2
    )

    mlflow.log_metric("val_accuracy", val_acc)
    mlflow.log_metric("val_f1", val_f1)
    mlflow.log_metric("eval_accuracy", eval_acc)
    mlflow.log_metric("eval_f1", eval_f1)

    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("num_classes", NUM_CLASSES)
    mlflow.log_artifact(CONF_MATRIX_IMG)

# %%
# LOG TO AZUREML
ws = azureml.core.Workspace.from_config()

food11_dataset = Dataset.get_by_name(ws, name="food11_dataset")

compute_name = "food-cluster"

compute_config = AmlCompute.provisioning_configuration(
    vm_size="STANDARD_DS11_V2",
    max_nodes=2,
    idle_seconds_before_scaledown=300
)

if compute_name not in ws.compute_targets:
    compute_target = azureml.core.ComputeTarget.create(ws, compute_name, compute_config)
    compute_target.wait_for_completion(show_output=True)
else:
    compute_target = ws.compute_targets[compute_name]

compute = azureml.core.ComputeTarget(workspace=ws, name=compute_name)

env = azureml.core.Environment.from_pip_requirements(name="food-env", file_path="Train/requirements.txt")

mount_context = food11_dataset.as_mount()

src = ScriptRunConfig(
    source_directory="./Train",
    script="train.py",
    compute_target=compute_target,
    environment=env,
    arguments=['--data-path', food11_dataset.as_mount()]
)

experiment = azureml.core.Experiment(ws, "Food11Training")
run = experiment.submit(src)

run.log("epochs", EPOCHS)
run.log("batch_size", BATCH_SIZE)
run.log("num_classes", NUM_CLASSES)
run.log("val_accuracy", val_acc)
run.log("val_f1", val_f1)
run.log("eval_accuracy", eval_acc)
run.log("eval_f1", eval_f1)

run.upload_file(name="outputs/confusion_matrix.png", path_or_stream=CONF_MATRIX_IMG)

import joblib
joblib.dump(model, "outputs/model.pkl")

run.wait_for_completion(show_output=True)

mount_context.stop()
