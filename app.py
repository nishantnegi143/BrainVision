from flask import Flask, request, render_template
import os
import torch
import numpy as np
from PIL import Image
import pickle
import time

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
with open("vit_model.pkl", "rb") as f:
    model = pickle.load(f)

model.eval()

classes = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']


# ================== MRI VALIDATION ==================
def is_valid_mri(image):
    img = np.array(image)

    if len(img.shape) == 3:
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        if np.mean(np.abs(r - g)) > 10 or np.mean(np.abs(r - b)) > 10:
            return False

    if np.std(img) < 15:
        return False

    return True


# ================== PREDICTION ==================
def predict(image):
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        start = time.time()

        outputs = model(img)          # ✅ FIX
        logits = outputs.logits       # ✅ FIX

        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        end = time.time()

    pred_idx = np.argmax(probs)
    pred_label = classes[pred_idx]
    confidence = probs[pred_idx]

    return pred_label, confidence, probs, pred_idx, round(end - start, 3)


# ================== REPORT ==================
def generate_report(pred, conf):
    if "Non" in pred:
        return "Low Risk", "No significant cognitive impairment detected."
    elif "Very Mild" in pred:
        return "Early Stage", "Minor signs observed."
    elif "Mild" in pred:
        return "Moderate Risk", "Clinical evaluation advised."
    else:
        return "High Risk", "Immediate medical attention recommended."


# ================== ROUTES ==================
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["image"]

        if file.filename == "":
            return render_template("index.html", error="No file selected")

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        image = Image.open(filepath).convert("RGB")

        if not is_valid_mri(image):
            return render_template("index.html", error="Invalid MRI image")

        pred, conf, probs, idx, t = predict(image)
        risk, note = generate_report(pred, conf)

        return render_template(
            "index.html",
            result=True,
            prediction=pred,
            confidence=round(conf * 100, 2),
            probs=[round(p * 100, 2) for p in probs],
            idx=idx,
            risk=risk,
            note=note,
            time=t,
            classes=classes
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run()