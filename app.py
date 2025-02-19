from flask import Flask, render_template, request
import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
import numpy as np

app = Flask(__name__)

# Load trained PyTorch model
model = torch.load("model/caption_model.pth", map_location=torch.device("cpu"))
model.eval()

# Load tokenizer
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_length = 35  # Max caption length

# Image processing (similar to InceptionV3 preprocessing)
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Function to extract image features
def extract_features(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model.encoder(img)  # Assuming Encoder-Decoder architecture
    return features

# Generate caption from image
def generate_caption(image_path):
    features = extract_features(image_path)
    input_text = ["startseq"]
    
    for _ in range(max_length):
        seq = [tokenizer[word] for word in input_text if word in tokenizer]
        seq_tensor = torch.tensor(seq).unsqueeze(0)

        with torch.no_grad():
            output = model.decoder(features, seq_tensor)
        
        predicted_index = torch.argmax(output, dim=1).item()
        word = tokenizer.get(predicted_index, "")

        if word == "endseq":
            break
        input_text.append(word)
    
    return " ".join(input_text[1:])  # Remove "startseq"

# Homepage
@app.route("/")
def home():
    return render_template("index.html")

# Handle image upload and captioning
@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return "No image uploaded"

    image = request.files["image"]
    image_path = "static/uploads/" + image.filename
    image.save(image_path)

    caption = generate_caption(image_path)

    return render_template("index.html", image_path=image_path, caption=caption)

if __name__ == "__main__":
    app.run(debug=True)
