import torch
from model_architecture import EncoderDecoder  # Import your model architecture

# Load trained model
model = EncoderDecoder()
model.load_state_dict(torch.load("model/caption_model.pth", map_location=torch.device("cpu")))
model.eval()

print("PyTorch Model Loaded Successfully!")