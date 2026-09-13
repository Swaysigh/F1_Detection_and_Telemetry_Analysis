"""Run trained classifier on a cropped car image to predict driver/team."""
import torch


def load_model(checkpoint_path, device="cpu"):
    model = torch.load(checkpoint_path, map_location=device)
    model.eval()
    return model


def predict(model, image_tensor, class_names):
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0))
        pred_idx = logits.argmax(dim=1).item()
    return class_names[pred_idx]
