import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import timm
import os
from gradio_client import utils as grc_utils

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ["normal", "pneumonia", "tuberculosis"]
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "models",
    "efficientnetv2s_best.pth",
)

# 2. Define Preprocessing (Must match training)
transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# 3. Load Model
def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    # Create model structure
    model = timm.create_model(
        "tf_efficientnetv2_s", pretrained=False, num_classes=len(class_names)
    )

    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                state_dict = checkpoint["model_state"]
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Running with random weights for debugging purposes.")
    else:
        print(
            f"Warning: Model file not found at {MODEL_PATH}. Running with random weights for testing."
        )

    model.to(device)
    model.eval()
    return model


def _patch_gradio_schema_utils():
    """Guard against bool JSON-schema nodes until upstream fix lands."""
    original = grc_utils.get_type

    def safe_get_type(schema):
        if isinstance(schema, bool):
            return "any" if schema else "never"
        return original(schema)

    if getattr(grc_utils.get_type, "__name__", "") != "safe_get_type":
        grc_utils.get_type = safe_get_type


_patch_gradio_schema_utils()

model = load_model()


# 4. Prediction Function
def predict_image(image):
    if image is None:
        return None

    # Preprocess
    try:
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]

        # Format Output {Class: Probability}
        return {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    except Exception as e:
        return {"Error": str(e)}


# 5. Launch Interface
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload Chest X-ray"),
    outputs=gr.Label(num_top_classes=3, label="Diagnosis Result"),
    title="🚑 AI Chest X-ray Assistant",
    description="Upload an X-ray image to detect: Normal, Pneumonia, or Tuberculosis.",
    examples=[],
)

if __name__ == "__main__":
    interface.launch(share=False, server_name="127.0.0.1", server_port=7860)
