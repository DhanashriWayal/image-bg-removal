# deep_lab_bgremoval.py
import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image

@torch.no_grad()
def load_model():
    try:
        model = torch.hub.load(
            "pytorch/vision",
            "deeplabv3_mobilenet_v3_large",  # smaller model
            pretrained=True,
            force_reload=False  # prevent re-downloading every time
        )
        model.eval()
        return model
    except Exception as e:
        print("Error loading model:", e)
        return None


def remove_background_deeplab(pil_image):
    model = load_model()
    if model is None:
        raise RuntimeError("Model failed to load. Try reloading the app.")

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(pil_image).unsqueeze(0)
    output = model(input_tensor)['out'][0]
    mask = output.argmax(0).byte().cpu().numpy()
    mask = (mask != 0).astype(np.uint8) * 255

    np_img = np.array(pil_image)
    rgba = cv2.cvtColor(np_img, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = mask
    return Image.fromarray(rgba)
