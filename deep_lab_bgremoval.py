# deep_lab_bgremoval.py
import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image

# Load pretrained model once
def load_model():
    model = torch.hub.load("pytorch/vision", "deeplabv3_resnet101", pretrained=True)
    model.eval()
    return model

# Perform background removal
def remove_background_deeplab(pil_image):
    """
    Removes background using DeepLabV3 model (semantic segmentation).
    Input: PIL Image
    Output: RGBA Image with transparent background
    """
    model = load_model()

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(pil_image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    mask = output.argmax(0).byte().cpu().numpy()

    # Everything except background (class 0)
    mask = (mask != 0).astype(np.uint8) * 255

    np_img = np.array(pil_image)
    rgba = cv2.cvtColor(np_img, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = mask  # Alpha channel from mask

    return Image.fromarray(rgba)
