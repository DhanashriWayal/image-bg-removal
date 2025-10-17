# app.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
import torch
import torchvision.transforms as T

st.set_page_config(page_title="Image Background Remover", layout="centered")

# Load DeepLabV3 model once
@st.cache_resource
def load_model():
    model = torch.hub.load("pytorch/vision", "deeplabv3_resnet101", pretrained=True)
    model.eval()
    return model

model = load_model()

def remove_background_deeplab(image):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    mask = output.argmax(0).byte().cpu().numpy()
    mask = (mask != 0).astype(np.uint8) * 255

    # Convert to RGBA (transparent background)
    np_img = np.array(image)
    rgba = cv2.cvtColor(np_img, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = mask
    return rgba

st.title("🧠 Image Background Remover")
st.write("Upload any image to remove its background using DeepLabV3 model.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)
    st.write("Processing... Please wait.")

    result = remove_background_deeplab(image)
    st.image(result, caption="Background Removed", use_column_width=True)

    # Save output for download
    result_img = Image.fromarray(result)
    buf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    result_img.save(buf.name)
    with open(buf.name, "rb") as file:
        st.download_button("⬇️ Download Result", data=file, file_name="output.png", mime="image/png")
