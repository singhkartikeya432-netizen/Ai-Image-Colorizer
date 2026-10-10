import os
import urllib.request
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="AI Image Colorizer", page_icon="🎨", layout="wide")

st.title("AI Image Colorizer")
st.write("Upload a black-and-white image and let AI predict its colors.")

MODEL_DIR = "colorization_model"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_URLS = {
    "colorization_deploy_v2.prototxt":
        "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_deploy_v2.prototxt",
    "pts_in_hull.npy":
        "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/resources/pts_in_hull.npy",
    "colorization_release_v2.caffemodel":
        "https://huggingface.co/chen0414/colorization_release_v2/resolve/main/colorization_release_v2.caffemodel?download=true"
}

@st.cache_resource
def load_model():
    for filename, url in MODEL_URLS.items():
        path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(path):
            with st.spinner(f"Downloading {filename}..."):
                urllib.request.urlretrieve(url, path)

    net = cv2.dnn.readNetFromCaffe(
        os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt"),
        os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")
    )

    points = np.load(os.path.join(MODEL_DIR, "pts_in_hull.npy"))
    points = points.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId("class8_ab")).blobs = [
        points.astype(np.float32)
    ]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [
        np.full((1, 313), 2.606, dtype=np.float32)
    ]
    return net

def colorize_image(image_rgb, net):
    image_float = image_rgb.astype(np.float32) / 255.0
    image_lab = cv2.cvtColor(image_float, cv2.COLOR_RGB2LAB)
    L = image_lab[:, :, 0]

    L_resized = cv2.resize(L, (224, 224)) - 50
    net.setInput(cv2.dnn.blobFromImage(L_resized))

    ab = net.forward("class8_ab")[0].transpose(1, 2, 0)
    ab = cv2.resize(ab, (image_rgb.shape[1], image_rgb.shape[0]))

    colorized_lab = np.concatenate([L[:, :, None], ab], axis=2)
    result = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2RGB)
    return np.clip(result * 255, 0, 255).astype(np.uint8)

uploaded_file = st.file_uploader(
    "Choose an image", type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image.thumbnail((512, 512))
    image_rgb = np.array(image)

    if st.button("Colorize Image"):
        try:
            net = load_model()
            with st.spinner("Colorizing image..."):
                result = colorize_image(image_rgb, net)

            left, right = st.columns(2)
            with left:
                st.subheader("Original")
                st.image(image_rgb, use_container_width=True)
            with right:
                st.subheader("Colorized")
                st.image(result, use_container_width=True)

            buffer = io.BytesIO()
            Image.fromarray(result).save(buffer, format="PNG")
            st.download_button(
                "Download Colorized Image",
                data=buffer.getvalue(),
                file_name="colorized_image.png",
                mime="image/png"
            )
        except Exception as error:
            st.error(f"Colorization failed: {error}")
