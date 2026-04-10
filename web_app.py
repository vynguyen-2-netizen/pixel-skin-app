import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="Environmental Pixel Skin", layout="centered")
st.title("Environmental Pixel Skin")
st.write("Live camera feed translated into a simplified pixel surface.")

pixel_size = st.slider("Pixel Size", 4, 40, 12)
num_colors = st.slider("Number of Colors", 2, 12, 6)

def pixelate_image(image, block_size):
    h, w = image.shape[:2]
    small_w = max(1, w // block_size)
    small_h = max(1, h // block_size)
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return pixelated

def reduce_colors_fast(image, levels):
    step = max(1, 256 // levels)
    return ((image // step) * step).astype(np.uint8)

class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # 🔥 Pixel first (keep structure)
        img = pixelate_image(img, pixel_size)

        # 🔥 Then reduce colors (not too aggressive)
        img = reduce_colors_fast(img, num_colors)

        return img

webrtc_streamer(
    key="pixel-skin",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    async_processing=True,
)
