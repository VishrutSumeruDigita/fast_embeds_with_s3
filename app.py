import streamlit as st
import requests
import json
from PIL import Image
import os

# === Configuration ===
API_URL = "http://localhost:8000"  # Your FastAPI endpoint
IMAGE_DIR = "test_images"          # Local directory with images used by FastAPI

# === Streamlit UI Setup ===
st.set_page_config(page_title="Face Search Dashboard", layout="centered")
st.title("🧠 Face Recognition API Tester")

tab1, tab2 = st.tabs(["📤 Embed Face", "🔍 Search Similar"])

# === Tab 1: Embed Face ===
with tab1:
    st.subheader("Upload an image to embed and index")
    embed_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="embed")

    if st.button("Submit to /embed") and embed_file:
        with st.spinner("Sending to /embed endpoint..."):
            response = requests.post(
                f"{API_URL}/embed",
                files={"image": (embed_file.name, embed_file, embed_file.type)}
            )
            if response.ok:
                st.success("✅ Success! Faces embedded.")
                st.json(response.json())
            else:
                st.error("❌ Failed to embed image.")
                st.text(response.text)

# === Tab 2: Search Similar ===
with tab2:
    st.subheader("Upload a query image to search similar faces")
    search_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="search")

    if st.button("Submit to /search") and search_file:
        with st.spinner("Sending to /search endpoint..."):
            response = requests.post(
                f"{API_URL}/search",
                files={"image": (search_file.name, search_file, search_file.type)}
            )

            if response.ok:
                st.success("✅ Matches found!")
                results = response.json()
                st.json(results)

                if "matches" in results:
                    for match in results["matches"]:
                        img_name = match.get("image_name")
                        score = match.get("score")
                        box = match.get("box")

                        st.markdown(f"**Image:** `{img_name}` | **Score:** `{score:.4f}`")
                        st.markdown(f"**Box:** `{box}`")

                        img_path = os.path.join(IMAGE_DIR, img_name)
                        if os.path.exists(img_path):
                            img = Image.open(img_path)
                            st.image(img, caption=f"{img_name}", use_column_width=True)
                        else:
                            st.warning(f"⚠️ Image not found locally: `{img_path}`")
            else:
                st.error("❌ Search failed.")
                st.text(response.text)

# === Sidebar Info ===
st.sidebar.title("ℹ️ App Info")
st.sidebar.markdown("Test endpoints from your FastAPI-based face search service using image uploads.")
st.sidebar.markdown("Built with ❤️ using Streamlit.")
