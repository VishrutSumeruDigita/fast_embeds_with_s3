import os
import argparse
import torch
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from elasticsearch import Elasticsearch
from insightface.app import FaceAnalysis

# Load .env
load_dotenv()

# Configs
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
INDEX_NAME = os.getenv("INDEX_NAME", "face_embeddings")
MAX_WORKERS = os.cpu_count() or 8

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}")

# Connect to Elasticsearch
es = Elasticsearch(ES_HOST, verify_certs=False)

# Create index if not exists
def create_index():
    if es.indices.exists(index=INDEX_NAME):
        print(f"✅ Index '{INDEX_NAME}' already exists.")
        return

    print(f"🚀 Creating index '{INDEX_NAME}' with vector search enabled...")

    es.indices.create(
        index=INDEX_NAME,
        body={
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "image_name": { "type": "keyword" },
                    "embeds": {
                        "type": "dense_vector",
                        "dims": 512,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "box": {
                        "type": "dense_vector",
                        "dims": 4
                    }
                }
            }
        }
    )

    print(f"✅ Index '{INDEX_NAME}' created successfully with 'box' support.")

# Process single image
def process_single_image(face_app, input_dir, image_file):
    try:
        path = os.path.join(input_dir, image_file)
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ Could not read image: {image_file}")
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_app.get(img)

        if not faces:
            return

        for i, face in enumerate(faces):
            emb = face.normed_embedding
            box = face.bbox  # [x1, y1, x2, y2]

            face_id = f"{Path(image_file).stem}_face_{i+1}"
            doc = {
                "image_name": image_file,
                "embeds": emb.tolist(),
                "box": box.tolist()
            }
            es.index(index=INDEX_NAME, id=face_id, document=doc)

    except Exception as e:
        print(f"⚠️ Error processing {image_file}: {e}")

# Face embedding pipeline using InsightFace with threading
# Face embedding pipeline using InsightFace with threading
def process_images(input_dir: str):
    # Use 'antelopev2' for best accuracy + InsightFace support
    face_app = FaceAnalysis(
        name='buffalo_l',  # ✅ Model name
        providers=['CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider']
    )
    face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"🔍 Found {len(files)} image(s) in '{input_dir}'")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(tqdm(executor.map(lambda f: process_single_image(face_app, input_dir, f), files), total=len(files)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="test_images", help="Path to image folder")
    args = parser.parse_args()

    create_index()
    process_images(args.input_dir)

if __name__ == "__main__":
    main()
