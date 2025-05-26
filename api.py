from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
import numpy as np
import cv2
from elasticsearch import Elasticsearch
from insightface.app import FaceAnalysis
from dotenv import load_dotenv
import torch

# Load .env
load_dotenv()

# Configs
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
INDEX_NAME = os.getenv("INDEX_NAME", "face_embeddings")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\ud83d\udd25 Using device: {device}")

# Initialize app & Elasticsearch
app = FastAPI()
es = Elasticsearch(ES_HOST, verify_certs=False)

# Initialize FaceAnalysis model
face_app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider'])
face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)

# Create index if it doesn't exist
def create_index():
    if es.indices.exists(index=INDEX_NAME):
        return

    es.indices.create(
        index=INDEX_NAME,
        body={
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {
                "properties": {
                    "image_name": {"type": "keyword"},
                    "embeds": {"type": "dense_vector", "dims": 512, "index": True, "similarity": "cosine"},
                    "box": {"type": "dense_vector", "dims": 4}
                }
            }
        }
    )

create_index()

@app.post("/embed")
async def embed_face(image: UploadFile = File(...)):
    try:
        content = await image.read()
        np_img = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_app.get(img_rgb)

        if not faces:
            raise HTTPException(status_code=404, detail="No face detected")

        responses = []
        for i, face in enumerate(faces):
            emb = face.normed_embedding
            box = face.bbox
            doc_id = f"{image.filename}_face_{i+1}"

            es.index(index=INDEX_NAME, id=doc_id, document={
                "image_name": image.filename,
                "embeds": emb.tolist(),
                "box": box.tolist()
            })
            responses.append({"face_id": doc_id, "box": box.tolist()})

        return {"status": "success", "faces": responses}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_similar(image: UploadFile = File(...)):
    try:
        content = await image.read()
        np_img = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_app.get(img_rgb)

        if not faces:
            raise HTTPException(status_code=404, detail="No face detected")

        emb = faces[0].normed_embedding  # Use the first face only

        query = {
            "size": 5,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embeds') + 1.0",
                        "params": {"query_vector": emb.tolist()}
                    }
                }
            }
        }

        res = es.search(index=INDEX_NAME, body=query)
        hits = [
            {
                "image_name": hit["_source"]["image_name"],
                "score": hit["_score"],
                "box": hit["_source"].get("box")
            } for hit in res["hits"]["hits"]
        ]

        return {"matches": hits}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn this_file_name:app --reload
