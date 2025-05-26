import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from elasticsearch import Elasticsearch

# Constants
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
INDEX_NAME = os.getenv("INDEX_NAME", "face_embeddings")
IMAGE_DIR = os.getenv("IMAGE_DIR", "test_images")  # Path to local images

# Connect to Elasticsearch
es = Elasticsearch(ES_HOST, verify_certs=False)

# Generate a random 512-dim embedding vector as query (replace with real embed if needed)
query_vector = np.random.rand(512).tolist()

# Build search query
query = {
    "size": 1,
    "query": {
        "script_score": {
            "query": { "match_all": {} },
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embeds') + 1.0",
                "params": {
                    "query_vector": query_vector
                }
            }
        }
    }
}

# Search
print("🔍 Querying Elasticsearch for most similar image...")
response = es.search(index=INDEX_NAME, body=query)

# Parse result
if not response["hits"]["hits"]:
    print("❌ No matches found.")
    exit()

top_hit = response["hits"]["hits"][0]
image_name = top_hit["_source"]["image_name"]
score = top_hit["_score"]
image_path = os.path.join(IMAGE_DIR, image_name)

print(f"✅ Match found: {image_name} (Score: {score:.4f})")

# Show image
if os.path.exists(image_path):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(f"Top Match: {image_name}")
    plt.axis('off')
    plt.show()
else:
    print(f"⚠️ Image not found at: {image_path}")
