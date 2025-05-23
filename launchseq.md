"""

# 1. Build your face-embedder image
docker build -t face-embedder:latest .

# 2. Run it with CUDA and your local Pinecone
docker run --gpus all \
  --rm \
  -v /path/to/your/images:/data/images \
  --env-file .env \
  face-embedder:latest


"""