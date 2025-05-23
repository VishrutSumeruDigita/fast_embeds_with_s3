"""

### Notes for commands and stuff to use in prod server for debug



# 1. Build your face-embedder image
docker build -t face-embedder:latest .

# 2. Run it with CUDA and your local Pinecone
docker run --gpus all \
  --rm \
  -v /path/to/your/images:/data/images \
  --env-file .env \
  face-embedder:latest




# 1. Pull Pinecone’s local emulator image
docker pull pinecone/pinecone:latest

# 2. Run it, exposing the default REST port 8100
docker run -d --name pinecone-local \
  -p 8100:8100 \
  pinecone/pinecone:latest




find . -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.gif" -o -iname "*.tiff" \) | wc -l

"""