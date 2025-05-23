import os
import argparse
import torch
import multiprocessing
from functools import partial
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN, InceptionResnetV1
from elasticsearch import Elasticsearch

# Load environment variables
load_dotenv()
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
INDEX_NAME = os.getenv("ES_INDEX", "face_embeddings")

# Globals for workers
device = torch.device("cuda")
mtcnn: MTCNN = None
resnet: InceptionResnetV1 = None
es: Elasticsearch = None


def create_index(es_client: Elasticsearch):
    """Create Elasticsearch index with dense_vector mapping."""
    if es_client.indices.exists(index=INDEX_NAME):
        return
    es_client.indices.create(
        index=INDEX_NAME,
        body={
            "mappings": {
                "properties": {
                    "image_name": {"type": "keyword"},
                    "embeds": {
                        "type": "dense_vector",
                        "dims": 512,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        }
    )


def init_worker(min_face_size, thresholds, factor):
    """Initialize MTCNN, ResNet, and ES client per worker."""
    global mtcnn, resnet, es
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)

    mtcnn = MTCNN(
        keep_all=True,
        min_face_size=min_face_size,
        thresholds=thresholds,
        factor=factor,
        post_process=True,
        device=device,
    )
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    es = Elasticsearch(ES_HOST)


def process_batch(batch_files, input_dir):
    """Detect faces, embed, and index into Elasticsearch."""
    global mtcnn, resnet, es
    uploads = []

    for image_file in batch_files:
        path = os.path.join(input_dir, image_file)
        try:
            img = Image.open(path).convert('RGB')
            boxes, probs = mtcnn.detect(img)
            if boxes is None:
                continue

            crops = mtcnn.extract(img, boxes, save_path=None)
            batch_tensor = torch.cat([c.unsqueeze(0).to(device) for c in crops], dim=0)
            embeddings = resnet(batch_tensor).cpu().numpy()

            for i, (box, conf, emb) in enumerate(zip(boxes, probs, embeddings)):
                image_name = Path(image_file).stem
                face_id = f"{image_name}_face_{i+1}"
                # Prepare document for ES
                doc = {
                    "image_name": image_file,
                    "embeds": emb.tolist()
                }
                # Index into ES
                es.index(index=INDEX_NAME, id=face_id, document=doc)
        except Exception as e:
            print(f"Error processing {image_file}: {e}")


def process_images(input_dir, batch_size=8):
    files = [f for f in os.listdir(input_dir)
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    batches = [files[i:i+batch_size] for i in range(0, len(files), batch_size)]

    num_workers = max(1, multiprocessing.cpu_count() - 1)
    with multiprocessing.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(20, [0.6, 0.7, 0.7], 0.709)
    ) as pool:
        worker = partial(process_batch, input_dir=input_dir)
        for _ in tqdm(pool.imap_unordered(worker, batches), total=len(batches)):
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="test_images")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    # Create ES index if missing
    es_client = Elasticsearch(ES_HOST)
    create_index(es_client)

    # Process images and index embeddings
    process_images(args.input_dir, args.batch_size)


if __name__ == "__main__":
    main()