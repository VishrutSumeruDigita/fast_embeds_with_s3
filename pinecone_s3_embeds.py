import os
import pickle
import torch
import pinecone
from pathlib import Path
from dotenv import load_dotenv
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from tqdm import tqdm
import multiprocessing
from functools import partial

# Load env
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV     = os.getenv("PINECONE_ENV", "local")
INDEX_NAME       = os.getenv("PINECONE_INDEX_NAME", "face-embeddings")

# --- Initialize Pinecone ---
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(
        name=INDEX_NAME,
        dimension=512,
        metric="cosine"
    )
index = pinecone.Index(INDEX_NAME)

# --- CUDA device & global models ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn: MTCNN       = None
resnet: InceptionResnetV1 = None

def init_worker(min_face_size, thresholds, factor):
    """Initialize in each worker."""
    global mtcnn, resnet
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
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

def process_batch(batch_files, input_dir):
    global mtcnn, resnet, index
    to_upsert = []

    for image_file in batch_files:
        img = Image.open(os.path.join(input_dir, image_file)).convert("RGB")
        boxes, probs = mtcnn.detect(img)
        if boxes is None:
            continue

        crops = mtcnn.extract(img, boxes, save_path=None)
        batch_tensor = torch.cat([c.unsqueeze(0).to(device) for c in crops], dim=0)
        embeddings = resnet(batch_tensor).cpu().numpy()

        for i, (box, conf, emb) in enumerate(zip(boxes, probs, embeddings)):
            image_name = os.path.splitext(image_file)[0]
            face_id = f"{image_name}_face_{i+1}"
            # metadata can hold JSON-friendly fields
            meta = {
                "source_image": image_file,
                "face_index"  : i,
                "region"      : {
                    "x": int(box[0]), "y": int(box[1]),
                    "w": int(box[2]-box[0]), "h": int(box[3]-box[1]),
                    "confidence": float(conf) if conf is not None else None
                }
            }
            to_upsert.append({
                "id"       : face_id,
                "values"   : emb.tolist(),
                "metadata" : meta
            })

    # push this batch to Pinecone
    if to_upsert:
        index.upsert(vectors=to_upsert)

def process_images(input_dir, batch_size=8):
    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.png','.jpg','.jpeg'))
    ]
    batches = [
        image_files[i:i+batch_size]
        for i in range(0, len(image_files), batch_size)
    ]

    num_workers = max(1, multiprocessing.cpu_count() - 1)
    with multiprocessing.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(20, [0.6,0.7,0.7], 0.709)
    ) as pool:
        worker = partial(process_batch, input_dir=input_dir)
        for _ in tqdm(pool.imap_unordered(worker, batches), total=len(batches)):
            pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",  default="test_images")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    process_images(args.input_dir, args.batch_size)
