import os
import boto3
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import multiprocessing
from functools import partial
import argparse
import tempfile
import shutil
from dotenv import load_dotenv

def download_images_from_s3(bucket_name, prefix, local_dir):
    """
    Download images from an S3 bucket to a local directory
    """
    print(f"Connecting to S3 bucket: {bucket_name}")
    s3_client = boto3.client('s3')
    os.makedirs(local_dir, exist_ok=True)
    print(f"Listing objects in {bucket_name}/{prefix}")
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if 'Contents' not in response:
        print(f"No objects found in {bucket_name}/{prefix}")
        return []
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    image_objects = [obj for obj in response['Contents']
                     if obj['Key'].lower().endswith(image_extensions)]
    print(f"Found {len(image_objects)} images in S3 bucket")
    downloaded_files = []
    for obj in tqdm(image_objects, desc="Downloading images"):
        filename = os.path.basename(obj['Key'])
        local_path = os.path.join(local_dir, filename)
        s3_client.download_file(bucket_name, obj['Key'], local_path)
        downloaded_files.append(filename)
    print(f"Downloaded {len(downloaded_files)} images to {local_dir}")
    return downloaded_files

# Global placeholders in worker processes
device = torch.device('cpu')
mtcnn: MTCNN = None
resnet: InceptionResnetV1 = None

def init_worker(min_face_size, thresholds, factor):
    """Initialize MTCNN and ResNet once per worker."""
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
    resnet = InceptionResnetV1(pretrained='vggface2').eval()


def process_batch(batch_files, input_dir, faces_dir):
    """
    Worker function: detect and embed all faces in batch_files,
    returning a dict of results.
    """
    global mtcnn, resnet
    batch_results = {}
    all_crops = []
    crop_meta = []

    # Step 1: detect and crop
    for image_file in batch_files:
        try:
            img = Image.open(os.path.join(input_dir, image_file)).convert('RGB')
            boxes, probs = mtcnn.detect(img)
            if boxes is None:
                continue
            crops = mtcnn.extract(img, boxes, save_path=None)
            for i, (box, crop) in enumerate(zip(boxes, crops)):
                crop_meta.append((image_file, i, box, float(probs[i]) if probs is not None else None))
                all_crops.append(crop.unsqueeze(0))
        except Exception as e:
            print(f"Error detecting faces in {image_file}: {e}")

    if not all_crops:
        return batch_results

    # Step 2: batch embed
    batch_tensor = torch.cat(all_crops, dim=0)
    embeddings = resnet(batch_tensor).cpu().numpy()

    # Step 3: save embeddings
    for idx, (image_file, face_idx, box, conf) in enumerate(crop_meta):
        image_name = os.path.splitext(image_file)[0]
        face_id = f"{image_name}_face_{face_idx+1}"
        emb = embeddings[idx]
        region = {
            'x': int(box[0]), 'y': int(box[1]),
            'w': int(box[2] - box[0]), 'h': int(box[3] - box[1]),
            'confidence': conf
        }
        data = {
            'embedding': emb,
            'region': region,
            'source_image': image_file,
            'face_index': face_idx,
            's3_source': True
        }
        out_path = os.path.join(faces_dir, f"{face_id}.pkl")
        with open(out_path, 'wb') as f:
            pickle.dump(data, f)
        batch_results.setdefault(image_file, {
            'face_regions': [], 'face_embeddings': [], 'face_ids': []
        })
        batch_results[image_file]['face_regions'].append(region)
        batch_results[image_file]['face_embeddings'].append(emb)
        batch_results[image_file]['face_ids'].append(face_id)

    return batch_results


def process_images(input_dir, output_dir, batch_size=8):
    os.makedirs(output_dir, exist_ok=True)
    faces_dir = os.path.join(output_dir, "faces")
    os.makedirs(faces_dir, exist_ok=True)

    image_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.gif','.tiff'))]
    batches = [image_files[i:i+batch_size]
               for i in range(0, len(image_files), batch_size)]

    total_faces = 0
    num_workers = max(1, multiprocessing.cpu_count() - 1)

    with multiprocessing.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(20, [0.6,0.7,0.7], 0.709)
    ) as pool:
        worker_fn = partial(process_batch, input_dir=input_dir, faces_dir=faces_dir)
        for result in tqdm(pool.imap_unordered(worker_fn, batches),
                           total=len(batches), desc="Parallel batches"):
            for img, info in result.items():
                total_faces += len(info['face_ids'])

    # Save combined embeddings if needed
    print(f"Processed {len(image_files)} images, found {total_faces} faces")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser("Download images from S3 and generate face embeddings (parallel)")
    parser.add_argument('--bucket', required=False)
    parser.add_argument('--prefix', required=False)
    parser.add_argument('--output-dir', default='embeds/s3_faces')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--keep-images', action='store_true')
    parser.add_argument('--images-dir', type=str)
    args = parser.parse_args()

    bucket = args.bucket or os.getenv('S3_BUCKET')
    prefix = args.prefix or os.getenv('S3_PREFIX')
    if not bucket or not prefix:
        print("Error: S3 bucket and prefix must be provided")
        return

    if args.keep_images and args.images_dir:
        images_dir = args.images_dir
        os.makedirs(images_dir, exist_ok=True)
        temp_dir = None
    else:
        temp_dir = tempfile.mkdtemp()
        images_dir = temp_dir

    try:
        download_images_from_s3(bucket, prefix, images_dir)
        process_images(images_dir, args.output_dir, args.batch_size)
    finally:
        if temp_dir and not args.keep_images:
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    main()