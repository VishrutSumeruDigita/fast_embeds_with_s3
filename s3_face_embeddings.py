import os
import boto3
import cv2
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
    
    Args:
        bucket_name (str): Name of the S3 bucket
        prefix (str): Prefix (folder path) within the bucket
        local_dir (str): Local directory to save downloaded images
    
    Returns:
        list: List of downloaded image filenames
    """
    print(f"Connecting to S3 bucket: {bucket_name}")
    s3_client = boto3.client('s3')
    
    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    # List objects in the bucket with the given prefix
    print(f"Listing objects in {bucket_name}/{prefix}")
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    
    if 'Contents' not in response:
        print(f"No objects found in {bucket_name}/{prefix}")
        return []
    
    # Filter for image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    image_objects = [obj for obj in response['Contents'] 
                    if obj['Key'].lower().endswith(image_extensions)]
    
    print(f"Found {len(image_objects)} images in S3 bucket")
    
    # Download each image
    downloaded_files = []
    for obj in tqdm(image_objects, desc="Downloading images"):
        # Get the filename from the S3 key
        filename = os.path.basename(obj['Key'])
        local_path = os.path.join(local_dir, filename)
        
        # Download the file
        s3_client.download_file(bucket_name, obj['Key'], local_path)
        downloaded_files.append(filename)
    
    print(f"Downloaded {len(downloaded_files)} images to {local_dir}")
    return downloaded_files

def process_batch(batch_files, input_dir, output_dir, mtcnn, resnet, face_counter):
    """Process a batch of images to extract face embeddings and save each unique face"""
    batch_results = {}
    
    for image_file in batch_files:
        try:
            # Load the image file
            image_path = os.path.join(input_dir, image_file)
            img = Image.open(image_path).convert('RGB')
            
            # Get image name without extension for face naming
            image_name = os.path.splitext(image_file)[0]
            
            # Detect faces using MTCNN
            boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
            
            if boxes is None or len(boxes) == 0:
                print(f"No faces found in {image_file}")
                continue
            
            # Get face crops from the image
            face_crops = mtcnn.extract(img, boxes, save_path=None)
            
            if face_crops is None:
                print(f"Failed to extract faces from {image_file}")
                continue
            
            # Store face regions and embeddings
            face_regions = []
            face_embeddings = []
            face_ids = []
            
            # Process each detected face
            for i, (box, face_crop) in enumerate(zip(boxes, face_crops)):
                try:
                    # Convert box coordinates to integers
                    box = box.astype(int).tolist()
                    
                    # Store face region
                    face_region = {
                        'x': box[0],
                        'y': box[1],
                        'w': box[2] - box[0],
                        'h': box[3] - box[1],
                        'confidence': float(probs[i]) if probs is not None else None
                    }
                    face_regions.append(face_region)
                    
                    # Get embedding using FaceNet
                    face_crop = face_crop.unsqueeze(0)
                    embedding = resnet(face_crop).numpy()[0]
                    
                    # Generate a unique ID for this face
                    face_id = f"{image_name}_face_{i+1}"
                    face_ids.append(face_id)
                    
                    # Save individual face embedding
                    face_data = {
                        'embedding': embedding,
                        'region': face_region,
                        'source_image': image_file,
                        'face_index': i,
                        's3_source': True
                    }
                    
                    # Save the face embedding to a separate file
                    face_file = os.path.join(output_dir, f"{face_id}.pkl")
                    with open(face_file, 'wb') as f:
                        pickle.dump(face_data, f)
                    
                    face_embeddings.append(embedding)
                    face_counter[0] += 1
                    
                except Exception as e:
                    print(f"Error getting embedding for face {i} in {image_file}: {str(e)}")
            
            # Store the results
            batch_results[image_file] = {
                'face_regions': face_regions,
                'face_embeddings': face_embeddings,
                'face_ids': face_ids
            }
            
            print(f"Processed {image_file} - Found {len(face_regions)} faces")
            
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
    
    return batch_results

def process_images(input_dir, output_dir, batch_size=4):
    """Process images in batches for better CPU utilization"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a directory for individual face embeddings
    faces_dir = os.path.join(output_dir, "faces")
    os.makedirs(faces_dir, exist_ok=True)
    
    # Force CPU usage
    device = torch.device('cpu')
    print("Using CPU for processing")
    
    # Enable PyTorch optimizations for CPU
    num_cpu_cores = multiprocessing.cpu_count()
    torch.set_num_threads(max(1, num_cpu_cores - 1))  # Use all CPU cores except one
    torch.set_grad_enabled(False)  # Disable gradient calculation for inference
    
    print(f"Using {torch.get_num_threads()} CPU threads for processing")
    
    # Initialize MTCNN for face detection with CPU optimizations
    mtcnn = MTCNN(
        keep_all=True,  # Keep all detected faces
        min_face_size=20,  # Minimum size of face to detect
        thresholds=[0.6, 0.7, 0.7],  # MTCNN thresholds
        factor=0.709,  # Scale factor
        post_process=True,  # Align faces
        device=device,
        select_largest=False  # Process all faces, not just the largest
    )
    
    # Initialize the FaceNet model for embeddings with CPU optimizations
    # Using VGGFace2 model which is more accurate
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    
    # Get all image files from the input directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
    
    print(f"Found {len(image_files)} images to process")
    
    # Create batches of images
    batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]
    print(f"Processing in {len(batches)} batches of size {batch_size}")
    
    # Counter for total faces processed (passed as reference)
    face_counter = [0]
    
    # Process each batch
    embeddings = {}
    for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
        print(f"\nProcessing batch {batch_idx+1}/{len(batches)}")
        batch_results = process_batch(batch, input_dir, faces_dir, mtcnn, resnet, face_counter)
        embeddings.update(batch_results)
    
    # Save the embeddings to a pickle file (contains all faces from all images)
    output_file = os.path.join(output_dir, 'face_embeddings.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings, f)
    
    # Create a metadata file with information about all faces
    metadata = {
        'total_images': len(image_files),
        'total_faces': face_counter[0],
        'face_embeddings_dir': faces_dir,
        'timestamp': Path(output_file).stat().st_mtime
    }
    
    metadata_file = os.path.join(output_dir, 'metadata.pkl')
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\nProcessing complete!")
    print(f"- Processed {len(image_files)} images")
    print(f"- Found {face_counter[0]} faces")
    print(f"- Embeddings saved to {output_file}")
    print(f"- Individual face embeddings saved in {faces_dir}")
    print(f"- Metadata saved to {metadata_file}")

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Download images from S3 and generate face embeddings')
    parser.add_argument('--bucket', type=str, help='S3 bucket name')
    parser.add_argument('--prefix', type=str, help='S3 prefix (folder path)')
    parser.add_argument('--output-dir', type=str, help='Output directory for embeddings')
    parser.add_argument('--batch-size', type=int, help='Batch size for processing')
    parser.add_argument('--keep-images', action='store_true', help='Keep downloaded images after processing')
    parser.add_argument('--images-dir', type=str, help='Directory to save downloaded images (if keep-images is True)')
    
    args = parser.parse_args()
    
    # Use environment variables as defaults if arguments are not provided
    bucket_name = args.bucket or os.getenv('S3_BUCKET')
    prefix = args.prefix or os.getenv('S3_PREFIX')
    output_dir = args.output_dir or os.getenv('OUTPUT_DIR', 'embeds/s3_faces')
    batch_size = args.batch_size or int(os.getenv('BATCH_SIZE', '4'))
    
    if not bucket_name or not prefix:
        print("Error: S3 bucket name and prefix must be provided either as arguments or in .env file")
        return
    
    # Create a temporary directory for downloaded images if not keeping them
    if args.keep_images and args.images_dir:
        images_dir = args.images_dir
        os.makedirs(images_dir, exist_ok=True)
        temp_dir = None
    else:
        temp_dir = tempfile.mkdtemp()
        images_dir = temp_dir
    
    try:
        print(f"Starting download from S3 bucket: {bucket_name}, prefix: {prefix}")
        download_images_from_s3(bucket_name, prefix, images_dir)
        
        # Process the downloaded images
        print(f"Starting face embedding generation")
        process_images(images_dir, output_dir, batch_size)
        
    finally:
        # Clean up temporary directory if we created one
        if temp_dir and not args.keep_images:
            print(f"Cleaning up temporary files")
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
