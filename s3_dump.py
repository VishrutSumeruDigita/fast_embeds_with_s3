# download_images.py

import os
import boto3
from dotenv import load_dotenv

# List of S3 prefixes to download from
PREFIXES = [
    "Divinepic-11_FEB_2025_EVENING_DARSHAN_SECRETARIAT-ucpishn9",
    "Divinepic-12_FEB_EVENING_DARSHAN-9h1l5zh7",
    "Divinepic-14_FEB_2025_EVENING_DARSHAN_AT_RADHAKUNJ-fh2s7k3d",
    "Divinepic-15_FEB_2025_EVENING_DARSHAN-hav45tk5",
    "Divinepic-16_FEB_EVENING_DARSHAN_BADRIVISHAL-9ikeg4v4",
    "Divinepic-18_FEB_JIND_GREEN_ROOM_MOMENTS-9jpfe3mw",
    "Divinepic-18_FEB_KARNAL_GREEN_ROOM-01n0jt4x"
]

def download_images_from_s3(bucket_name, prefixes, local_dir):
    """
    Download all images from a list of S3 prefixes into a flat local directory
    """
    print(f"Downloading images from bucket {bucket_name} into {local_dir}")
    s3 = boto3.client('s3')
    os.makedirs(local_dir, exist_ok=True)

    for prefix in prefixes:
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                    # Avoid filename clashes
                    unique_name = key.replace("/", "_")
                    dest = os.path.join(local_dir, unique_name)
                    print(f"Downloading {key} to {dest}")
                    s3.download_file(bucket_name, key, dest)

    print("Download complete.")

def main():
    load_dotenv()
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        print("Error: Set S3_BUCKET in your .env file.")
        return

    download_images_from_s3(bucket, PREFIXES, local_dir="test_images")

if __name__ == "__main__":
    main()
