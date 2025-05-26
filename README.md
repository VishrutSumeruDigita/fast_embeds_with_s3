# Face Embedding Generation from S3 Images

This project provides tools to generate face embeddings from images, including directly from AWS S3 buckets.

## Features

- Download images from AWS S3 buckets
- Detect faces in images using MTCNN
- Generate face embeddings using FaceNet (InceptionResnetV1)
- Process images in batches for better performance
- Save individual face embeddings for later comparison

## Requirements

See `requirements.txt` for all dependencies. Main requirements:
- Python 3.6+
- PyTorch
- facenet-pytorch
- boto3 (for S3 access)

## Installation

```bash
pip install -r requirements.txt
```

## AWS Configuration

Before using the S3 functionality, ensure you have AWS credentials configured. You have two options:

### Option 1: Using the .env file (Recommended)

1. Edit the `.env` file in the project root and add your AWS credentials:
   ```
   AWS_ACCESS_KEY_ID=your_access_key_here
   AWS_SECRET_ACCESS_KEY=your_secret_key_here
   AWS_DEFAULT_REGION=your_region_here
   
   # You can also configure other settings
   S3_BUCKET=divinepic-high2
   S3_PREFIX=Divinepic-11_FEB_2025_EVENING_DARSHAN_SECRETARIAT-ucpishn9/
   ```

### Option 2: Using AWS CLI configuration

1. Install AWS CLI
2. Run `aws configure` and enter your credentials

## Usage

### Generating Face Embeddings from S3

#### Using .env file (Recommended)

If you've set up your `.env` file with all the necessary configurations, simply run:

```bash
python s3_face_embeddings.py
```

#### Using Command Line Arguments

You can also override the `.env` settings with command line arguments:

```bash
python s3_face_embeddings.py --bucket BUCKET_NAME --prefix PREFIX_PATH --output-dir OUTPUT_DIR
```

Example:
```bash
python s3_face_embeddings.py --bucket divinepic-high2 --prefix Divinepic-11_FEB_2025_EVENING_DARSHAN_SECRETARIAT-ucpishn9/ --output-dir embeds/s3_faces
```

### Additional Options

- `--batch-size`: Number of images to process in each batch (default: 4)
- `--keep-images`: Keep downloaded images after processing
- `--images-dir`: Directory to save downloaded images (if keep-images is True)

### Generating Face Embeddings from Local Images

```bash
python generate_embeddings.py
```

## Output

The script generates:
- A directory with individual face embeddings (`.pkl` files)
- A combined `face_embeddings.pkl` file with all embeddings
- A `metadata.pkl` file with processing statistics
