import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import argparse
from tqdm import tqdm
import math

def load_embeddings(embeddings_file):
    """Load the face embeddings from the pickle file"""
    with open(embeddings_file, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

def visualize_test_images(input_dir, embeddings, output_dir):
    """Visualize all test images with face detections highlighted"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a directory for individual images
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Get all image files that have embeddings
    image_files = list(embeddings.keys())
    print(f"Found {len(image_files)} images with face embeddings")
    
    # Calculate grid dimensions
    grid_size = min(len(image_files), 25)  # Show up to 25 images in the grid
    grid_rows = math.ceil(math.sqrt(grid_size))
    grid_cols = math.ceil(grid_size / grid_rows)
    
    # Create a grid figure
    plt.figure(figsize=(4*grid_cols, 4*grid_rows))
    
    # Process each image
    for i, image_file in enumerate(tqdm(image_files, desc="Visualizing images")):
        # Load the image
        image_path = os.path.join(input_dir, image_file)
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Warning: Could not load image {image_file}")
            continue
            
        # Convert from BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get face regions for this image
        face_regions = embeddings[image_file]['face_regions']
        
        # Create a copy to draw on
        img_with_faces = img_rgb.copy()
        
        # Draw rectangles and labels for each face
        for i, region in enumerate(face_regions):
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
            # Draw rectangle
            cv2.rectangle(img_with_faces, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw face number
            cv2.putText(img_with_faces, f"Face {i+1}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save individual image
        output_path = os.path.join(images_dir, f"{os.path.splitext(image_file)[0]}_faces.png")
        plt.figure(figsize=(10, 10))
        plt.imshow(img_with_faces)
        plt.title(f"{image_file} - {len(face_regions)} faces detected")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        # Add to grid (only first 25 images)
        if i < grid_size:
            plt.figure(1)  # Switch back to the grid figure
            plt.subplot(grid_rows, grid_cols, i+1)
            plt.imshow(img_with_faces)
            plt.title(f"{len(face_regions)} faces", fontsize=10)
            plt.axis('off')
    
    # Save the grid
    plt.figure(1)
    plt.tight_layout()
    grid_path = os.path.join(output_dir, 'test_images_grid.png')
    plt.savefig(grid_path, dpi=150)
    plt.close()
    
    print(f"Saved individual images to {images_dir}")
    print(f"Saved image grid to {grid_path}")

def create_face_montage(embeddings, input_dir, output_dir):
    """Create a montage of all detected faces"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all faces
    all_faces = []
    face_labels = []
    
    for image_file, data in tqdm(embeddings.items(), desc="Collecting faces"):
        # Load the image
        image_path = os.path.join(input_dir, image_file)
        img = cv2.imread(image_path)
        
        if img is None:
            continue
            
        # Convert from BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Extract each face
        for i, region in enumerate(data['face_regions']):
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
            # Add margin
            margin_x = int(w * 0.1)
            margin_y = int(h * 0.1)
            
            # Make sure coordinates are within image bounds
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(img_rgb.shape[1], x + w + margin_x)
            y2 = min(img_rgb.shape[0], y + h + margin_y)
            
            # Extract face
            face = img_rgb[y1:y2, x1:x2]
            
            # Resize to standard size
            face_resized = cv2.resize(face, (112, 112))
            
            all_faces.append(face_resized)
            face_labels.append(f"{image_file}_{i+1}")
    
    # Create montage
    num_faces = len(all_faces)
    if num_faces == 0:
        print("No faces found")
        return
    
    # Calculate grid dimensions
    grid_cols = min(20, num_faces)  # Max 20 faces per row
    grid_rows = math.ceil(num_faces / grid_cols)
    
    # Create montage image
    montage_size = (112 * grid_cols, 112 * grid_rows)
    montage = np.zeros((montage_size[1], montage_size[0], 3), dtype=np.uint8)
    
    # Place faces in the montage
    for i, face in enumerate(all_faces):
        row = i // grid_cols
        col = i % grid_cols
        
        y_start = row * 112
        x_start = col * 112
        
        montage[y_start:y_start+112, x_start:x_start+112] = face
    
    # Save montage
    montage_path = os.path.join(output_dir, 'all_faces_montage.png')
    cv2.imwrite(montage_path, cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    print(f"Saved face montage to {montage_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize test images with face detections')
    parser.add_argument('--input_dir', type=str, default='test_images',
                        help='Directory containing test images')
    parser.add_argument('--embeddings_file', type=str, default='embeds/face_embeddings.pkl',
                        help='Path to face embeddings pickle file')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Load embeddings
    embeddings = load_embeddings(args.embeddings_file)
    
    # Visualize test images
    visualize_test_images(args.input_dir, embeddings, args.output_dir)
    
    # Create face montage
    create_face_montage(embeddings, args.input_dir, args.output_dir)
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()
