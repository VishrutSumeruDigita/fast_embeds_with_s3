import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import argparse
from tqdm import tqdm
import random
from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings(embeddings_file):
    """Load the face embeddings from the pickle file"""
    with open(embeddings_file, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

def extract_face(image_path, face_region):
    """Extract a face from an image using the face region coordinates"""
    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert from BGR to RGB (for matplotlib)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Extract face region
        x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
        
        # Add a small margin around the face (10% of width/height)
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)
        
        # Make sure coordinates are within image bounds
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(img.shape[1], x + w + margin_x)
        y2 = min(img.shape[0], y + h + margin_y)
        
        # Extract the face with margin
        face_img = img[y1:y2, x1:x2]
        
        return face_img
    except Exception as e:
        print(f"Error extracting face: {str(e)}")
        return None

def get_image_with_face_highlighted(image_path, face_region):
    """Get the image with the face highlighted"""
    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert from BGR to RGB (for matplotlib)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create a copy to draw on
        img_with_face = img.copy()
        
        # Extract face region
        x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
        
        # Draw rectangle
        cv2.rectangle(img_with_face, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return img_with_face
    except Exception as e:
        print(f"Error highlighting face: {str(e)}")
        return None

def compare_random_image_pairs(embeddings, input_dir, output_dir, num_pairs=9):
    """Compare random pairs of images and visualize them side by side with similarity scores"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files with embeddings
    image_files = list(embeddings.keys())
    
    if len(image_files) < 2:
        print("Not enough images to compare")
        return
    
    # Create a list of all face pairs
    all_pairs = []
    
    for img1_idx, img1 in enumerate(image_files):
        for face1_idx, face1_embedding in enumerate(embeddings[img1]['face_embeddings']):
            for img2_idx, img2 in enumerate(image_files):
                # Skip comparing the same image
                if img1 == img2:
                    continue
                    
                for face2_idx, face2_embedding in enumerate(embeddings[img2]['face_embeddings']):
                    # Calculate similarity
                    similarity = cosine_similarity([face1_embedding], [face2_embedding])[0][0]
                    
                    all_pairs.append({
                        'img1': img1,
                        'img2': img2,
                        'face1_idx': face1_idx,
                        'face2_idx': face2_idx,
                        'similarity': similarity,
                        'face1_region': embeddings[img1]['face_regions'][face1_idx],
                        'face2_region': embeddings[img2]['face_regions'][face2_idx]
                    })
    
    # Sort by similarity (both highest and lowest)
    all_pairs.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Select pairs to visualize
    selected_pairs = []
    
    # Add 3 highest similarity pairs
    selected_pairs.extend(all_pairs[:3])
    
    # Add 3 middle similarity pairs
    middle_idx = len(all_pairs) // 2
    selected_pairs.extend(all_pairs[middle_idx-1:middle_idx+2])
    
    # Add 3 lowest similarity pairs
    selected_pairs.extend(all_pairs[-3:])
    
    # Ensure we have the requested number of pairs
    if len(selected_pairs) < num_pairs:
        # Add random pairs if we don't have enough
        additional_needed = num_pairs - len(selected_pairs)
        if additional_needed > 0 and len(all_pairs) > num_pairs:
            remaining_pairs = [p for p in all_pairs if p not in selected_pairs]
            selected_pairs.extend(random.sample(remaining_pairs, min(additional_needed, len(remaining_pairs))))
    
    # Limit to requested number
    selected_pairs = selected_pairs[:num_pairs]
    
    # Create the visualization
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, pair in enumerate(selected_pairs):
        if i >= num_pairs:
            break
            
        img1 = pair['img1']
        img2 = pair['img2']
        face1_idx = pair['face1_idx']
        face2_idx = pair['face2_idx']
        similarity = pair['similarity']
        
        # Get paths
        img1_path = os.path.join(input_dir, img1)
        img2_path = os.path.join(input_dir, img2)
        
        # Get images with faces highlighted
        img1_with_face = get_image_with_face_highlighted(img1_path, pair['face1_region'])
        img2_with_face = get_image_with_face_highlighted(img2_path, pair['face2_region'])
        
        # Get face crops
        face1 = extract_face(img1_path, pair['face1_region'])
        face2 = extract_face(img2_path, pair['face2_region'])
        
        if img1_with_face is None or img2_with_face is None or face1 is None or face2 is None:
            continue
        
        # Create a side-by-side comparison
        # Top: full images with faces highlighted
        # Bottom: face crops
        
        # Create a figure for this pair
        plt.figure(figsize=(10, 8))
        
        # Plot full images
        plt.subplot(2, 2, 1)
        plt.imshow(img1_with_face)
        plt.title(f"{img1}\nFace {face1_idx+1}")
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(img2_with_face)
        plt.title(f"{img2}\nFace {face2_idx+1}")
        plt.axis('off')
        
        # Plot face crops
        plt.subplot(2, 2, 3)
        plt.imshow(face1)
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.imshow(face2)
        plt.axis('off')
        
        plt.suptitle(f"Similarity: {similarity:.4f}", fontsize=16)
        plt.tight_layout()
        
        # Save individual comparison
        plt.savefig(os.path.join(output_dir, f"pair_{i+1}_sim_{similarity:.4f}.png"))
        plt.close()
        
        # Add to the grid
        ax = axes[i]
        
        # Create a side-by-side comparison of just the faces
        # Make sure faces are the same height
        h1, w1 = face1.shape[:2]
        h2, w2 = face2.shape[:2]
        
        target_height = max(h1, h2)
        if h1 != target_height:
            scale = target_height / h1
            face1 = cv2.resize(face1, (int(w1 * scale), target_height))
        if h2 != target_height:
            scale = target_height / h2
            face2 = cv2.resize(face2, (int(w2 * scale), target_height))
        
        # Add a separator
        separator = np.ones((target_height, 5, 3), dtype=np.uint8) * 255
        combined = np.hstack((face1, separator, face2))
        
        ax.imshow(combined)
        ax.set_title(f"Similarity: {similarity:.4f}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "nine_pairs_comparison.png"), dpi=150)
    plt.close()
    
    print(f"Saved 9 image pair comparisons to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Compare random pairs of images with similarity scores')
    parser.add_argument('--input_dir', type=str, default='test_images',
                        help='Directory containing test images')
    parser.add_argument('--embeddings_file', type=str, default='embeds/face_embeddings.pkl',
                        help='Path to face embeddings pickle file')
    parser.add_argument('--output_dir', type=str, default='image_comparisons',
                        help='Directory to save comparisons')
    parser.add_argument('--num_pairs', type=int, default=9,
                        help='Number of image pairs to compare')
    
    args = parser.parse_args()
    
    # Load embeddings
    embeddings = load_embeddings(args.embeddings_file)
    
    # Compare random image pairs
    compare_random_image_pairs(embeddings, args.input_dir, args.output_dir, args.num_pairs)
    
    print("\nComparison complete!")

if __name__ == "__main__":
    main()
