import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import argparse
import pandas as pd
from collections import defaultdict
from PIL import Image
import cv2
import math

def load_face_embeddings(faces_dir):
    """Load all face embeddings from the faces directory"""
    face_files = [f for f in os.listdir(faces_dir) if f.endswith('.pkl')]
    
    if not face_files:
        raise ValueError(f"No face embedding files found in {faces_dir}")
    
    print(f"Loading {len(face_files)} face embeddings...")
    
    face_data = {}
    for face_file in tqdm(face_files, desc="Loading embeddings"):
        file_path = os.path.join(faces_dir, face_file)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            face_id = os.path.splitext(face_file)[0]  # Remove .pkl extension
            face_data[face_id] = data
    
    return face_data

def compute_similarity_matrix(face_data):
    """Compute similarity matrix between all face embeddings"""
    face_ids = list(face_data.keys())
    num_faces = len(face_ids)
    
    # Extract embeddings into a matrix
    embeddings = np.zeros((num_faces, len(next(iter(face_data.values()))['embedding'])))
    
    for i, face_id in enumerate(face_ids):
        embeddings[i] = face_data[face_id]['embedding']
    
    # Compute cosine similarity matrix
    print("Computing similarity matrix...")
    similarity_matrix = cosine_similarity(embeddings)
    
    return similarity_matrix, face_ids

def extract_face_from_image(image_path, face_region):
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

def find_similar_faces(similarity_matrix, face_ids, face_data, threshold=0.7, input_dir=None):
    """Find similar faces based on similarity threshold"""
    num_faces = len(face_ids)
    similar_pairs = []
    
    print(f"Finding similar faces with threshold {threshold}...")
    
    # Create a dictionary to group faces by source image
    faces_by_image = defaultdict(list)
    for i, face_id in enumerate(face_ids):
        source_image = face_data[face_id]['source_image']
        faces_by_image[source_image].append((i, face_id))
    
    # Find similar faces across different images
    for i in tqdm(range(num_faces), desc="Comparing faces"):
        face_i_id = face_ids[i]
        face_i_source = face_data[face_i_id]['source_image']
        face_i_region = face_data[face_i_id]['region']
        
        for j in range(i + 1, num_faces):
            face_j_id = face_ids[j]
            face_j_source = face_data[face_j_id]['source_image']
            face_j_region = face_data[face_j_id]['region']
            
            # Skip if faces are from the same image
            if face_i_source == face_j_source:
                continue
            
            similarity = similarity_matrix[i, j]
            if similarity >= threshold:
                similar_pairs.append({
                    'face1_id': face_i_id,
                    'face2_id': face_j_id,
                    'similarity': similarity,
                    'face1_source': face_i_source,
                    'face2_source': face_j_source,
                    'face1_region': face_i_region,
                    'face2_region': face_j_region
                })
    
    # Sort by similarity (highest first)
    similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
    
    return similar_pairs

def analyze_similarity_distribution(similarity_matrix, face_ids, face_data):
    """Analyze the distribution of similarity scores"""
    # Extract similarities between different images
    different_image_similarities = []
    same_image_similarities = []
    
    num_faces = len(face_ids)
    
    for i in range(num_faces):
        face_i_id = face_ids[i]
        face_i_source = face_data[face_i_id]['source_image']
        
        for j in range(i + 1, num_faces):
            face_j_id = face_ids[j]
            face_j_source = face_data[face_j_id]['source_image']
            
            similarity = similarity_matrix[i, j]
            
            if face_i_source == face_j_source:
                same_image_similarities.append(similarity)
            else:
                different_image_similarities.append(similarity)
    
    # Plot similarity distributions
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(different_image_similarities, bins=50, alpha=0.7, label='Different Images')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title('Similarity Distribution (Different Images)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(same_image_similarities, bins=50, alpha=0.7, label='Same Image')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title('Similarity Distribution (Same Image)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('similarity_distribution.png')
    print(f"Saved similarity distribution plot to similarity_distribution.png")
    
    # Calculate statistics
    stats = {
        'different_images': {
            'count': len(different_image_similarities),
            'mean': np.mean(different_image_similarities),
            'median': np.median(different_image_similarities),
            'std': np.std(different_image_similarities),
            'min': np.min(different_image_similarities),
            'max': np.max(different_image_similarities)
        },
        'same_image': {
            'count': len(same_image_similarities),
            'mean': np.mean(same_image_similarities) if same_image_similarities else 0,
            'median': np.median(same_image_similarities) if same_image_similarities else 0,
            'std': np.std(same_image_similarities) if same_image_similarities else 0,
            'min': np.min(same_image_similarities) if same_image_similarities else 0,
            'max': np.max(same_image_similarities) if same_image_similarities else 0
        }
    }
    
    return stats

def create_comparison_visualizations(similar_pairs, face_data, input_dir, output_dir, max_pairs=20):
    """Create side-by-side visual comparisons of similar faces"""
    if not similar_pairs:
        return
    
    # Create output directory for comparisons
    comparisons_dir = os.path.join(output_dir, 'comparisons')
    os.makedirs(comparisons_dir, exist_ok=True)
    
    # Create a grid of the top similar pairs
    num_pairs = min(max_pairs, len(similar_pairs))
    
    # Also create individual comparison images
    for i, pair in enumerate(similar_pairs[:num_pairs]):
        face1_id = pair['face1_id']
        face2_id = pair['face2_id']
        similarity = pair['similarity']
        
        # Get source image paths
        face1_source_path = os.path.join(input_dir, pair['face1_source'])
        face2_source_path = os.path.join(input_dir, pair['face2_source'])
        
        # Load full original images
        face1_full_img = cv2.imread(face1_source_path)
        face2_full_img = cv2.imread(face2_source_path)
        
        if face1_full_img is None or face2_full_img is None:
            print(f"Warning: Could not load images for comparison {i+1}")
            continue
            
        # Convert from BGR to RGB for matplotlib
        face1_full_img = cv2.cvtColor(face1_full_img, cv2.COLOR_BGR2RGB)
        face2_full_img = cv2.cvtColor(face2_full_img, cv2.COLOR_BGR2RGB)
        
        # Extract faces from images
        face1_img = extract_face_from_image(face1_source_path, pair['face1_region'])
        face2_img = extract_face_from_image(face2_source_path, pair['face2_region'])
        
        if face1_img is None or face2_img is None:
            continue
        
        # Draw rectangles on the full images to highlight the faces
        face1_region = pair['face1_region']
        face2_region = pair['face2_region']
        
        # Create copies to draw on
        face1_full_with_rect = face1_full_img.copy()
        face2_full_with_rect = face2_full_img.copy()
        
        # Draw rectangles
        cv2.rectangle(face1_full_with_rect, 
                     (face1_region['x'], face1_region['y']), 
                     (face1_region['x'] + face1_region['w'], face1_region['y'] + face1_region['h']), 
                     (0, 255, 0), 2)
        
        cv2.rectangle(face2_full_with_rect, 
                     (face2_region['x'], face2_region['y']), 
                     (face2_region['x'] + face2_region['w'], face2_region['y'] + face2_region['h']), 
                     (0, 255, 0), 2)
        
        # Create detailed comparison image with 4 panels
        plt.figure(figsize=(15, 10))
        
        # Top row: Full images with rectangles
        plt.subplot(2, 2, 1)
        plt.imshow(face1_full_with_rect)
        plt.title(f"Image 1: {pair['face1_source']}")
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(face2_full_with_rect)
        plt.title(f"Image 2: {pair['face2_source']}")
        plt.axis('off')
        
        # Bottom row: Extracted faces
        plt.subplot(2, 2, 3)
        plt.imshow(face1_img)
        plt.title(f"Face 1: {face1_id}")
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.imshow(face2_img)
        plt.title(f"Face 2: {face2_id}")
        plt.axis('off')
        
        plt.suptitle(f"Similarity: {similarity:.4f}", fontsize=16)
        plt.tight_layout()
        
        # Save detailed comparison
        comparison_path = os.path.join(comparisons_dir, f"comparison_{i+1}_{similarity:.4f}.png")
        plt.savefig(comparison_path)
        plt.close()
    
    # Create a grid of all similar face pairs (just the face crops)
    rows = math.ceil(num_pairs / 4)  # 4 pairs per row
    plt.figure(figsize=(20, 5 * rows))
    
    for i, pair in enumerate(similar_pairs[:num_pairs]):
        if i >= num_pairs:
            break
            
        similarity = pair['similarity']
        
        # Get source image paths
        face1_source = os.path.join(input_dir, pair['face1_source'])
        face2_source = os.path.join(input_dir, pair['face2_source'])
        
        # Extract faces from images
        face1_img = extract_face_from_image(face1_source, pair['face1_region'])
        face2_img = extract_face_from_image(face2_source, pair['face2_region'])
        
        if face1_img is None or face2_img is None:
            continue
        
        # Add to the grid
        plt.subplot(rows, 4, i+1)
        
        # Make sure images have the same height for proper side-by-side display
        h1, w1 = face1_img.shape[:2]
        h2, w2 = face2_img.shape[:2]
        
        # Resize to same height if needed
        if h1 != h2:
            target_height = max(h1, h2)
            if h1 < target_height:
                scale = target_height / h1
                face1_img = cv2.resize(face1_img, (int(w1 * scale), target_height))
            if h2 < target_height:
                scale = target_height / h2
                face2_img = cv2.resize(face2_img, (int(w2 * scale), target_height))
        
        # Create a side-by-side comparison with a separator
        separator = np.ones((face1_img.shape[0], 5, 3), dtype=np.uint8) * 255  # White separator
        combined_img = np.hstack((face1_img, separator, face2_img))
        
        plt.imshow(combined_img)
        plt.title(f"Similarity: {similarity:.4f}")
        plt.axis('off')
    
    # Save the grid
    plt.tight_layout()
    grid_path = os.path.join(output_dir, 'similar_faces_grid.png')
    plt.savefig(grid_path, dpi=150)
    
    # Create a single image with side-by-side comparisons
    # This is a vertical stack of the top pairs
    if num_pairs > 0:
        side_by_side_comparisons = []
        
        for i, pair in enumerate(similar_pairs[:min(10, num_pairs)]):
            similarity = pair['similarity']
            
            # Get source image paths
            face1_source = os.path.join(input_dir, pair['face1_source'])
            face2_source = os.path.join(input_dir, pair['face2_source'])
            
            # Extract faces from images
            face1_img = extract_face_from_image(face1_source, pair['face1_region'])
            face2_img = extract_face_from_image(face2_source, pair['face2_region'])
            
            if face1_img is None or face2_img is None:
                continue
            
            # Resize both images to the same size for consistency
            target_size = (224, 224)  # Standard size
            face1_resized = cv2.resize(face1_img, target_size)
            face2_resized = cv2.resize(face2_img, target_size)
            
            # Create text with similarity score
            text_img = np.ones((40, target_size[0]*2 + 10, 3), dtype=np.uint8) * 255
            cv2.putText(text_img, f"Similarity: {similarity:.4f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Create a side-by-side comparison with a separator
            separator = np.ones((target_size[1], 10, 3), dtype=np.uint8) * 255  # White separator
            row_img = np.hstack((face1_resized, separator, face2_resized))
            
            # Add text below the images
            comparison = np.vstack((row_img, text_img))
            
            # Add a horizontal separator between pairs
            if i > 0:
                h_separator = np.ones((10, comparison.shape[1], 3), dtype=np.uint8) * 200  # Light gray separator
                side_by_side_comparisons.append(h_separator)
            
            side_by_side_comparisons.append(comparison)
        
        if side_by_side_comparisons:
            # Combine all comparisons vertically
            all_comparisons = np.vstack(side_by_side_comparisons)
            
            # Save the combined image
            combined_path = os.path.join(output_dir, 'side_by_side_comparisons.png')
            cv2.imwrite(combined_path, cv2.cvtColor(all_comparisons, cv2.COLOR_RGB2BGR))
            print(f"Saved side-by-side comparisons to {combined_path}")
    
    print(f"Saved {num_pairs} face comparison images to {comparisons_dir}")
    print(f"Saved comparison grid to {grid_path}")

def save_results(similar_pairs, stats, output_dir):
    """Save results to CSV and text files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save similar pairs to CSV
    if similar_pairs:
        # Create a copy without the region data (too large for CSV)
        csv_pairs = []
        for pair in similar_pairs:
            csv_pair = pair.copy()
            csv_pair.pop('face1_region', None)
            csv_pair.pop('face2_region', None)
            csv_pairs.append(csv_pair)
            
        df = pd.DataFrame(csv_pairs)
        csv_path = os.path.join(output_dir, 'similar_faces.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(similar_pairs)} similar face pairs to {csv_path}")
    else:
        print("No similar faces found with the current threshold")
    
    # Save statistics to text file
    stats_path = os.path.join(output_dir, 'similarity_stats.txt')
    with open(stats_path, 'w') as f:
        f.write("Similarity Statistics\n")
        f.write("====================\n\n")
        
        f.write("Different Images:\n")
        f.write(f"  Count: {stats['different_images']['count']}\n")
        f.write(f"  Mean: {stats['different_images']['mean']:.4f}\n")
        f.write(f"  Median: {stats['different_images']['median']:.4f}\n")
        f.write(f"  Std Dev: {stats['different_images']['std']:.4f}\n")
        f.write(f"  Min: {stats['different_images']['min']:.4f}\n")
        f.write(f"  Max: {stats['different_images']['max']:.4f}\n\n")
        
        f.write("Same Image:\n")
        f.write(f"  Count: {stats['same_image']['count']}\n")
        if stats['same_image']['count'] > 0:
            f.write(f"  Mean: {stats['same_image']['mean']:.4f}\n")
            f.write(f"  Median: {stats['same_image']['median']:.4f}\n")
            f.write(f"  Std Dev: {stats['same_image']['std']:.4f}\n")
            f.write(f"  Min: {stats['same_image']['min']:.4f}\n")
            f.write(f"  Max: {stats['same_image']['max']:.4f}\n")
    
    print(f"Saved similarity statistics to {stats_path}")

def main():
    parser = argparse.ArgumentParser(description='Compare face embeddings for similarity')
    parser.add_argument('--faces_dir', type=str, default='embeds/faces',
                        help='Directory containing face embedding files')
    parser.add_argument('--input_dir', type=str, default='test_images',
                        help='Directory containing original images')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Similarity threshold (0.0 to 1.0) for matching faces')
    parser.add_argument('--max_comparisons', type=int, default=20,
                        help='Maximum number of face comparisons to visualize')
    
    args = parser.parse_args()
    
    # Load face embeddings
    face_data = load_face_embeddings(args.faces_dir)
    
    # Compute similarity matrix
    similarity_matrix, face_ids = compute_similarity_matrix(face_data)
    
    # Find similar faces
    similar_pairs = find_similar_faces(similarity_matrix, face_ids, face_data, args.threshold, args.input_dir)
    
    # Analyze similarity distribution
    stats = analyze_similarity_distribution(similarity_matrix, face_ids, face_data)
    
    # Save results
    save_results(similar_pairs, stats, args.output_dir)
    
    # Create visual comparisons
    if similar_pairs:
        create_comparison_visualizations(similar_pairs, face_data, args.input_dir, args.output_dir, args.max_comparisons)
    
    print("\nCross-validation complete!")
    print(f"Found {len(similar_pairs)} similar face pairs with threshold {args.threshold}")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
