import os
from pathlib import Path
from typing import List, Union
import numpy as np
from tqdm import tqdm
from model import ImageSimilaritySearch
import torch
import sys
absolute_path = os.path.dirname(__file__)
sys.path.append(absolute_path)
sys.path.append(os.path.join(absolute_path, '..'))


def create_embeddings(
    image_folder: Union[str, Path],
    output_folder: Union[str, Path],
    batch_size: int = 32
) -> None:
    """
    Summary: Create and save embeddings for all images in the specified folder
    
    Args:
        image_folder (Union[str, Path]): Path to folder containing source images
        output_folder (Union[str, Path]): Path to save the generated embeddings
        batch_size (int): Number of images to process in parallel
        
    Returns:
        None: Saves embeddings as .npy files in the output folder
    """
    # Initialize the similarity search model without building index
    similarity_search = ImageSimilaritySearch(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        build_index=False
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of image files
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    image_files = [
        f for f in Path(image_folder).rglob("*")
        if f.suffix.lower() in valid_extensions
    ]
    
    # Process images and save embeddings
    for image_path in tqdm(image_files, desc="Generating embeddings"):
        try:
            # Generate embedding
            embedding = similarity_search.extract_feature(str(image_path), is_crop=True)
            
            # Create output filename (maintaining folder structure)
            relative_path = image_path.relative_to(image_folder)
            output_path = Path(output_folder) / f"{relative_path.stem}.npy"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save embedding
            np.save(output_path, embedding)
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings for image similarity search")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to folder containing images")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save embeddings")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    
    args = parser.parse_args()
    
    create_embeddings(
        image_folder=args.image_folder,
        output_folder=args.output_folder,
        batch_size=args.batch_size
    )
