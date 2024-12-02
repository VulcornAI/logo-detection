import pandas as pd
import os
from tqdm import tqdm
import easyocr
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LabelGenerator:
    """
    Summary: Generate label.csv file from images using OCR
    """
    def __init__(self, image_dir: str, output_path: str = 'label.csv'):
        """
        Summary: Initialize LabelGenerator
        
        Args:
            image_dir (str): Directory containing images
            output_path (str): Path to save output CSV file
        """
        self.image_dir = Path(image_dir)
        self.output_path = Path(output_path)
        self.reader = easyocr.Reader(['en'])
        
    def _get_image_files(self) -> List[Path]:
        """
        Summary: Get list of image files from directory
        
        Returns:
            List[Path]: List of image file paths
        """
        return [f for f in self.image_dir.glob('*') if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]

    def _extract_text(self, image_path: Path) -> str:
        """
        Summary: Extract text from image using OCR
        
        Args:
            image_path (Path): Path to image file
            
        Returns:
            str: Extracted text or error message
        """
        try:
            results = self.reader.readtext(str(image_path))
            return ' '.join([text[1] for text in results]) if results else "No text detected"
        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {str(e)}")
            return "Error in OCR processing"

    def generate_labels(self) -> Optional[pd.DataFrame]:
        """
        Summary: Generate labels DataFrame from images
        
        Returns:
            Optional[pd.DataFrame]: DataFrame with image names and labels
        """
        try:
            image_files = self._get_image_files()
            
            # Initialize lists to store data
            image_names = []
            ocr_labels = []

            logger.info("Processing images with OCR...")
            for image_file in tqdm(image_files):
                detected_text = self._extract_text(image_file)
                image_names.append(image_file.name)
                ocr_labels.append(detected_text)

            # Create DataFrame
            df = pd.DataFrame({
                'name': image_names,
                'label': ocr_labels
            })

            # Save DataFrame
            df.to_csv(self.output_path, index=False)
            logger.info(f"Labels saved to {self.output_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating labels: {str(e)}")
            return None

def parse_arguments() -> Tuple[str, str]:
    """
    Summary: Parse command-line arguments for image directory and output path.
    
    Returns:
        Tuple[str, str]: Image directory and output path.
    """
    parser = argparse.ArgumentParser(description="Generate labels from images using OCR.")
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--output_path', type=str, default='label.csv', help='Path to save output CSV file')
    args = parser.parse_args()
    return args.image_dir, args.output_path

if __name__ == "__main__":
    image_dir, output_path = parse_arguments()
    generator = LabelGenerator(image_dir, output_path)
    generator.generate_labels()
