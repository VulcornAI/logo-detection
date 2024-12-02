from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import pandas as pd
from pathlib import Path
import uvicorn
from pydantic import BaseModel
import logging
from PIL import Image
import io
import yaml

from src.model import ImageSimilaritySearch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
with open('configs/config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app
app = FastAPI(
    title="Logo Identification API",
    description="API for logo detection and similarity search",
    version="1.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response model
class SearchResponse(BaseModel):
    matches: List[dict]

# Initialize the model
try:
    similarity_search = ImageSimilaritySearch(
        yolo_path=config['detection_model_weight_path'],
        device='cpu',  # Change to 'cuda' if GPU is available
        top_n=4,
        index_type='cosine',
        support_dataset_embeddings_path=config['support_dataset_embeddings_path']
    )
    label_df = pd.read_csv(config['support_dataset_label_csv_path'])
    logger.info("Model and resources loaded successfully")
except Exception as e:
    logger.error(f"Failed to initialize model: {str(e)}")
    raise

@app.get("/")
async def root() -> dict:
    """
    Summary: Default route that returns API information
    
    Returns:
        dict: Basic API information and status
    """
    return {
        "name": "Logo Identification API",
        "version": "1.0.0",
        "status": "active",
        "description": "API for logo detection and similarity search",
        "endpoints": {
            "root": "/",
            "search": "/search"
        }
    }

@app.post("/search", response_model=SearchResponse)
async def search_logo(
    file: UploadFile = File(...),
    top_k: Optional[int] = 4
) -> SearchResponse:
    """
    Summary: Perform logo detection and similarity search on uploaded image
    
    Args:
        file (UploadFile): Uploaded image file
        top_k (int, optional): Number of similar logos to return
        
    Returns:
        SearchResponse: List of matching logos with similarity scores and company names
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an image file."
            )

        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Save temporary file for processing
        temp_path = Path("temp_upload.jpg")
        image.save(temp_path)

        # Extract features and search
        query_embedding = similarity_search.extract_feature(str(temp_path), is_crop=True)
        
        # Handle case where no logos are detected
        if query_embedding is None or len(query_embedding) == 0:
            return SearchResponse(matches=[])
            
        indices, similarities = similarity_search.search(query_embedding)
        
        # Validate indices before getting image paths
        if len(indices) == 0:
            return SearchResponse(matches=[])
            
        # Get similar image paths
        try:
            similar_image_paths = similarity_search.search_image(indices, config['support_dataset_images_path'])
        except IndexError:
            logger.error("No matching images found in support dataset")
            return SearchResponse(matches=[])

        # Prepare response
        matches = []
        for path, similarity in zip(similar_image_paths, similarities):
            if path is None:
                continue
                
            try:
                image_name = Path(path).name
                company_name = "Not found"
                
                # Look up company name from CSV
                if image_name in label_df['name'].values:
                    company_name = label_df[label_df['name'] == image_name]['label'].values[0]

                matches.append({
                    "image_name": image_name,
                    "similarity_score": float(similarity),
                    "company_name": company_name
                })
            except Exception as e:
                logger.error(f"Error processing match {path}: {str(e)}")
                continue

        # Clean up
        temp_path.unlink()

        return SearchResponse(matches=matches)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
