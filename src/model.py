import os
import sys
from pathlib import Path
absolute_path = os.path.dirname(__file__)
sys.path.append(absolute_path)
sys.path.append(os.path.join(absolute_path, '..'))

from typing import List, Tuple, Union
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import torch
from ultralytics import YOLO
from torchvision.ops import nms
from transformers import AutoProcessor, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import glob
import faiss

Image.MAX_IMAGE_PIXELS = None

class ImageSimilaritySearch:
    """
    Visual Content Search And Matching Engine (ImageSimilaritySearch) for image similarity search
    using YOLO for object detection and SIGLIP for feature extraction
    """
    
    def __init__(self, 
                 yolo_path: str = './weights/logo_detection_weight.pt',
                 device: str = 'cpu',
                 top_n: int = 4,
                 index_type: str = 'l2',
                 build_index: bool = True,
                 model_name: str = 'google/siglip-base-patch16-224',
                 support_dataset_embeddings_path: str = './support_dataset/embeddings'
                ):
        """
        Initialize ImageSimilaritySearch with models and configurations
        
        Args:
            yolo_path: Path to YOLO model weights
            device: Device to run models on ('cpu' or 'cuda')
            top_n: Number of top results to return
            index_type: Type of FAISS index to use ('l2' or 'cosine')
            build_index: Whether to load embeddings and build FAISS index
            model_name: Name of the embedding model to use
            support_dataset_embeddings_path: Path to support dataset containing embeddings
        """
        self.device = device
        self.yolo_model = YOLO(yolo_path).to(self.device)
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        if device == 'cuda':
            self.model = self.model.cuda()
        
        self.model.eval()
         
        self.top_n = top_n
        self.index_type = index_type

        # Only load embeddings and build index if needed
        if build_index:
            self.embed_paths, self.embed_numpy = self._load_embed(
                glob.glob(os.path.join(support_dataset_embeddings_path, '*.npy'))
            )
            self._build_index()

    def _build_index(self):
        """Build FAISS index for fast similarity search"""
        dimension = self.embed_numpy.shape[1]
        
        if self.index_type == 'cosine':
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            # Normalize vectors to use inner product as cosine similarity
            faiss.normalize_L2(self.embed_numpy)
        else:
            self.index = faiss.IndexFlatL2(dimension)  # L2 distance
            
        self.index.add(self.embed_numpy.astype('float32'))

    def _load_embed(self, paths: List[str]) -> Tuple[List[str], np.ndarray]:
        """
        Load embedding files
        
        Args:
            paths: List of paths to embedding files
            
        Returns:
            Tuple of embedding paths and numpy array of embeddings
        """
        embed_paths = sorted(paths)  # Sort for consistent ordering
        embed_numpy = []

        for path in tqdm(embed_paths, desc="Loading embeddings"):
            embed_numpy.append(np.load(path))

        embed_numpy = np.array(embed_numpy)
        return embed_paths, embed_numpy
    
    def _apply_nms(self, detections: torch.Tensor, iou_threshold: float) -> np.ndarray:
        """
        Apply non-maximum suppression to detections
        
        Args:
            detections: Detection boxes and scores
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Filtered detections after NMS
        """
        boxes = detections[:, :4]
        confidences = detections[:, 4]
        nms_indices = nms(boxes, confidences, iou_threshold=iou_threshold)
        return detections[nms_indices].numpy()

    def _crop(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Detect and crop logo regions from image
        
        Args:
            image_path: Path to input image
            
        Returns:
            Cropped image containing detected logos
        """
        yolo_results = self.yolo_model(image_path, verbose=False, conf=0.25)
        pos_list = []
        
        for result in yolo_results:
            detections = result.boxes.data.cpu()
            if len(detections) == 0:
                continue
            filtered_detections = self._apply_nms(detections, 0.5)
            for detection in filtered_detections:
                x1, y1, x2, y2, conf, cls = map(int, detection.astype(float))
                pos_list.append([x1, y1, x2, y2])

        if not pos_list:
            return Image.open(image_path).convert("RGB")

        img = Image.open(image_path)
        new_img = Image.new('RGB', img.size, (255, 255, 255))

        for box in pos_list:
            x1, y1, x2, y2 = box
            cropped_img = img.crop((x1, y1, x2, y2))
            new_img.paste(cropped_img, (x1, y1))

        x_min, y_min, _, _ = np.min(pos_list, axis=0)
        _, _, x_max, y_max = np.max(pos_list, axis=0)
        return new_img.crop((x_min, y_min, x_max, y_max))

    @torch.no_grad()
    def extract_feature(self, url: Union[str, Path], is_crop: bool = True) -> np.ndarray:
        """
        Extract image features using SIGLIP model
        
        Args:
            url: Path to input image
            is_crop: Whether to crop logo regions first
            
        Returns:
            Image embedding vector
        """
        image = self._crop(url) if is_crop else Image.open(url).convert("RGB")
        inputs = self.processor(text=[""], images=image, padding="max_length", return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        image_embeds = outputs.image_embeds.cpu().float().numpy()[0]
        
        if self.index_type == 'cosine':
            image_embeds = image_embeds / np.linalg.norm(image_embeds)
            
        return image_embeds

    def search(self, embed: np.ndarray, mode: str = 'image') -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar images
        
        Args:
            embed: Query image embedding
            mode: Search mode ('image' only for now)
            
        Returns:
            Tuple of (indices of top matches, similarity scores)
        """
        distances, indices = self.index.search(
            embed.reshape(1, -1).astype('float32'), 
            self.top_n
        )
        
        if self.index_type == 'cosine':
            similarities = 1 - distances/2  # Convert distance to similarity score
        else:
            similarities = 1 / (1 + distances)  # Convert L2 distance to similarity score
            
        return indices[0], similarities[0]

    def extract_id(self, index: int, mode: str = 'image') -> str:
        """Extract image ID from embedding path"""
        return os.path.splitext(os.path.basename(self.embed_paths[index]))[0]

    def search_image(self, indices: np.ndarray, image_folder_path: Union[str, Path]) -> List[str]:
        """Get image paths for given indices"""
        image_names = [self.extract_id(index=index, mode='image') for index in indices]
        image_paths = [glob.glob(f'{image_folder_path}/{name}*')[0] for name in image_names]
        return image_paths