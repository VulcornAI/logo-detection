"""
This module contains utility functions for view separation using YOLOv10 model.

The functions in this module provide functionality for object detection on images using the YOLOv10 model, as well as other utility functions such as cropping images, loading images, and applying Non-Maximum Suppression (NMS) to filter detections.

The main functions in this module are:
- `predict_with_yolo`: Perform object detection on an image using the YOLOv10 model.
- `get_device`: Determine the available device (CUDA or CPU).
- `load_yolov10_model`: Load the YOLOv10 model from model weight path.
- `log_error`: Log error messages to a file.
- `apply_nms`: Apply Non-Maximum Suppression (NMS) to filter detections.
- `crop_image`: Crop an image based on given coordinates.
- `extract_base_name_and_ext`: Extract the base name and extension from an image path.
"""

import os
from ultralytics import YOLO
from torchvision.ops import nms
from typing import List, TypeAlias
import numpy as np
from numpy.typing import NDArray
from torch import Tensor
import torch
import cv2

Image3D: TypeAlias = NDArray[np.uint8]  # Assuming image is 3D, such as (height, width, channels)
BoundingBoxTensor: TypeAlias = Tensor  # Tensor of shape (N, 6) where N is the number of bounding boxes and 6 is the number of elements in each bounding box (x1, y1, x2, y2, confidence, class)


def predict_with_yolo(yolo_model: YOLO, image_path: str, **kwargs) -> List:
    """
    Perform object detection on an image using the YOLOv10 model.

    Args:
        yolo_model (YOLOv10): The YOLOv10 model used for detection.
        image_path (str): The path of the image to be processed.
        conf_threshold (float): The confidence threshold for detection.

    Returns:
        List[Result]: A list of detection results from the YOLOv10 model.
    """
    return yolo_model(image_path, verbose=False, **kwargs)


def get_device():
    """
    Determine the available device (CUDA or CPU).

    Returns:
        torch.device: The available device.
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_yolo_model(model_weight_path: str, device=None) -> YOLO:
    """
    Load the YOLO model from model weight path.

    Args:
        model_weight_path (str): Path to YOLOV10 model weight.
        device (torch.device, optional): Device to load the model on. If None, uses get_device().

    Returns:
        YOLOv10: Initialized YOLO model.
    """
    if device is None:
        device = get_device()
    return YOLO(model_weight_path).to(device)


def log_error(output_folder: str, error_message: str) -> None:
    """Log error messages to a file.

    Args:
        output_folder (str): The directory where processed images will be saved.
        error_message (str): The error message to log.

    Returns:
        None
    """
    log_file_path = os.path.join(os.path.dirname(output_folder), "Logfile.txt")
    with open(log_file_path, "a") as file:
        file.write(error_message + "\n")


def apply_nms(detections: BoundingBoxTensor, iou_threshold: float) -> BoundingBoxTensor:
    """
    Apply Non-Maximum Suppression (NMS) to filter detections.

    Args:
        detections (BoundingBoxTensor): The raw detection results, including bounding boxes and confidence scores.
        iou_threshold (float): The Intersection over Union (IoU) threshold for NMS.

    Returns:
        np.ndarray: The filtered detections after applying NMS.
    """
    boxes = detections[:, :4]
    confidences = detections[:, 4]
    nms_indices = nms(boxes, confidences, iou_threshold=iou_threshold)
    return detections[nms_indices].numpy()


def crop_image(original_image, x1, y1, x2, y2):
    """Crops an image based on given coordinates.

    Args:
        original_image: The original image as a NumPy array.
        x1, y1: Top-left coordinates of the crop region.
        x2, y2: Bottom-right coordinates of the crop region.

    Returns:
        The cropped image as a NumPy array.
    """

    cropped_view = original_image[y1:y2, x1:x2]
    return cropped_view


def extract_base_name_and_ext(image_path):
    """Extracts the base name and extension from an image path.

    Args:
        image_path: The path to the image file.

    Returns:
        A tuple containing the base name and extension.
    """

    base_name, ext = os.path.splitext(os.path.basename(image_path))
    return base_name, ext

def load_image_cv2(image_path: str) -> Image3D:
    """Loads an image from the specified path using OpenCV.

    Parameters:
    image_path: The path to the image file.

    Returns:
    A NumPy array representing the loaded image.
    """

    original_image = cv2.imread(image_path)
    return original_image

def save_image(
        image: np.ndarray, 
        file_path: str) -> None:
    """
    Save a numpy array as an image file.

    Parameters:
    image (np.ndarray): Numpy array of the image.
    file_path (str): Path to save the image.
    """
    cv2.imwrite(file_path, image)