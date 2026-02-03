# -*- coding: utf-8 -*-
# Simplified live camera inference function
# Can be imported and used in other projects like face-attendance-system

import os
import cv2
import numpy as np
import time
import torch
import torch.nn.functional as F

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name


class SimpleLiveAntiSpoofPredictor:
    """Simple wrapper for live camera anti-spoofing prediction"""
    
    def __init__(self, model_dir, device_id=0):
        """
        Initialize the predictor
        
        Args:
            model_dir: Path to anti-spoof models
            device_id: GPU device id
        """
        self.model_test = AntiSpoofPredict(device_id)
        self.image_cropper = CropImage()
        self.model_dir = model_dir
        self.device_id = device_id

    def predict_image(self, image):
        """
        Predict if face in image is real or fake
        
        Args:
            image: Input image (numpy array from cv2.imread or camera)
            
        Returns:
            dict with keys:
                - is_real: bool (True if real face, False if fake)
                - score: float (confidence score)
                - label: int (1 for real, 0 for fake)
                - bbox: list [x, y, w, h] of detected face
        """
        try:
            # Get face bounding box
            image_bbox = self.model_test.get_bbox(image)
            
            if image_bbox is None or len(image_bbox) == 0:
                return {
                    'is_real': None,
                    'score': 0.0,
                    'label': None,
                    'bbox': None,
                    'error': 'No face detected'
                }
            
            prediction = np.zeros((1, 3))
            
            # Sum predictions from all models
            for model_name in os.listdir(self.model_dir):
                if not model_name.endswith('.pth'):
                    continue
                
                h_input, w_input, model_type, scale = parse_model_name(model_name)
                param = {
                    "org_img": image,
                    "bbox": image_bbox,
                    "scale": scale,
                    "out_w": w_input,
                    "out_h": h_input,
                    "crop": True,
                }
                
                if scale is None:
                    param["crop"] = False
                
                img = self.image_cropper.crop(**param)
                
                if img is None:
                    continue
                
                prediction += self.model_test.predict(img, os.path.join(self.model_dir, model_name))
            
            # Get result
            label = np.argmax(prediction)
            score = prediction[0][label] / 2
            is_real = (label == 1)
            
            return {
                'is_real': is_real,
                'score': float(score),
                'label': int(label),
                'bbox': image_bbox,
                'error': None
            }
        
        except Exception as e:
            return {
                'is_real': None,
                'score': 0.0,
                'label': None,
                'bbox': None,
                'error': str(e)
            }


# Convenience function for direct use
def check_face_spoofing(image, model_dir, device_id=0):
    """
    Quick function to check if face is real or fake
    
    Args:
        image: Input image (numpy array)
        model_dir: Path to anti-spoof models
        device_id: GPU device id
        
    Returns:
        dict with prediction results
    """
    predictor = SimpleLiveAntiSpoofPredictor(model_dir, device_id)
    return predictor.predict_image(image)
