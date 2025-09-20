import torch
import json
import cv2 as cv
import numpy as np
import os

def predict_mask(image_array, model, device, metadata_path="models/metadata.json"):
    """Predict if a person is wearing a mask in the given image/frame."""
    
    try:
        # Check if metadata file exists
        if not os.path.exists(metadata_path):
            print(f"Warning: Metadata file not found: {metadata_path}")
            # Use default values
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            # Load metadata
            with open(metadata_path) as f:
                metadata = json.load(f)
            mean = metadata["preprocessing"]["mean"]
            std = metadata["preprocessing"]["std"]
        
        # preprocessing pipeline

        # 1. Input is BGR from OpenCV
        img = cv.cvtColor(image_array, cv.COLOR_BGR2RGB)  # Convert to RGB
        img = cv.resize(img, (224, 224))  # Resize
        
        # 2. Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # 3. Apply ImageNet normalization
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        img = (img - mean) / std
        
        # 4. Convert to tensor: (H, W, C) ---permute---> (C, H, W) ---unsqueeze---> (1, C, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Predict
        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            
            # Get probabilities for both classes
            mask_prob = probs[0, 0].item()      # Class 0: "Mask"
            no_mask_prob = probs[0, 1].item()   # Class 1: "No_Mask"
            
            # Use confidence threshold and better logic
            confidence_threshold = 0.5
            max_prob = max(mask_prob, no_mask_prob)
            
            if max_prob < confidence_threshold:
                return "Uncertain", max_prob
            
            if mask_prob > no_mask_prob:
                return "Mask", mask_prob
            else:
                return "No Mask", no_mask_prob
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Error", 0.0

