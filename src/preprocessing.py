import cv2 as cv
import random
import numpy as np
import os

def creating_training_data(img_size):
    """Create training data with proper preprocessing"""


    dataDir = "Dataset/"
    classes = ["Mask", "No_Mask"]
    
    if not os.path.exists(dataDir):
        raise FileNotFoundError(f"Dataset directory '{dataDir}' not found!")
    
    training_Data = []
    x = []
    y = []
    
    print("Processing images...")
    total_images = 0
    
    for category in classes:
        path = os.path.join(dataDir, category)
        if not os.path.exists(path):
            print(f"Warning: {path} directory not found!")
            continue
            
        class_num = classes.index(category)
        category_count = 0
        
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                img_array = cv.imread(img_path)
                
                if img_array is None:
                    continue
                
                # Keep as BGR for consistency with OpenCV
                resized_array = cv.resize(img_array, (img_size, img_size))
                training_Data.append([resized_array, class_num])
                category_count += 1
                total_images += 1
                
            except Exception as e:
                continue
        
        print(f"{category}: {category_count} images processed")
    
    if total_images == 0:
        raise ValueError("No images were successfully processed!")
    
    print(f"Total images processed: {total_images}")
    
    # Shuffle the data
    random.shuffle(training_Data)
    
    # Extract features and labels
    for feature, label in training_Data:
        x.append(feature)
        y.append(label)
    
    # Convert to numpy arrays - keep as uint8 initially
    x = np.array(x, dtype=np.uint8)
    y = np.array(y, dtype=np.int64)
    
    print(f"Final data shape: X={x.shape}, y={y.shape}")
    print(f"Data range: [{x.min()}, {x.max()}]")
    print(f"Classes distribution: {np.bincount(y)}")
    
    return x, y

# Run preprocessing
if __name__ == "__main__":
    try:
        X, y = creating_training_data(224)
        
        # Save data
        np.save("x.npy", X)
        np.save("y.npy", y)
        
        print("✅ Data preprocessed and saved!")
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
