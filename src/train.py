import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import json
from sklearn.model_selection import train_test_split

def main():
    try:
        # Check if preprocessed data exists
        if not os.path.exists("x.npy") or not os.path.exists("y.npy"):
            print("❌ Preprocessed data not found! Please run preprocessing.py first.")
            return
        
        # Loading preprocessed data
        print("Loading data...")
        X = np.load("x.npy")
        y = np.load("y.npy")
        
        print(f"Data loaded: X={X.shape}, y={y.shape}")
        print(f"Data range: [{X.min()}, {X.max()}]")
        print(f"Classes distribution: {np.bincount(y)}")
        
        # Split into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")
        
        # preprocessing pipeline
        def preprocess_data(X_batch):
            # 1. Convert BGR to RGB and normalize to [0,1]
            X_batch = X_batch.astype(np.float32)
            X_batch = X_batch[..., ::-1]  # BGR to RGB
            X_batch = X_batch / 255.0
            
            # 2. Transpose to (N, C, H, W)
            X_batch = np.transpose(X_batch, (0, 3, 1, 2))
            
            # 3. ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
            X_batch = (X_batch - mean) / std
            
            return X_batch
        
        # Preprocess training and validation data
        X_train = preprocess_data(X_train)
        X_val = preprocess_data(X_val)
        
        print(f"After preprocessing: Train={X_train.shape}, Val={X_val.shape}")
        print(f"Train data range: [{X_train.min():.3f}, {X_train.max():.3f}]")
        
        # Model setup
        print("Setting up model...")
        model = models.mobilenet_v3_small(pretrained=True)
        
        num_classes = 2
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        
        # Freeze base layers
        for param in model.features.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier parameters
        for param in model.classifier.parameters():
            param.requires_grad = True
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
        
        # Move model to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Using device: {device}")
        
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Training loop with validation
        print("Starting training...")
        epochs = 10  # Increased epochs
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            train_loss = running_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            val_acc = 100 * correct / total
            
            print(f"Epoch [{epoch+1}/{epochs}]:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), "models/best_mobilenetv3_mask_detection.pth")
                print(f"  ✅ New best model saved! (Val Acc: {val_acc:.2f}%)")
            
            print("-" * 50)
        
        # Save final model
        torch.save(model.state_dict(), "models/mobilenetv3_mask_detection.pth")
        
        # Save metadata
        metadata = {
            "model_architecture": "MobileNetV3-Small",
            "input_size": [3, 224, 224],
            "classes": ["Mask", "No_Mask"],
            "num_classes": 2,
            "preprocessing": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            },
            "training_params": {
                "epochs": epochs,
                "batch_size": 32,
                "learning_rate": 0.001,
                "optimizer": "Adam",
                "loss_function": "CrossEntropyLoss"
            },
            "best_validation_accuracy": best_val_acc
        }
        
        with open("models/metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
        
        print(f"✅ Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
