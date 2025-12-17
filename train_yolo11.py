from ultralytics import YOLO
import torch

def train_air_defense_detector():
    """Train YOLO11 model for air defense detection"""

    # Auto-detect device (GPU if available, otherwise CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Load YOLO11 model (start with pretrained weights)
    model = YOLO('yolo11n.pt')  # yolo11n = nano (fastest), can use yolo11s, yolo11m, yolo11l, yolo11x for more accuracy

    # Train the model
    results = model.train(
        data='yolo_dataset/dataset.yaml',
        epochs=100,  # Number of training epochs
        imgsz=640,  # Image size
        batch=16,  # Batch size (adjust based on your memory)
        name='air_defense_detector',  # Name for this training run
        patience=20,  # Early stopping patience
        save=True,  # Save checkpoints
        device=device,  # Auto-detected device
        workers=8,  # Number of worker threads
        project='runs/detect',  # Project directory
        optimizer='auto',  # Optimizer (auto, SGD, Adam, AdamW, etc.)
        verbose=True,  # Verbose output
        seed=42,  # Random seed for reproducibility
        plots=True,  # Create training plots
    )

    print("\nTraining completed!")
    print(f"Best model saved at: runs/detect/air_defense_detector/weights/best.pt")
    print(f"Last model saved at: runs/detect/air_defense_detector/weights/last.pt")

    # Validate the model
    print("\nValidating model...")
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    return model

if __name__ == '__main__':
    # Check if CUDA is available
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("Training will use CPU (this will be slower)")

    # Train the model
    model = train_air_defense_detector()
