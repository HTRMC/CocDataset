from ultralytics import YOLO
import torch

def train_air_defense_detector_improved():
    """
    Train improved YOLO11 model for air defense detection

    Improvements:
    - Using YOLO11s (small) instead of nano for better accuracy
    - More epochs (150)
    - Optimized hyperparameters
    - Better data augmentation
    """

    # Auto-detect device (GPU if available, otherwise CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load YOLO11s model (small - better accuracy than nano)
    print("\nLoading YOLO11s model...")
    model = YOLO('yolo11s.pt')  # Will auto-download if not present

    print("\n" + "="*60)
    print("IMPROVED TRAINING CONFIGURATION")
    print("="*60)
    print("Model: YOLO11s (Small - Better accuracy than nano)")
    print("Epochs: 150 (with early stopping)")
    print("Image size: 640")
    print("Batch size: 16")
    print("Data augmentation: Enhanced")
    print("="*60 + "\n")

    # Train the model with improved parameters
    results = model.train(
        # Data
        data='yolo_dataset/dataset.yaml',

        # Training duration
        epochs=150,
        patience=30,  # Increased patience for early stopping

        # Image settings
        imgsz=640,
        batch=16,  # Adjust down to 8 if you get memory errors

        # Save settings
        name='air_defense_detector_v2',
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs

        # Device
        device=device,
        workers=8,
        project='runs/detect',

        # Optimizer
        optimizer='AdamW',  # AdamW often works better than auto
        lr0=0.001,  # Initial learning rate
        lrf=0.01,   # Final learning rate (lr0 * lrf)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5.0,  # Warmup for 5 epochs

        # Loss weights (optimized for detection)
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Data augmentation (enhanced)
        hsv_h=0.015,      # HSV-Hue augmentation
        hsv_s=0.7,        # HSV-Saturation augmentation
        hsv_v=0.4,        # HSV-Value augmentation
        degrees=10.0,     # Rotation (+/- deg)
        translate=0.1,    # Translation (+/- fraction)
        scale=0.5,        # Scaling (+/- gain)
        shear=2.0,        # Shear (+/- deg)
        perspective=0.0,  # Perspective (+/- fraction)
        flipud=0.0,       # Flip up-down (probability)
        fliplr=0.5,       # Flip left-right (probability)
        mosaic=1.0,       # Mosaic augmentation (probability)
        mixup=0.1,        # MixUp augmentation (probability)
        copy_paste=0.1,   # Copy-paste augmentation (probability)

        # Other settings
        verbose=True,
        seed=42,
        deterministic=True,
        plots=True,
        amp=True,  # Automatic Mixed Precision for faster training

        # Validation
        val=True,
        close_mosaic=10,  # Disable mosaic augmentation in last N epochs
    )

    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Best model: runs/detect/air_defense_detector_v2/weights/best.pt")
    print(f"Last model: runs/detect/air_defense_detector_v2/weights/last.pt")
    print("="*60)

    # Validate the model
    print("\nRunning validation on best model...")
    metrics = model.val()

    print("\n" + "="*60)
    print("VALIDATION METRICS")
    print("="*60)
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP75:    {metrics.box.map75:.4f}")
    print("\nPer-class metrics:")
    for i, name in enumerate(model.names.values()):
        if i < len(metrics.box.maps):
            print(f"  {name:20s} mAP50-95: {metrics.box.maps[i]:.4f}")
    print("="*60)

    return model

if __name__ == '__main__':
    # Check CUDA availability
    import torch
    print("="*60)
    print("SYSTEM CHECK")
    print("="*60)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("WARNING: Training will use CPU (this will be MUCH slower)")
        print("If you have a GPU, install CUDA-enabled PyTorch")
    print("="*60)

    # Train the model
    model = train_air_defense_detector_improved()

    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Check training plots in: runs/detect/air_defense_detector_v2/")
    print("2. Update overlay_detection.py to use the new model:")
    print("   Change: 'runs/detect/air_defense_detector/weights/best.pt'")
    print("   To:     'runs/detect/air_defense_detector_v2/weights/best.pt'")
    print("3. Run: python overlay_detection.py")
    print("="*60)
