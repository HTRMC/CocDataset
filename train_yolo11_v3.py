from ultralytics import YOLO
import torch

def train_air_defense_detector_v3():
    """
    Train YOLO11 v3 model for air defense detection

    Improvements over v2:
    - Using YOLO11n (nano) - faster, worked better for you
    - 95/5 train/val split (314 train images instead of 264)
    - Longer training (200 epochs)
    - Optimized hyperparameters for small dataset
    """

    # Auto-detect device (GPU if available, otherwise CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load YOLO11n model (nano - fast and effective)
    print("\nLoading YOLO11n model...")
    model = YOLO('yolo11n.pt')

    print("\n" + "="*60)
    print("TRAINING CONFIGURATION V3")
    print("="*60)
    print("Model: YOLO11n (Nano - Fast & Effective)")
    print("Training images: 314 (95% of labeled data)")
    print("Validation images: 17 (5% of labeled data)")
    print("Epochs: 200 (with early stopping)")
    print("Image size: 640")
    print("Batch size: 16")
    print("="*60 + "\n")

    # Train the model
    results = model.train(
        # Data
        data='yolo_dataset/dataset.yaml',

        # Training duration
        epochs=200,
        patience=40,  # Increased patience - let it train longer

        # Image settings
        imgsz=640,
        batch=16,

        # Save settings
        name='air_defense_detector_v3',
        save=True,
        save_period=20,  # Save checkpoint every 20 epochs

        # Device
        device=device,
        workers=8,
        project='runs/detect',

        # Optimizer settings
        optimizer='AdamW',
        lr0=0.001,       # Initial learning rate
        lrf=0.001,       # Final learning rate
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5.0,

        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Data augmentation - aggressive for small dataset
        hsv_h=0.02,       # HSV-Hue augmentation
        hsv_s=0.7,        # HSV-Saturation augmentation
        hsv_v=0.4,        # HSV-Value augmentation
        degrees=15.0,     # Rotation (+/- deg)
        translate=0.15,   # Translation (+/- fraction)
        scale=0.6,        # Scaling (+/- gain)
        shear=3.0,        # Shear (+/- deg)
        perspective=0.0,  # Perspective (+/- fraction)
        flipud=0.0,       # Flip up-down (probability)
        fliplr=0.5,       # Flip left-right (probability)
        mosaic=1.0,       # Mosaic augmentation (probability)
        mixup=0.15,       # MixUp augmentation (probability)
        copy_paste=0.15,  # Copy-paste augmentation (probability)
        erasing=0.4,      # Random erasing probability
        auto_augment='randaugment',  # Auto augmentation

        # Other settings
        verbose=True,
        seed=42,
        deterministic=True,
        plots=True,
        amp=True,  # Automatic Mixed Precision

        # Validation
        val=True,
        close_mosaic=15,  # Disable mosaic in last 15 epochs
    )

    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Best model: runs/detect/air_defense_detector_v3/weights/best.pt")
    print(f"Last model: runs/detect/air_defense_detector_v3/weights/last.pt")
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
    print("="*60)

    # Train the model
    model = train_air_defense_detector_v3()

    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Check training plots in: runs/detect/air_defense_detector_v3/")
    print("2. To use this model, run:")
    print("   python -c \"import shutil; shutil.copy('runs/detect/air_defense_detector_v3/weights/best.pt', 'runs/detect/air_defense_detector_v2/weights/best.pt')\"")
    print("3. Then run: python overlay_detection.py")
    print("="*60)
