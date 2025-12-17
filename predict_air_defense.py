from ultralytics import YOLO
import cv2
from pathlib import Path

def predict_air_defense(model_path, image_path, conf_threshold=0.25):
    """
    Predict air defenses in an image using trained YOLO11 model

    Args:
        model_path: Path to trained model weights (.pt file)
        image_path: Path to input image or directory
        conf_threshold: Confidence threshold for detections (0-1)
    """

    # Load trained model
    model = YOLO(model_path)

    # Run prediction
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=True,  # Save annotated images
        save_txt=True,  # Save labels in YOLO format
        save_conf=True,  # Save confidences in labels
        project='runs/detect',
        name='predictions',
        exist_ok=True,
        line_width=2,
        show_labels=True,
        show_conf=True,
    )

    # Print results
    for result in results:
        boxes = result.boxes
        print(f"\nImage: {result.path}")
        print(f"Number of air defenses detected: {len(boxes)}")

        for i, box in enumerate(boxes):
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            class_name = model.names[cls]  # Get class name from model
            xyxy = box.xyxy[0].cpu().numpy()
            print(f"  {class_name} (#{i+1}): confidence={conf:.3f}, bbox=[{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")

    print(f"\nAnnotated images saved to: runs/detect/predictions")

    return results

if __name__ == '__main__':
    # Example usage
    model_path = 'runs/detect/air_defense_detector_v2/weights/best.pt'  # Path to trained model

    # Test on a single image (change this to your test image)
    image_path = 'th14/th14_defence_1.jpg'

    print(f"Running air defense detection...")
    print(f"Model: {model_path}")
    print(f"Image: {image_path}")
    print("-" * 50)

    results = predict_air_defense(model_path, image_path, conf_threshold=0.25)
