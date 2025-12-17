"""
Helper script to update overlay_detection.py to use the improved model
"""

import os

def update_overlay_script():
    """Update overlay script to use the improved model"""

    overlay_script = 'overlay_detection.py'
    predict_script = 'predict_air_defense.py'

    old_path = "runs/detect/air_defense_detector/weights/best.pt"
    new_path = "runs/detect/air_defense_detector_v2/weights/best.pt"

    # Check if new model exists
    if not os.path.exists(new_path):
        print(f"ERROR: New model not found at {new_path}")
        print("Please train the improved model first using: python train_yolo11_improved.py")
        return False

    scripts_updated = []

    # Update overlay_detection.py
    if os.path.exists(overlay_script):
        with open(overlay_script, 'r') as f:
            content = f.read()

        if old_path in content:
            content = content.replace(old_path, new_path)
            with open(overlay_script, 'w') as f:
                f.write(content)
            scripts_updated.append(overlay_script)
            print(f"✓ Updated {overlay_script}")
        else:
            print(f"  {overlay_script} already using correct model path")

    # Update predict_air_defense.py
    if os.path.exists(predict_script):
        with open(predict_script, 'r') as f:
            content = f.read()

        if old_path in content:
            content = content.replace(old_path, new_path)
            with open(predict_script, 'w') as f:
                f.write(content)
            scripts_updated.append(predict_script)
            print(f"✓ Updated {predict_script}")
        else:
            print(f"  {predict_script} already using correct model path")

    if scripts_updated:
        print(f"\nSuccessfully updated {len(scripts_updated)} script(s) to use improved model!")
        print(f"New model: {new_path}")
        return True
    else:
        print("\nAll scripts already using the improved model.")
        return True

if __name__ == '__main__':
    print("="*60)
    print("Update Scripts to Use Improved Model")
    print("="*60)
    update_overlay_script()
    print("="*60)
