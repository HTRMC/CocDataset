import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
import time

class RealtimeDetector:
    def __init__(self, model_path, conf_threshold=0.3):
        """
        Initialize real-time air defense detector

        Args:
            model_path: Path to trained YOLO model (.pt file)
            conf_threshold: Confidence threshold for detections (0-1)
        """
        print("Loading YOLO model...")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.sct = mss()

        # Colors for each class (BGR format for OpenCV)
        self.colors = {
            0: (255, 0, 0),      # air_defense_9 - Blue
            1: (0, 255, 0),      # air_defense_10 - Green
            2: (0, 165, 255),    # air_defense_11 - Orange
            3: (0, 0, 255),      # air_defense_12 - Red
        }

        print(f"Model loaded! Classes: {self.model.names}")
        print(f"Confidence threshold: {conf_threshold}")

    def select_region(self):
        """Let user select screen region to capture"""
        print("\nScreen capture options:")
        print("1. Full screen (primary monitor)")
        print("2. Custom region (select with mouse)")

        choice = input("Choose option (1 or 2): ").strip()

        if choice == "2":
            print("\nClick and drag to select the region to capture.")
            print("Press ENTER when done, ESC to use full screen.")
            return self.get_custom_region()
        else:
            # Full screen of primary monitor
            monitor = self.sct.monitors[1]
            return monitor

    def get_custom_region(self):
        """Allow user to select custom region with mouse"""
        # Capture full screen first
        monitor = self.sct.monitors[1]
        screenshot = self.sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Let user select ROI
        roi = cv2.selectROI("Select Region (press ENTER when done)", img, fromCenter=False)
        cv2.destroyWindow("Select Region (press ENTER when done)")

        if roi[2] > 0 and roi[3] > 0:
            # Return custom monitor region
            return {
                "left": monitor["left"] + roi[0],
                "top": monitor["top"] + roi[1],
                "width": roi[2],
                "height": roi[3]
            }
        else:
            # If no region selected, use full screen
            return monitor

    def draw_boxes(self, img, results):
        """Draw bounding boxes and labels on image"""
        for result in results:
            boxes = result.boxes

            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())

                # Get class name and color
                class_name = self.model.names[cls]
                color = self.colors.get(cls, (255, 255, 255))

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # Create label with class name and confidence
                label = f"{class_name}: {conf:.2f}"

                # Calculate label size and draw background
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                cv2.rectangle(
                    img,
                    (x1, y1 - label_height - baseline - 5),
                    (x1 + label_width, y1),
                    color,
                    -1
                )

                # Draw label text
                cv2.putText(
                    img,
                    label,
                    (x1, y1 - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

        return img

    def run(self, monitor=None, show_fps=True):
        """
        Run real-time detection

        Args:
            monitor: Monitor region to capture (if None, will prompt user)
            show_fps: Whether to show FPS counter
        """
        if monitor is None:
            monitor = self.select_region()

        print(f"\nCapture region: {monitor}")
        print("\nStarting real-time detection...")
        print("Press 'q' to quit, 'p' to pause/unpause")

        fps_counter = []
        paused = False

        try:
            while True:
                if not paused:
                    loop_start = time.time()

                    # Capture screen
                    screenshot = self.sct.grab(monitor)
                    img = np.array(screenshot)
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                    # Run YOLO detection
                    results = self.model.predict(
                        img,
                        conf=self.conf_threshold,
                        verbose=False,
                        device='cuda' if self.model.device.type == 'cuda' else 'cpu'
                    )

                    # Draw boxes
                    img = self.draw_boxes(img, results)

                    # Calculate and display FPS
                    if show_fps:
                        loop_time = time.time() - loop_start
                        fps = 1 / loop_time if loop_time > 0 else 0
                        fps_counter.append(fps)

                        # Keep only last 30 frames for average
                        if len(fps_counter) > 30:
                            fps_counter.pop(0)

                        avg_fps = sum(fps_counter) / len(fps_counter)

                        # Draw FPS counter
                        fps_text = f"FPS: {avg_fps:.1f}"
                        cv2.putText(
                            img,
                            fps_text,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )

                    # Count detections
                    total_detections = sum(len(result.boxes) for result in results)
                    detection_text = f"Detections: {total_detections}"
                    cv2.putText(
                        img,
                        detection_text,
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

                # Display image
                cv2.imshow('Air Defense Detector - Press Q to quit, P to pause', img)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('p'):
                    paused = not paused
                    if paused:
                        print("PAUSED - Press 'p' to resume")
                    else:
                        print("RESUMED")

        except KeyboardInterrupt:
            print("\nStopped by user")

        finally:
            cv2.destroyAllWindows()
            print("Detection stopped.")

def main():
    """Main function"""
    print("=" * 60)
    print("Real-Time Air Defense Detector")
    print("=" * 60)

    # Model path
    model_path = 'runs/detect/air_defense_detector/weights/best.pt'

    # Check if model exists
    import os
    if not os.path.exists(model_path):
        print(f"\nERROR: Model not found at {model_path}")
        print("Please train the model first using: python train_yolo11.py")
        return

    # Confidence threshold
    conf_threshold = 0.3
    print(f"\nUsing model: {model_path}")
    print(f"Confidence threshold: {conf_threshold}")

    # Create detector
    detector = RealtimeDetector(model_path, conf_threshold=conf_threshold)

    # Run detection
    detector.run(show_fps=True)

if __name__ == '__main__':
    main()
