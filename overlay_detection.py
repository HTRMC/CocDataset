import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
import time
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk, ImageFont
import win32gui
import win32con
import win32api
from pynput import keyboard
import threading
import ctypes

# Enable DPI awareness for crisp rendering
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
except:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except:
        pass

class TransparentOverlay:
    def __init__(self, model_path, conf_threshold=0.3):
        """
        Create transparent overlay for real-time air defense detection

        Args:
            model_path: Path to trained YOLO model (.pt file)
            conf_threshold: Confidence threshold for detections (0-1)
        """
        print("Loading YOLO model...")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.sct = mss()
        self.running = True
        self.paused = False

        # Colors for each class (RGB format)
        self.colors = {
            0: (255, 0, 0),      # air_defense_9 - Red
            1: (0, 255, 0),      # air_defense_10 - Green
            2: (255, 165, 0),    # air_defense_11 - Orange
            3: (0, 0, 255),      # air_defense_12 - Blue
        }

        # Create tkinter window
        self.root = tk.Tk()
        self.root.title("Air Defense Detector Overlay")

        # Get screen dimensions
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()

        # Configure window
        self.root.geometry(f"{self.screen_width}x{self.screen_height}+0+0")
        self.root.attributes('-transparentcolor', 'white')
        self.root.attributes('-topmost', True)
        self.root.attributes('-fullscreen', True)
        self.root.overrideredirect(True)

        # Create canvas
        self.canvas = tk.Canvas(
            self.root,
            width=self.screen_width,
            height=self.screen_height,
            bg='white',
            highlightthickness=0
        )
        self.canvas.pack()

        # Make window click-through
        self.make_clickthrough()

        # Setup keyboard listener
        self.setup_keyboard_listener()

        print(f"Model loaded! Classes: {self.model.names}")
        print(f"Confidence threshold: {conf_threshold}")
        print("\nOverlay active!")
        print("Hotkeys:")
        print("  F9: Pause/Resume detection")
        print("  F10: Quit")

    def make_clickthrough(self):
        """Make window click-through using Windows API"""
        hwnd = win32gui.FindWindow(None, "Air Defense Detector Overlay")
        if hwnd:
            # Get current window style
            styles = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            # Add transparent and layered styles
            styles = styles | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
            # Set new window style
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, styles)

    def setup_keyboard_listener(self):
        """Setup global keyboard listener for hotkeys"""
        def on_press(key):
            try:
                if key == keyboard.Key.f9:
                    self.paused = not self.paused
                    status = "PAUSED" if self.paused else "RESUMED"
                    print(f"\n{status}")
                elif key == keyboard.Key.f10:
                    print("\nQuitting...")
                    self.running = False
                    self.root.quit()
            except:
                pass

        # Start listener in separate thread
        listener = keyboard.Listener(on_press=on_press)
        listener.daemon = True
        listener.start()

    def draw_detections(self, results):
        """Draw detection boxes on canvas using PIL for crisp text"""
        # Clear canvas
        self.canvas.delete("all")

        # Create transparent image
        img = Image.new('RGBA', (self.screen_width, self.screen_height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        if self.paused:
            # Show paused message
            try:
                font = ImageFont.truetype("arial.ttf", 32)
            except:
                font = ImageFont.load_default()

            draw.text(
                (self.screen_width // 2, 50),
                "PAUSED (F9 to resume, F10 to quit)",
                font=font,
                fill=(255, 0, 0, 255),
                anchor="mm"
            )

            # Convert to PhotoImage and display
            photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, image=photo, anchor="nw")
            self.canvas.image = photo
            return

        # Load fonts
        try:
            label_font = ImageFont.truetype("arialbd.ttf", 14)  # Bold Arial 14
            info_font = ImageFont.truetype("arialbd.ttf", 18)   # Bold Arial 18
        except:
            label_font = ImageFont.load_default()
            info_font = ImageFont.load_default()

        # Draw detection boxes
        total_detections = 0
        if results:
            for result in results:
                boxes = result.boxes
                total_detections += len(boxes)

                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())

                    # Get class name and color
                    class_name = self.model.names[cls]
                    color = self.colors.get(cls, (255, 255, 255))

                    # Draw bounding box with thicker lines
                    draw.rectangle(
                        [x1, y1, x2, y2],
                        outline=color + (255,),
                        width=4
                    )

                    # Create label
                    label = f"{class_name}: {conf:.2f}"

                    # Get text size
                    bbox = draw.textbbox((0, 0), label, font=label_font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    # Draw label background
                    draw.rectangle(
                        [x1, y1 - text_height - 10, x1 + text_width + 10, y1],
                        fill=color + (220,)
                    )

                    # Draw label text
                    draw.text(
                        (x1 + 5, y1 - text_height - 5),
                        label,
                        font=label_font,
                        fill=(255, 255, 255, 255)
                    )

        # Draw detection count with shadow for better visibility
        info_text = f"Detections: {total_detections}"

        # Draw shadow
        draw.text((22, 22), info_text, font=info_font, fill=(0, 0, 0, 200))
        # Draw text
        draw.text((20, 20), info_text, font=info_font, fill=(0, 255, 0, 255))

        # Convert to PhotoImage and display
        photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, image=photo, anchor="nw")
        self.canvas.image = photo  # Keep reference to prevent garbage collection

    def update_frame(self):
        """Capture screen and run detection"""
        if not self.running:
            return

        if not self.paused:
            # Capture screen
            monitor = self.sct.monitors[1]
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

            # Draw detections on overlay
            self.draw_detections(results)
        else:
            # Still show paused state
            self.draw_detections([])

        # Schedule next update (30 FPS)
        if self.running:
            self.root.after(33, self.update_frame)

    def run(self):
        """Start the overlay"""
        # Make window click-through after a short delay
        self.root.after(100, self.make_clickthrough)

        # Start detection loop
        self.root.after(500, self.update_frame)

        # Run tkinter main loop
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            print("Overlay stopped.")

def main():
    """Main function"""
    print("=" * 60)
    print("Transparent Overlay Air Defense Detector")
    print("=" * 60)

    # Model path
    model_path = 'runs/detect/air_defense_detector_v2/weights/best.pt'

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
    print("\nStarting transparent overlay...")
    print("The overlay will appear on top of your screen.")
    print("You can click through it to interact with your game.")

    # Create and run overlay
    overlay = TransparentOverlay(model_path, conf_threshold=conf_threshold)
    overlay.run()

if __name__ == '__main__':
    main()
