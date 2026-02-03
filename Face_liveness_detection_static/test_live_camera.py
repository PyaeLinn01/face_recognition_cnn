# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test_live_camera.py
# @Software : PyCharm
# Modified to support live camera inference

import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings('ignore')


class LiveCameraTest:
    def __init__(self, model_dir, device_id=0, camera_id=0):
        """
        Initialize live camera testing
        
        Args:
            model_dir: Path to the anti-spoof models directory
            device_id: GPU device id (0/1/2/3)
            camera_id: Camera id to use (0 for default webcam)
        """
        self.model_dir = model_dir
        self.device_id = device_id
        self.camera_id = camera_id
        
        self.model_test = AntiSpoofPredict(device_id)
        self.image_cropper = CropImage()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera with id {camera_id}")
        
        # Set camera resolution for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"Camera initialized successfully (ID: {camera_id})")
        print(f"Device ID: {device_id}")
        print(f"Model Directory: {model_dir}")
        print("\nControls:")
        print("  Press 'q' to quit")
        print("  Press 's' to save screenshot")
        print("-" * 50)

    def process_frame(self, frame):
        """
        Process a single frame and predict face spoofing
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Annotated frame with prediction results
        """
        try:
            # Get face bounding box
            image_bbox = self.model_test.get_bbox(frame)
            
            if image_bbox is None or len(image_bbox) == 0:
                return frame, None
            
            prediction = np.zeros((1, 3))
            test_speed = 0
            
            # Sum predictions from all models
            for model_name in os.listdir(self.model_dir):
                if not model_name.endswith('.pth'):
                    continue
                    
                h_input, w_input, model_type, scale = parse_model_name(model_name)
                param = {
                    "org_img": frame,
                    "bbox": image_bbox,
                    "scale": scale,
                    "out_w": w_input,
                    "out_h": h_input,
                    "crop": True,
                }
                
                if scale is None:
                    param["crop"] = False
                
                img = self.image_cropper.crop(**param)
                
                if img is None:
                    continue
                
                start = time.time()
                prediction += self.model_test.predict(img, os.path.join(self.model_dir, model_name))
                test_speed += time.time() - start
            
            # Get prediction result
            label = np.argmax(prediction)
            value = prediction[0][label] / 2
            
            # Prepare result text and color
            if label == 1:
                result_text = f"Real Face: {value:.2f}"
                color = (0, 255, 0)  # Green for real
                is_real = True
            else:
                result_text = f"Fake Face: {value:.2f}"
                color = (0, 0, 255)  # Red for fake
                is_real = False
            
            # Draw bounding box
            cv2.rectangle(
                frame,
                (image_bbox[0], image_bbox[1]),
                (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
                color, 2)
            
            # Put text on frame
            cv2.putText(
                frame,
                result_text,
                (image_bbox[0], image_bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2)
            
            # Add inference time
            time_text = f"Inference: {test_speed:.2f}s"
            cv2.putText(
                frame,
                time_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1)
            
            return frame, {'label': label, 'score': value, 'is_real': is_real}
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame, None

    def run(self, output_dir=None):
        """
        Run live camera inference loop
        
        Args:
            output_dir: Optional directory to save screenshots
        """
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    print("Failed to read frame from camera")
                    break
                
                # Flip frame horizontally for mirror view
                frame = cv2.flip(frame, 1)
                
                # Process frame
                annotated_frame, result = self.process_frame(frame)
                
                # Display result
                cv2.imshow("Live Face Spoofing Detection", annotated_frame)
                
                # Print result every frame
                if result:
                    status = "REAL" if result['is_real'] else "FAKE"
                    print(f"Frame {frame_count} - {status} (Score: {result['score']:.2f})")
                else:
                    print(f"Frame {frame_count} - No face detected")
                
                frame_count += 1
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s'):
                    # Save screenshot
                    if output_dir:
                        filename = os.path.join(output_dir, f"screenshot_{frame_count}.jpg")
                    else:
                        filename = f"screenshot_{frame_count}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"Screenshot saved: {filename}")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print(f"\nTotal frames processed: {frame_count}")


def main():
    desc = "Live camera face spoofing detection test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model directory containing anti-spoof models")
    parser.add_argument(
        "--camera_id",
        type=int,
        default=0,
        help="camera id to use (0 for default webcam)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="directory to save screenshots (optional)")
    
    args = parser.parse_args()
    
    # Create and run live camera test
    live_test = LiveCameraTest(
        model_dir=args.model_dir,
        device_id=args.device_id,
        camera_id=args.camera_id
    )
    
    live_test.run(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
