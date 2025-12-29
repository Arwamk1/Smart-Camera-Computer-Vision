import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time

class SmartCamera:
    def __init__(self, model_path='yolov8n.pt', source=0):
        """
        Initialize the Smart Camera with YOLO model and video source.
        
        Args:
            model_path (str): Path to the YOLOv8 model weights.
            source (int/str): Video source (0 for webcam, or video file path).
        """
        print(f"Loading YOLO model: {model_path}...")
        self.model = YOLO(model_path)
        self.source = source
        
        # Tracking history to calculate movement
        self.track_history = defaultdict(lambda: [])
        self.movement_threshold = 10  # Pixels of movement to consider "moving"
        self.history_length = 20      # Number of frames to keep in history
        
        # Visualization settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.colors = {
            'standing': (0, 0, 255),  # Red for standing
            'moving': (0, 255, 0),    # Green for moving
            'text': (255, 255, 255)   # White for text
        }

    def calculate_movement_status(self, track_id, current_center):
        """
        Determine if the object is moving or standing based on position history.
        
        Args:
            track_id (int): The unique ID of the tracked object.
            current_center (tuple): (x, y) coordinates of the current center.
            
        Returns:
            str: "Moving" or "Standing"
        """
        history = self.track_history[track_id]
        history.append(current_center)
        
        # Keep history limited to a certain number of frames
        if len(history) > self.history_length:
            history.pop(0)
            
        if len(history) < 5:
            return "Analyzing..."
            
        # Calculate displacement between oldest and newest point in history
        start_point = history[0]
        end_point = history[-1]
        displacement = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
        
        if displacement > self.movement_threshold:
            return "Moving"
        else:
            return "Standing"

    def run(self):
        """
        Start the video processing loop.
        """
        cap = cv2.VideoCapture(self.source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {self.source}")
            return

        print("Starting Smart Camera... Press 'q' to quit.")

        while True:
            success, frame = cap.read()
            if not success:
                break

            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            # classes=0 ensures we only detect people (class 0 in COCO dataset)
            results = self.model.track(frame, persist=True, classes=0, verbose=False)

            if results[0].boxes.id is not None:
                # Get the boxes and track IDs
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    center = (float(x), float(y))
                    
                    # Determine status
                    status = self.calculate_movement_status(track_id, center)
                    
                    # Visuals
                    color = self.colors['moving'] if status == "Moving" else self.colors['standing']
                    
                    # Draw bounding box (convert xywh center format to top-left xy format for plotting)
                    x1 = int(x - w/2)
                    y1 = int(y - h/2)
                    x2 = int(x + w/2)
                    y2 = int(y + h/2)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw interactive overlay
                    label = f"ID: {track_id} | {status}"
                    
                    # Text background for better readability
                    (text_w, text_h), _ = cv2.getTextSize(label, self.font, 0.7, 2)
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + text_w, y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5), self.font, 0.7, self.colors['text'], 2)
                    
                    # Draw trail for visualization of movement
                    history = self.track_history[track_id]
                    points = np.hstack(history).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)

            # Display system status
            cv2.putText(frame, "Smart Camera: Active", (10, 30), self.font, 1, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to exit", (10, 60), self.font, 0.6, (200, 200, 200), 1)

            cv2.imshow("AI Smart Camera - Movement Detection", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Case 1: Real-time Camera (Webcam)
    # app = SmartCamera(source=0)
    # app.run()

    # Case 2: Run on a Video File
    # Ensure you have a video file named 'test_video.mp4' in the project directory
    app = SmartCamera(source='test_video.mp4')
    app.run()
