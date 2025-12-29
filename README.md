# Smart Camera - Computer Vision AI

A real-time computer vision project that detects people, tracks their movement status (Standing vs. Moving), and displays interactive overlays using **YOLOv8** and **OpenCV**.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-orange.svg)

##  Features

- **Real-time Person Detection**: Uses YOLOv8 nano model for fast and accurate human detection.
- **Movement Tracking**: Analyzes position history to classify subjects as "Moving" or "Standing".
- **Interactive Overlay**: Displays bounding boxes, ID tags, status labels, and movement trails.
- **Robust Tracking**: Uses BoT-SORT/ByteTrack (built into YOLOv8) to maintain unique IDs for individuals.

##  Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Smart-Camera-Computer-Vision.git
   cd Smart-Camera-Computer-Vision
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Required packages: `ultralytics`, `opencv-python`, `numpy`*

##  Usage

Run the main script to start the camera feed:

```bash
python main.py
```

- **Press 'q'** to quit the application.
- By default, it uses your primary webcam (`source=0`). You can modify `main.py` to use a video file instead.

##  How It Works

1. **Detection**: The YOLOv8 model scans each frame for objects of class `person`.
2. **Tracking**: The system assigns a unique ID to each detected person.
3. **State Analysis**:
   - The system stores the last 20 positions (centroids) of each person.
   - It calculates the displacement between the current position and the position 20 frames ago.
   - If the displacement exceeds a threshold, the status is **"Moving"**.
   - Otherwise, the status is **"Standing"**.
4. **Visualization**: OpenCV draws the bounding boxes, status text, and a trail of previous positions.

##  Project Structure

```
├── main.py              # Main application script
├── requirements.txt     # Python dependencies
├── yolov8n.pt           # YOLOv8 model weights (downloaded automatically)
└── README.md            # Project documentation
```

##  Future Improvements

- Add zone-based monitoring (e.g., alert if someone enters a specific area).
- Implement counting (total people passed).
- Save snapshots of "Moving" events.
