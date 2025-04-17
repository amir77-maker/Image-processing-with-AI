# Image-processing-with-AI

# Real-Time Object & Face Detection with Distance Estimation

This project uses **YOLOv8** and **MTCNN** to perform real-time object and face detection from a webcam feed, along with basic **distance estimation** based on bounding box width.

##  Features

-  Real-time object detection using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
-  Face detection using [MTCNN (FaceNet PyTorch)](https://github.com/timesler/facenet-pytorch)
-  Estimates real-world object distance using a simple pinhole camera model
-  Multithreaded for smoother performance
-  Custom annotations like "So far!" for distant people

##  Requirements

Install the dependencies using pip:

```bash
pip install ultralytics opencv-python torch torchvision facenet-pytorch numpy

Make sure you have:
A webcam connected
A CUDA-compatible GPU (optional but recommended for better performance)

How it works
1_ Captures webcam feed using OpenCV
2_ Uses YOLOv8 for object detection (e.g., person, cellphone)
3_ Uses MTCNN for detecting human faces
4_ Calculates approximate distance using:
distance = (known_width * focal_length) / box_width

5_Annotates frame with:
Bounding boxes
Labels and confidence scores
Distance to objects
Special alert ("So far!") if person is more than 2m away

Configuration
You can adjust the YOLO model by changing this line:
model = YOLO('yolov8n.pt')
to another model like 'yolov8s.pt' for more accuracy.

Modify the constants:
KNOWN_WIDTH = 0.5  # meters
FOCAL_LENGTH = 700  # mm


Sample Output:

https://github.com/user-attachments/assets/b5107149-a5fd-4790-905f-d0ada063936d


 Notes
Distance estimation is approximate and assumes consistent camera parameters
MTCNN may be slower on CPU-only machines — consider switching to a GPU

Run
python your_script_name.py
Press Q to exit the window

License
This project is for educational and research purposes. Feel free to modify and use it!

