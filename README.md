# Suspicious (Shoplifting) Activity Detection

## Introduction

In retail environments, shoplifting and other suspicious activities can result in significant losses. This project leverages advanced computer vision techniques to detect potentially suspicious activities, helping store staff quickly identify and respond to such events. Using YOLOv8 for pose estimation, the model captures human movements in real time, which are then analyzed for specific actions that may indicate suspicious behavior.

For a detailed overview, you can read my blog on Medium: [Suspicious Activity Detection Using Pose Estimation and Action Recognition](https://medium.com/@sg.sparsh06/suspicious-activity-shoplifting-detection-using-yolov8-pose-estimation-and-classification-b59fd73cdba3).

## DEMO

[![Suspicious Activity Detection Demo](https://github.com/user-attachments/assets/355be203-402e-4648-9f43-f8e4b67026e8)](https://www.youtube.com/watch?v=oL9wGJXr-SM)

Click the image above to watch a demo of the project on YouTube.


## Features

- **Pose Estimation**: Uses YOLOv8 for accurate pose estimation of multiple individuals.
- **Action Recognition**: Identifies specific actions that may signify suspicious behavior.
- **Real-Time Detection**: Processes live camera feeds and detects actions in real-time.
- **Customizable Detection Threshold**: Allows users to adjust sensitivity and detection thresholds.
- **Email Alerts**: Triggers email alerts when the confidence of suspicious activity exceeds a user-defined threshold, notifying staff to investigate.

## Technologies Used

- **YOLOv8** for Pose Estimation
- **Action Recognition ** using XGBoost
- **Python** and **PyTorch** for model development
- **OpenCV** for video processing and visualization
- **NumPy**, **Pandas** for data handling and preprocessing
- **MIME** for sending email alerts
