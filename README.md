# Computer-vision
Volume-Brightness control
# 🖐️ HandyControl: Gesture-Based Volume and Brightness Control

## Overview

HandyControl is an innovative Python application that allows you to control your computer's volume and screen brightness using hand gestures through your webcam. Using MediaPipe hand tracking, you can intuitively adjust system settings with simple hand movements.

## 🌟 Features

- **Volume Control**: Adjust system volume using left hand gestures
- **Brightness Control**: Modify screen brightness using right hand gestures
- Real-time visual feedback with volume and brightness bars
- Works with multiple operating systems
- Supports single and dual-hand interactions

## 🛠️ Prerequisites

### Hardware
- Webcam
- Computer running Windows, macOS, or Linux

### Software
- Python 3.7+
- OpenCV
- MediaPipe
- PyCaw
- Screen Brightness Control Library

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/HandyControl.git
cd HandyControl
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## 📦 Required Libraries

Install the following libraries:
```bash
pip install opencv-python
pip install mediapipe
pip install pycaw
pip install comtypes
pip install screen-brightness-control
```

## 🎮 How to Use

### Gesture Controls

#### Volume Control (Left Hand)
- **Thumb and Index Finger**: Adjust volume
- Fingers closer together = Lower volume
- Fingers spread apart = Higher volume

#### Brightness Control (Right Hand)
- **Middle and Ring Finger**: Adjust screen brightness
- Fingers closer together = Lower brightness
- Fingers spread apart = Higher brightness

### Running the Application
```bash
python hand_control.py
```

### Exit
- Press 'q' to quit the application

## 🚨 Troubleshooting

- Ensure your webcam is working correctly
- Good lighting helps with hand detection
- Keep hands fully visible in the camera frame
- Some systems might require additional permissions for brightness control

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 🙌 Acknowledgements

- [MediaPipe](https://learnopencv.com/introduction-to-mediapipe/)
- [OpenCV](https://opencv.org/)
- [PyCaw](https://pycaw.readthedocs.io/en/latest/)

---

**Happy Gesturing! 🖐️✨**
