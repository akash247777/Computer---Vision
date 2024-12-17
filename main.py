import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc

# solution APIs
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Volume Control Library Usage 
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]

# Webcam Setup
wCam, hCam = 640, 480
cam = cv2.VideoCapture(0)
cam.set(3, wCam)
cam.set(4, hCam)

# Mediapipe Hand Landmark Model
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2) as hands:

    while cam.isOpened():
        success, image = cam.read()
        if not success:
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Initialize variables
        volBar = 400
        volPer = 0
        brightBar = 400
        brightPer = 0

        if results.multi_hand_landmarks and results.multi_handedness:
            # Track left and right hands
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Get hand type (left or right)
                hand_type = handedness.classification[0].label

                # Process landmark list
                lmList = []
                h, w, c = image.shape
                for id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])

                if len(lmList) != 0:
                    # Volume Control (Left Hand) - Thumb and Index finger
                    if hand_type == 'Left':
                        x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tiP
                        x2, y2 = lmList[8][1], lmList[8][2]  # Index finger tip

                        # Mark key points
                        cv2.circle(image, (x1,y1), 15, (255,255,255), -1)  
                        cv2.circle(image, (x2,y2), 15, (255,255,255), -1)

                        # Volume Control
                        volLength = math.hypot(x2-x1, y2-y1)
                        if volLength < 50:
                            cv2.line(image, (x1,y1), (x2,y2), (0,0,255), 3)

                        vol = np.interp(volLength, [50, 220], [minVol, maxVol])
                        volume.SetMasterVolumeLevel(vol, None)
                        volBar = np.interp(volLength, [50, 220], [400, 150])
                        volPer = np.interp(volLength, [50, 220], [0, 100])

                        # Volume Bar
                        cv2.rectangle(image, (50, 150), (85, 400), (0, 0, 0), 3)
                        cv2.rectangle(image, (50, int(volBar)), (85, 400), (0, 0, 0), cv2.FILLED)
                        cv2.putText(image, f'Vol: {int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)

                    # Brightness Control (Right Hand) - Middle and Ring finger
                    elif hand_type == 'Right':
                        x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tip
                        x2, y2 = lmList[8][1], lmList[8][2]  # Index finger tip

                        # Mark key points
                        cv2.circle(image, (x1,y1), 15, (255,255,255), -1)  
                        cv2.circle(image, (x2,y2), 15, (255,255,255), -1)

                        # Brightness Control
                        brightLength = math.hypot(x2-x2, y2-y1)
                        try:
                            # Convert brightness range (typically 0-100)
                            bright = np.interp(brightLength, [50, 220], [0, 100])
                            sbc.set_brightness(bright)
                            brightBar = np.interp(brightLength, [50, 220], [400, 150])
                            brightPer = np.interp(brightLength, [50, 220], [0, 100])
                        except Exception as e:
                            print(f"Brightness control error: {e}")

                        # Brightness Bar
                        cv2.rectangle(image, (600, 150), (635, 400), (0, 0, 0), 3)
                        cv2.rectangle(image, (600, int(brightBar)), (635, 400), (0, 0, 0), cv2.FILLED)
                        cv2.putText(image, f'Bright: {int(brightPer)} %', (540, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)
        
        cv2.imshow('Dual Hand Gesture Control', image) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()