import cv2
import mediapipe as mp
import numpy as np
import pydirectinput as pyd

mpdraw = mp.solutions.drawing_utils
mphands = mp.solutions.hands
hands = mphands.Hands()
cap = cv2.VideoCapture(0)

def Main():
    state = [False] * 6 
    background = np.zeros([512, 512, 3], np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mpdraw.draw_landmarks(background, hand_landmarks, mphands.HAND_CONNECTIONS)

                for idx, lm in enumerate(hand_landmarks.landmark):
                    h, w, _ = frame.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    
                    if idx == 0:
                        x1, y1 = x, y
                        cv2.putText(background, "IDX0", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160,32,240), 1)
                    elif idx == 4:
                        x2, y2 = x, y
                        cv2.putText(background, "IDX4", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160,32,240), 1)
                    elif idx == 8:
                        x3, y3 = x, y
                        cv2.putText(background, "IDX8", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160,32,240), 1)
                    elif idx == 12:
                        x4, y4 = x, y
                        cv2.putText(background, "IDX12", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160,32,240), 1)
                    elif idx == 16:
                        x5, y5 = x, y
                        cv2.putText(background, "IDX16", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160,32,240), 1)
                    elif idx == 20:
                        x6, y6 = x, y
                        cv2.putText(background, "IDX20", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160,32,240), 1)

                    cv2.putText(background, f"X:{x+w//2} Y:{y+h//2}", (x,y-20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,0), 1)

                # Check distances and manage key states
                try:
                    distances = [x2 - x1, x3 - x1, x4 - x1, x5 - x1 + 60, x6 - x1 + 90]
                    cv2.rectangle(background, (x2,y2), (x1,y1), (0,0,255), 1)
                    print(distances)
                    for i, dist in enumerate(distances):
                        if dist < 80:
                            if not state[i]:  # If key is not pressed
                                pyd.keyDown(str(i + 1))
                                state[i] = True
                        else:
                            if state[i]:  # If key is pressed
                                pyd.keyUp(str(i + 1))
                                state[i] = False

                except Exception as e:
                    print(f"Error: {e}")

        cv2.imshow('frame', background)
        background.fill(0)  # Clear the background
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()  # Release the camera
    cv2.destroyAllWindows()

if __name__ == '__main__':
    Main()
