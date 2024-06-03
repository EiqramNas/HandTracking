import cv2
import mediapipe as mp

mpdraw = mp.solutions.drawing_utils
mpdrawstyle = mp.solutions.drawing_styles
mphands = mp.solutions.hands


cap = cv2.VideoCapture(0)

hands=mphands.Hands()
while True:
    data, image=cap.read()
    
    image=cv2.cvtColor(cv2.flip(image,1),cv2.COLOR_BGR2RGB)
    results=hands.process(image)
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mpdraw.draw_landmarks(
                image,
                hand_landmarks,mphands.HAND_CONNECTIONS
            )
    cv2.imshow('Handtracker', image)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.release()
cv2.destroyAllWindows()
