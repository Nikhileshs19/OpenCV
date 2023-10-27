import cv2
import numpy as np

lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([40, 255, 255])

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.medianBlur(mask, 5)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        if radius > 40:
            cv2.circle(frame, (int(x), int(y)), int(radius), (255, 25, 255), 3)
            cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

    cv2.imshow('frame', frame)
    cv2.imshow('mask',mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()