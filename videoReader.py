import os
import cv2


video = "IMG_0111.MOV"

cap = cv2.VideoCapture(video)
count = 0


if cap.isOpened() == False:
    print("Failed to open")
else:
    while cap.isOpened():
        ret, frame = cap.read()
        count += 1
        if ret:
            cv2.imwrite("myVid5/frame-" + str(count).zfill(6) + ".jpg", frame )
        else:
            break

cap.release()
cv2.destroyAllWindows()
