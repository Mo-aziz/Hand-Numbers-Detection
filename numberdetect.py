import cv2
import os
import HandTrackingModule as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "FingerImages"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))
pTime = 0

detector = htm.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=True)
    # print(lmList)

    if len(lmList) != 0:
        fingers = []

       # For thumb detection (horizontal movement)
       # [1] compares x-coordinates to detect thumb position
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
       # For finger detection (vertical movement)
       # [2] compares y-coordinates to detect if finger is raised
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        # Check for thumb touching different points
        thumb_tip = lmList[4]  # point 4 is the thumb tip
        
        # Define the points we want to check (8: index, 12: middle, 16: ring, 20: pinky)
        points_to_check = {
            6: 20,  # thumb touches pinky (20) for 6
            7: 16,   # thumb touches ring (16) for 7
            8: 12,   # thumb touches middle (12) for 8
            9: 8     # thumb touches index (8) for 9
        }
        
        # Check each point to see if thumb is touching it
        detected_number = None
        for number, point_id in points_to_check.items():
            point = lmList[point_id]
            distance = ((thumb_tip[1] - point[1]) ** 2 + (thumb_tip[2] - point[2]) ** 2) ** 0.5
            if distance < 40:  # Adjust this threshold as needed
                detected_number = number
                break
        
        if detected_number is not None:
            totalFingers = detected_number
        else:
            totalFingers = fingers.count(1)
        print(totalFingers)

        # Make sure we don't exceed the overlay list index
        totalFingers = min(totalFingers, len(overlayList))  
        overlay = overlayList[totalFingers - 1]
        new_w = max(1, overlay.shape[1] // 2)
        new_h = max(1, overlay.shape[0] // 2)
        overlay_small = cv2.resize(overlay, (new_w, new_h))
        h, w, c = overlay_small.shape
        img[0:h, 0:w] = overlay_small

   
    cv2.imshow("Image", img)
    # Break the loop if 'e' or 'E' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('e'), ord('E')):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
