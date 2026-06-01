import cv2  #openCV
from cvzone.HandTrackingModule import HandDetector #hand tracking
import numpy as np  #Z axis calculation
import cvzone  #handtracking
import angles  #turret angles
import serial  #usb communication
import pyfirmata

# webcam
camera = cv2.VideoCapture(0)  #turn on webcam
camera.set(3, 1280)           #window width
camera.set(4,720)             #window height

# # USB
# serialcomm = serial.Serial('COM4', 9600)
# serialcomm.timeout = 1

port = "COM3"
board = pyfirmata.Arduino(port)
servo_pinX = board.get_pin('d:9:s') #pin 9 Arduino
servo_pinY = board.get_pin('d:10:s') #pin 10 Arduino

# Hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# find function
x = [193, 153, 129, 107, 97, 86, 78, 70, 64, 59, 56, 50, 49, 46, 43, 40,  38, ] #data set taken "distance_virtual"
y = [20,   25,  30,  35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, ] #data set taken "distance form hand to camera using measuring tape"
coff = np.polyfit(x,y,2)   # y = Ax2 + Bx + C
A, B, C = coff
#print(A, B, C)
A = 0.012636680507237852
B = -2.710541724316941
C = 182.62076069382988

while camera:
    success, img = camera.read()
    hands = detector.findHands(img, draw=False)


    if hands[0] != []:
        lmList = hands[0][0]["lmList"]
        # Landmarks are specific points on the hand, such as the tips of fingers.
        bx, by, bw, bh = hands[0][0]['bbox']
        hand_x1, hand_y1, hand_z1 = lmList[5]   #https://www.geeksforgeeks.org/face-and-hand-landmarks-detection-using-python-mediapipe-opencv/
        hand_x2, hand_y2, hand_z2 = lmList[17]
        center_x = abs((hand_x2+hand_x1)/2)
        center_y = abs((hand_y2+hand_y1)/2)

        distance_virtual = (((hand_x2-hand_x1)**2) + ((hand_y2 - hand_y1)**2))**(1/2)  #dist b/w pt 5 & 17

        # Z axis calculation
        if distance_virtual > 153 :
            Z_real = (-0.125*distance_virtual)+44.125

        elif distance_virtual < 153 and distance_virtual > 107:
            Z_real = (-0.217391304348*distance_virtual) + 58.2608695652

        else:
            Z_real = (A * (distance_virtual**2)) + (B * distance_virtual) + C

        # X, Y axis calculation based on the similar triangle concept
        scrnCenter_x = 512
        scrnCenter_y = 288
        X_virtual = -(center_x - scrnCenter_x)
        Y_virtual = -(center_y - scrnCenter_y)
        try:
            X_real = X_virtual * (6.3/distance_virtual)
            Y_real = Y_virtual * (6.3/distance_virtual)
        except ZeroDivisionError as e:
            X_real = 0
            Y_real = 0
            continue

        cvzone.putTextRect(img,f'{int(X_real)}cm {int(Y_real)}cm {int(Z_real)}cm', (bx, by))
        cv2.circle(img, (int(center_x),int(center_y)), 2, (0,0,255), 2)
        cv2.rectangle(img, (bx-5,by-5), (bx+bw+2, by+bh+2), (255,0,0), 2, 5)

        angle_1 = angles.turret(X_real,Y_real,Z_real)
        angle_1.offsets(12, 0, 7)
        angle_1.getAngles()
        
        motorX = (int(angle_1.getTheta_x()))
        motorY = (int(angle_1.getTheta_y()))

        servo_pinX.write(motorX)
        servo_pinY.write(motorY)
        # # serialcomm.write(motorX.encode())
        # # serialcomm.write(motorY.encode())

    cv2.imshow("win name", img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # Close on ESC key
        cv2.destroyAllWindows()
        break