
from collections import OrderedDict
import numpy as np
import cv2
import dlib
import imutils

CHEEK_IDXS = OrderedDict([("whole_face", (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,27,26,25,24,23,22,21,20,19,18)),
                          ("left_cheek", (1,17,18,19,20,21,22,23,24,25,26,15,28)),
                          ("right_eye", (36,37,38,39,40,41)),
                        ("left_eye", (42,43,44,45,46,47)),
                        ("right_cheek", (49,50,51,52,53,54,55,56,57,58,59,60))
                        
                         ])

                         

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


img = cv2.imread('model.jpg')
img = imutils.resize(img, width=600)


overlay = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

overlay2 = img.copy()
gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

detections = detector(gray, 0)
for k,d in enumerate(detections):
    shape = predictor(gray, d)
    for (_, name) in enumerate(CHEEK_IDXS.keys()):
        pts = np.zeros((len(CHEEK_IDXS[name]), 2), np.int32) 
        for i,j in enumerate(CHEEK_IDXS[name]): 
            pts[i] = [shape.part(j).x, shape.part(j).y]
        
        pts = pts.reshape((-1,1,2))
        color = (0, 0, 0)
        thickness = 10
        if name=="right_cheek":
            color = (255, 255, 255)
        if name=="left_cheek":
            color = (255, 255, 255)
        if name=="whole_face":
            thickness = 800

        
        cv2.fillPoly(overlay,[pts],color)
        cv2.polylines(overlay,[pts],True,color,thickness)
        
    
    cv2.imshow("Image", overlay)
    status = cv2.imwrite('python_output.png',overlay)
    print(name)
    cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break



