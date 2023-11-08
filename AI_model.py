import numpy as np
import cv2
import time

start_time = time.time()

image_path = 'top.jpg'
prototxt_path = 'Model/MobileNetSSD_deploy.prototxt'
model_path = 'Model/MobileNetSSD_deploy.caffemodel'
min_confidence = 0.5

classes = ["background","aeroplene", "bicycle","bird", "boat","bottle", "bus",
           " car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
           "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


np.random.seed(543210)
colors = np.random.uniform(0,255, size=(len(classes),3))

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

cap = cv2.VideoCapture(0)

while True:
    _, image = cap.read()
    height, width = image.shape[0], image.shape[1]
    blob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)), 0.007, (300,300),130)

    net.setInput(blob)
    detected_objects = net.forward()
    
    for i in range(detected_objects.shape[2]):
        confidence = detected_objects[0][0][i][2]
    
        if confidence>min_confidence and detected_objects[0][0][0][1] == 15:
            print("Human was detected")
    end_time = time.time()
    
    cv2.waitKey(100)
    if abs(start_time-end_time) > 10:
        break
cap.release()