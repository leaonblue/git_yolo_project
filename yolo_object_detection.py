import cv2
import numpy as np
import math
import requests, json 

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
print(layer_names)
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
img = cv2.imread("others.jpg")
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
print(boxes)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print("indexes: ", indexes.reshape(-1))

font = cv2.FONT_HERSHEY_PLAIN

URL = ""
person_figure = 0
h = 2 #m
θ_C = math.pi/4
θ_V = math.pi/12
θ_H = math.pi/6
O1_O2 = h*math.tan(θ_C-θ_V)
O1_O4 = h*math.tan(θ_C+θ_V)
O2_O5 = h/math.cos(θ_C-θ_V)
O4_O5 = h/math.cos(θ_C+θ_V)
L = O1_O4-O1_O2
A_O2 = O2_O5*math.tan(θ_H)
D_O4 = O4_O5*math.tan(θ_H)
S = (A_O2+D_O4)*L
print(S,"㎡")

stand_points = []

for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        
        label = str(classes[class_ids[i]])
        if label == 'person':
            person_figure = person_figure+1
        color = colors[i]
        
        stand_point = [int(x+w/2), y+h]
        stand_points.append(stand_point)

        cv2.line(img, (stand_point[0], stand_point[1]), (stand_point[0], stand_point[1]), (0, 0, 0), 5)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        #cv2.putText(img, label, (x, y + 30), font, 2, color, 2)
            
print("person_figure: %d" % person_figure)
print("saturation rate(person/1㎡): %f" % float(person_figure/S))
cv2.putText(img, "saturation: %f" % float(person_figure/S), (0, 0 + 15), font, 1, (0,0,0), 2)

data = {
    "person_data": person_figure,
    "saturation_data": person_figure/S,
    "api_data": "test"
    }
res = requests.post(URL, data=json.dumps(data))

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()