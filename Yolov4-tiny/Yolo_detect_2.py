import cv2
import serial
import numpy as np

# Load Yolo
net = cv2.dnn.readNet("/home/hoanghai/tflite1/Run/TFLite-Trash-Detection/Yolov4-tiny/yolov4-tiny-custom_30000.weights", "/home/hoanghai/tflite1/Run/TFLite-Trash-Detection/Yolov4-tiny/yolov4-tiny-custom.cfg")
classes = []
with open("/home/hoanghai/tflite1/Run/TFLite-Trash-Detection/Yolov4-tiny/obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image

def detect_object(img):
    height, width, channels = img.shape

    # object Detection
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    for b in blob:
        for n, img_blob in enumerate(b):
            cv2.imshow(str(n), img_blob)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_ITALIC = 16
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 1)
            cv2.putText(img, label, (x, y + 30), font, 1, (255,0,0), 2)
return img

while True:
    cap = cv2.VideoCapture(0)
    user_input = input("Input:")
    ret, frame = cap.read()
    if user_input == "0":
        if ret:
            cv2.imwrite('baocao1.jpg', frame)
        else:
            print("No Camera")
        image = cv2.imread(r"/home/hoanghai/tflite1/Run/TFLite-Trash-Detection/Yolov4-tiny/baocao1.jpg")
        image = cv2.resize(image, (416, 416))
        print('Processing...')

        img1 = detect_object(image)
        print("Passed")
        cv2.imwrite("result.jpg", img1)
    cap.release()