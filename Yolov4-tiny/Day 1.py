import numpy as np
import cv2
import serial
import gmail
import time

#serial_port = '/dev/ttyUSB0'
#baud_rate = 9600


#ser = serial.Serial(serial_port, baud_rate, timeout=1)
weights = "D:\AI Challenge\Yolov4 Tiny\yolov4-tiny ver4\yolov4-tiny-custom_30000.weights"
config = "D:\AI Challenge\Yolov4 Tiny\yolov4-tiny ver4\yolov4-tiny-custom.cfg"
labels = "D:\AI Challenge\Yolov4 Tiny\yolov4-tiny ver4\obj.names"


with open(labels, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

        colors = np.random.uniform(0, 255, size=(len(labels), 3))

        net = cv2.dnn.readNet(weights,config)

        layer_names = net.getLayerNames()

        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


def detect_objects(image, confidence=0.6, nms_thresh=0.3):
    left_line_x = image.shape[1] // 3
    middle_line_x = (image.shape[1] // 3) * 2
    right_line_x = image.shape[1]
    trongtam_x = image.shape[1] // 2
    # Vẽ các đường thẳng
    cv2.line(image, (left_line_x, 0), (left_line_x, image.shape[0]), (255, 0, 0), 2)
    cv2.line(image, (middle_line_x, 0), (middle_line_x, image.shape[0]), (255, 0, 0), 2)
    cv2.line(image, (right_line_x, 0), (right_line_x, image.shape[0]), (255, 0, 0), 2)

    Height, Width = image.shape[:2]
    scale = 0.00392

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True,
                                     crop=False)

    net.setInput(blob)

    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []


    for out in outs:
      for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            max_conf = scores[class_id]
            if max_conf > confidence:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - (w / 2)
                y = center_y - (h / 2)
                class_ids.append(class_id)
                confidences.append(float(max_conf))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence, nms_thresh)

    bbox = []
    label = []
    conf = []
    max_detect = 0  # Biến lưu giữ Bounding Box lớn nhất
    max_trongtam = 0  # Biến lưu giữ tọa độ trọng tâm của Bounding Box lớn nhất
    distance = ""
    for i in indices:
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        width = int(w)
        height = int(h)
        square = width * height
        trongtam = (int(x) + int(w) // 2)
        print(f"Bounding Box: {i}: {width}")
        print(f"Tọa độ trọng tâm {i} là: ", trongtam)

        bbox.append([int(x), int(y), int(x + w), int(y + h)])
        label.append(str(labels[class_ids[i]]))
        conf.append(confidences[i])

        if (square) > max_detect:
            max_detect = square
            max_trongtam = trongtam
            phanloai = str(labels[class_ids[i]])

        if (max_detect == 0):
            distance = ""
        elif (max_detect > 80):
            distance = "CLOSE"
        else:
            distance = "FAR"

    print("Bounding Box Max: ", max_detect)
    print("Trọng tâm của Bounding Box lớn nhất: ", max_trongtam)

    dolech =  max_trongtam - trongtam_x
    print("Độ lệch: ", dolech)
    if distance == "":
        print("NO OBJECT")
    if distance == "FAR":
        #ser.write("20".encode())
        if abs(dolech) <= 72:
            print("MIDDLE")
        elif (dolech < -73):
            print("TURN LEFT")
        elif (dolech > 73):
            print("TURN RIGHT")
    elif distance == "CLOSE":
        print("Phan loai: ",phanloai)
    return bbox, label, conf    
            
def draw_bbox(img, bbox, labels, confidence, colors=None, write_conf=False):
    if colors is None:
        colors = [(0, 255, 0)] * len(labels)

    for i, label in enumerate(labels):
        color = colors[i]  

        if write_conf:
            label += ' ' + str(format(confidence[i] * 100, '.2f')) + '%'

        cv2.rectangle(img, (bbox[i][0], bbox[i][1]), (bbox[i][2], bbox[i][3]), color, 2)
        cv2.putText(img, label, (bbox[i][0], bbox[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img



           
while True:
    cap = cv2.VideoCapture(0)
    user_input = "0"
    time.sleep(2)
    ret, frame = cap.read()
    if user_input == "0":
        if ret:
            cv2.imwrite('baocao1.jpg', frame)
        else:
            print("No Camera")
        image = cv2.imread(r"D:\AI Challenge\Yolov4 Tiny\dataset\NCKH\Bottle\bottle_531.jpg")
        image = cv2.resize(image, (416, 416))
        print('Processing...')
    
        bbox, label, conf = detect_objects(image)
        img1 = draw_bbox(image, bbox, label, conf)
        print("Passed")
        cv2.imwrite("result.jpg", img1)
    elif user_input == "1":
        gmail.main("Thùng Rác Đã Đầy!")
    cap.release()
