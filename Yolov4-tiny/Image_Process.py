import serial
import cv2
import argparse
import numpy as np
import LiveStreamVid
import gmail

number = 0
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', default="/home/pi/Documents/Yolov4-tiny/yolov4-tiny-custom.cfg",
                help='path to yolo config file')
ap.add_argument('-w', '--weights', default="/home/pi/Documents/Yolov4-tiny/yolov4-tiny-custom_6000.weights",
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', default="/home/pi/Documents/Yolov4-tiny/obj.names",
                help='path to text file containing class names')
args = ap.parse_args()

ser = serial.Serial('/dev/ttyUSB0',9600)
count = 0

def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    predict = str(round(confidence*100)) + "%"
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(img, predict, (x + 40,y-10), cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
    print(label)
    m = 'm'
    b = '0'
    p = 'p'
    if (label == "bottle"):
        ser.write(m.encode())
    elif (label == "can"):
        ser.write(p.encode())
    else:
        ser.write(b.encode())

def capture_image():
    if ret:
        cv2.imwrite('INPUT.jpg',frame)
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("No Camera")

while True:
    cap = cv2.VideoCapture(0)
    ret,frame = cap.read()
    frame = cv2.flip(frame,0)

    data = ser.read()
    user_input = data.decode('utf-8')
    if user_input == "8":
        capture_image()
        image = cv2.imread('INPUT.jpg')
        print('Processing...')
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392

        classes = None

        with open(args.classes, 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

        net = cv2.dnn.readNet(args.weights, args.config)

        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)

        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.7
        nms_threshold = 0.4

        # Thực hiện xác định bằng HOG và SVM

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
            count += 1
        if (count!=0):
            print("Object: ",count)
        else:
            print("None")
        print('Done!')
        cv2.imwrite("object-detection.jpg", image)
    elif user_input == "f":
        # gửi email về cho người dùng
        gmail.main("Thùng Rác Đã Đầy. Hãy Thu Gom Rác!!!")
        count = 0
    elif user_input == "2":
        # Luồng hình ảnh trực tiếp
        LiveStreamVid.main()