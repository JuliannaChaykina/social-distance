import cv2
from scaled_yolov4.detect import detect

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if frame is None: 
        break
    
    img = frame
    objs = detect(frame)
    for obj in objs:
        if obj["name"] == "person" and obj["confidence"] > 0.5:
    	    cv2.rectangle(img, (int(obj['xmin']), int(obj['ymin'])), (int(obj['xmax']), int(obj['ymax'])), (255, 0, 0), 3)
    
    for i in range(len(objs)):
        for j in range(len(objs)):
            if i != j:
                o0 = objs[i]
                o1 = objs[j]
                if o0["name"] == "person" and o0["confidence"] > 0.5 and o1["name"] == "person" and o1["confidence"] > 0.5:
               	    x0 = (o0["xmin"] + o0["xmax"]) / 2
                    x1 = (o1["xmin"] + o1["xmax"]) / 2
                    d = abs(x0 - x1)
                    print(d)
                    m = (o0["xmax"] - o0["xmin"]) * 1.5
                    if d < m:
                        cv2.rectangle(img, (int(o0['xmin']), int(o0['ymin'])), (int(o0['xmax']), int(o0['ymax'])), (0, 0, 255), 3)
                        cv2.rectangle(img, (int(o1['xmin']), int(o1['ymin'])), (int(o1['xmax']), int(o1['ymax'])), (0, 0, 255), 3)
    cv2.namedWindow("image")
    cv2.imshow("image", img)
    cv2.waitKey(1)
