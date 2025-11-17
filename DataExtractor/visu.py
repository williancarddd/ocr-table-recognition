import cv2
import numpy as np


img_path = "..\\yolo\\dataset_YOLO_train\\images\\train\\cTDaR_t00001.jpg"
txt_path = "..\\yolo\\dataset_YOLO_train\\labels\\train\\cTDaR_t00001.txt"

img = cv2.imread(img_path)
if img is None:
    print(f"tenta outra imagem pois essta provavelmente caiu no test.. {img_path}")
    exit()

h, w = img.shape[:2]
print(f"tamanho da imagem {w}x{h}")


overlay = img.copy()

try:
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    colors = []
    for i in range(len(lines)):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        colors.append(color)
    for idx, line in enumerate(lines):
        parts = line.strip().split()
        class_id = int(parts[0])
        coords = list(map(float, parts[1:]))
        points = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * w)
            y = int(coords[i+1] * h)
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(overlay, [points], colors[idx])
        cv2.polylines(img, [points], isClosed=True, color=colors[idx], thickness=2)
        M = cv2.moments(points)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(img, str(idx+1), (cx, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    alpha = 0.4
    result = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    max_height = 900
    if h > max_height:
        scale = max_height / h
        new_w = int(w * scale)
        new_h = int(h * scale)
        result = cv2.resize(result, (new_w, new_h))
    cv2.imshow('YOLO Segmentation - Verificacao', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
except FileNotFoundError:
    print(f"erro{txt_path}")
except Exception as e:
    print(f"erro {e}")