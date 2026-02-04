import cv2
import insightface
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

img = cv2.imread("your_photo.jpg")

# Perform inference
faces = app.get(img)

print(f"Detected {len(faces)} faces")

# Draw boxes on the image
res = img.copy()
for face in faces:
    # bounding box is in face.bbox (x1, y1, x2, y2)
    box = face.bbox.astype(int)
    cv2.rectangle(res, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

cv2.imwrite("output_detected.jpg", res)