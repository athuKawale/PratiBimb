import cv2
import insightface
from pathlib import Path

def extract_faces(image_path, output_dir=None):
    def bbox_area(face):
        x1, y1, x2, y2 = face.bbox
        return (x2 - x1) * (y2 - y1)
    # Load image
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError("Could not load image!")
    
    app = insightface.app.FaceAnalysis(name="buffalo_l")

    app.prepare(ctx_id=0, det_size=(640, 640))

    faces = app.get(img)

    faces_sorted = sorted(faces, key=bbox_area, reverse=True)

    if output_dir is None:
        output_dir = Path("extracted_faces")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    height, width = img.shape[:2]
    
    face_paths = []
    for i, face in enumerate(faces_sorted):
        x1, y1, x2, y2 = [int(coord) for coord in face.bbox]
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height, y2))
        if x2 > x1 and y2 > y1:
            face_img = img[y1:y2, x1:x2]
            save_path = output_dir / f"face_{i}.jpg"
            cv2.imwrite(str(save_path), face_img)
            face_paths.append(str(save_path))
    return face_paths