import cv2
import numpy as np
from pathlib import Path
from roop.FaceSet import FaceSet
from roop.face_util import extract_face_images
from roop.globals import BASE_URL

def process_and_save_template_faces(source_path, generation_id, template_id, output_dir, globals):
    
    print("Analyzing source image...")
    print(source_path)
    
    source_faces_data = extract_face_images(globals, source_path, (False, 0))
    if not source_faces_data:
        print("Error: No face detected in the source image.")
        return None, []
    
    face_set = FaceSet()
    for face_data in source_faces_data:
        face = face_data[0]
        face.mask_offsets = (0,0,0,0,1,20) # Default mask offsets
        face_set.faces.append(face)
        
    if len(face_set.faces) > 1:
         face_set.AverageEmbeddings()
    
    if len(face_set.faces) <= 1:
        globals.face_swap_mode = 'all_input'
    else :
        globals.face_swap_mode = 'selected'
        
    globals.TARGET_FACES.append(face_set)

    # Create output directories
    results_dir = Path(output_dir) / str(generation_id)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the original image
    original_image = cv2.imread(source_path)
    output_image = original_image.copy()

    try:
        overlay = np.zeros_like(original_image, dtype=np.uint8)

        for face, _ in source_faces_data:
            (startX, startY, endX, endY) = face.bbox.astype("int")
            center_x = (startX + endX) // 2
            center_y = (startY + endY) // 2
            axis_x = (endX - startX) // 2
            axis_y = (endY - startY) // 2
            cv2.ellipse(overlay, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, (255, 255, 255), -1)

        alpha = 0.4
        cv2.addWeighted(overlay, alpha, output_image, 1 - alpha, 0, output_image)

        masked_filename = f"template_masked_{generation_id}_{template_id}.jpg"
        masked_path = results_dir / masked_filename
        cv2.imwrite(str(masked_path), output_image)
        masked_face_url = f"{BASE_URL}/{output_dir}/{generation_id}/{masked_filename}"

    except Exception as e:
        print(f"Error creating masked face: {e}")
        masked_face_url = None

    # Save cropped faces
    detected_face_urls = []
    for i, (face, face_image) in enumerate(source_faces_data):
        try:
            face_filename = f"template_face_{i}_{generation_id}_{template_id}.jpg"
            face_path = results_dir / face_filename
            cv2.imwrite(str(face_path), face_image)
            detected_face_urls.append(f"{BASE_URL}/{output_dir}/{generation_id}/{face_filename}")
        except Exception as e:
            print(f"Error saving face {i}: {e}")
            continue
    
    return masked_face_url, detected_face_urls