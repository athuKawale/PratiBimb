import os
import cv2
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any
from roop import globals as roop_globals
from roop.core import batch_process_regular
from roop.face_util import extract_face_images
from roop.ProcessEntry import ProcessEntry
from roop.FaceSet import FaceSet
from roop import utilities as util
from prepare_env import prepare_environment

def process_and_save_faces(source_path, generation_id, template_id, output_dir="static/Face-swap/results"):
    print("Analyzing source image...")
    print(source_path)
    
    source_faces_data = extract_face_images(roop_globals.source_path, (False, 0))
    if not source_faces_data:
        print("Error: No face detected in the source image.")
        return None, []
    
    print(source_faces_data)
    
    # Create output directories
    results_dir = Path(output_dir) / str(generation_id)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    face_set = FaceSet()
    face, face_image = source_faces_data[0]
    face.mask_offsets = (0,0,0,0,1,20)  # Default mask offsets
    face_set.faces.append(face)
    
    print(f"face_set: {face_set}")
    print(f"face_set.faces: {face_set.faces}")
    
    roop_globals.INPUT_FACESETS.append(face_set)
    print(f"roop_globals.INPUT_FACESETS: {roop_globals.INPUT_FACESETS}")
    print(f"Found {len(source_faces_data)} face(s), using the first one.")
    
    print("face_image type:", type(face_image))
    print("face_image shape:", getattr(face_image, 'shape', None))
    print("Mask is None?", face)

    
    # Generate masked face
    try:
        h, w = face_image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w//2, h//2)
        axes = (w//3, h//2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        if len(face_image.shape) == 3:
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            masked_face = cv2.bitwise_and(face_image, mask_3d)
        else:
            masked_face = cv2.bitwise_and(face_image, face_image, mask=mask)

        masked_filename = f"{template_id}_{generation_id}_masked.jpg"
        masked_path = results_dir / masked_filename
        cv2.imwrite(str(masked_path), masked_face)
        masked_face_path = f"/static/Face-swap/results/{generation_id}/{masked_filename}"
    except Exception as e:
        print(f"Error creating masked face: {e}")
        masked_face_path = None

    # Save cropped faces
    detected_face_urls = []
    for i, (face, face_image) in enumerate(source_faces_data):
        try:
            face_filename = f"face_{i}_{generation_id}.jpg"
            face_path = results_dir / face_filename
            cv2.imwrite(str(face_path), face_image)
            detected_face_urls.append(f"/static/results/{generation_id}/{face_filename}")
        except Exception as e:
            print(f"Error saving face {i}: {e}")
            continue
    
    return masked_face_path, detected_face_urls