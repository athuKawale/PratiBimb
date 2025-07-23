import os
import cv2
import numpy as np
from pathlib import Path
from fastapi import UploadFile
from typing import List
from roop.face_util import extract_face_images

def process_and_save_target_faces(files: List[UploadFile], user_id: str, generation_id: str, output_dir: str):
    target_urls = []
    signed_target_urls = []
    target_face_urls = []
    signed_target_face_urls = []
    target_face_indices = []
    
    results_dir = Path(output_dir) / str(generation_id)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for i, file in enumerate(files):
        target_filename = f"{user_id}_target_{i}_{generation_id}.jpg"
        target_path = results_dir / target_filename
        
        with open(target_path, "wb") as buffer:
            buffer.write(file.file.read())
            
        target_urls.append(f"/{output_dir}/{generation_id}/{target_filename}")
        signed_target_urls.append(f"/{output_dir}/{generation_id}/{target_filename}?dummy_signed_url")
        
        target_faces_data = extract_face_images(str(target_path), (False, 0))
        
        if not target_faces_data:
            continue
            
        for j, (face, face_image) in enumerate(target_faces_data):
            face_filename = f"{user_id}_target_{i}_face_{j}_{generation_id}.jpg"
            face_path = results_dir / face_filename
            cv2.imwrite(str(face_path), face_image)
            
            target_face_urls.append(f"/{output_dir}/{generation_id}/{face_filename}")
            signed_target_face_urls.append(f"/{output_dir}/{generation_id}/{face_filename}?dummy_signed_url")
            target_face_indices.append(j)
            
    return target_urls, signed_target_urls, target_face_urls, signed_target_face_urls, target_face_indices
