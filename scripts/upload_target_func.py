import os
from pathlib import Path
import cv2
from fastapi import UploadFile
from typing import List
from urllib.parse import urlparse
import requests
from roop.FaceSet import FaceSet
from roop.globals import BASE_URL
from roop.face_util import extract_face_images
from roop import globals as roop_globals

def process_and_save_target_faces(files: List[UploadFile], file_url : str, user_id: str, generation_id: str, output_dir: str):
    target_urls = []
    signed_target_urls = []
    target_face_urls = []
    signed_target_face_urls = []
    target_face_indices = []
    target_paths = []

    results_dir = f"{output_dir}/{generation_id}/targets"
    os.makedirs(results_dir, exist_ok=True)

    if files[0].size != 0: # if size of first file is zero it means we don't have any files

        for file in files :

            filename = str(file.filename)

            target_filename = f"target_{generation_id}_{user_id}_{filename}"

            target_path = f"{results_dir}/{target_filename}"

            target_paths.append(target_path)

            with open(target_path, "wb") as buffer:
                buffer.write(file.file.read())
        
    if file_url :

        response = requests.get(file_url)

        filename = str(Path(urlparse(file_url).path).name)

        target_filename = f"target_{generation_id}_{user_id}_{filename}"

        target_path = f"{results_dir}/{target_filename}"
        
        target_paths.append(target_path)

        if response.status_code == 200:
            with open(target_path, 'wb') as f:
                f.write(response.content)
            print("Image downloaded successfully!")
        else:
            print("Failed to download image")

    
    j = 0 # face count

    for target_path in target_paths:

        target_urls.append(f"{BASE_URL}/{target_path}")
        signed_target_urls.append(f"{BASE_URL}/{target_path}?dummy_signed_url")

        target_faces_data = extract_face_images(target_path, (False, 0))
        
        if not target_faces_data:
            print(f"No faces detected in the {target_path} image.")
            continue
        
        for face_data in target_faces_data:
            face_set = FaceSet()
            face = face_data[0]
            face.mask_offsets = (0,0,0,0,1,20) # Default mask offsets
            face_set.faces.append(face)
            roop_globals.INPUT_FACESETS.append(face_set)
        

        print(f"Found {len(target_faces_data)} face(s), in {target_path} applying mask.")

        for face, face_image in target_faces_data:
            face_filename = f"target_face_{j}_{generation_id}_{user_id}.jpg"
            face_path = f"{results_dir}/{face_filename}"
            cv2.imwrite(face_path, face_image)
            
            target_face_urls.append(f"{BASE_URL}/{face_path}")
            signed_target_face_urls.append(f"{BASE_URL}/{face_path}?dummy_signed_url")
            target_face_indices.append(j)

            j+=1
            
    return target_urls, signed_target_urls, target_face_urls, signed_target_face_urls, target_face_indices, target_paths
