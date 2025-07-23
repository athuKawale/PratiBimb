
from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any

import json
import os
from fastapi import Form
import requests
import cv2
import numpy as np
from roop.face_util import get_all_faces, extract_face_images
from roop.processors.Frame_Masking import Frame_Masking
import uuid
from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any
from roop import globals as roop_globals
from roop.core import batch_process_regular
from roop.face_util import extract_face_images
from roop.ProcessEntry import ProcessEntry
from roop.FaceSet import FaceSet
from roop import utilities as util
from prepare_env import prepare_environment

app = FastAPI(
    title="Face Swap API",
    description="An API for performing face swaps and managing templates.",
    version="1.0.0"
)

UPLOAD_TEMPLATES_DIR = "static/uploads"

# Load templates from the JSON file
with open("static/templates.json", "r") as f:
    TEMPLATES_DATA = json.load(f)

@app.get("/templates", response_model=Dict[str, Any])
async def get_available_templates():
    """
    Returns a list of available templates with their IDs, filenames, and pre-signed URLs.
    """
    if not TEMPLATES_DATA or not TEMPLATES_DATA["available_templates"]:
        raise HTTPException(status_code=404, detail="No templates found.")
    
    templates = TEMPLATES_DATA.get("available_templates", [])
    
    # Create a new list containing dictionaries with only the desired keys
    filtered_templates = []
    for template in templates:
        filtered_templates.append({
            "template_id": template.get("template_id"),
            "template_url": template.get("template_url"),
            "template_filename": template.get("template_filename")
        })
    
    return {"available_templates": filtered_templates}

@app.get("/templates/{template_id}/info", response_model=Dict[str, Any])
async def get_template_info(template_id: str):
    """
    Returns detailed information for a specific template.
    """
    for template in TEMPLATES_DATA.get("available_templates", []):
        if template.get("template_id") == template_id:
            return template
    raise HTTPException(status_code=404, detail=f"Template with ID '{template_id}' not found.")

@app.post("/upload_template")
async def upload_template(template_id: str = Form(...), user_id: str = Form(...)):
    
    for template in TEMPLATES_DATA.get("available_templates", []):
        if template.get("template_id") == template_id:
            img_url = template.get("template_url")
            break
    else:
        raise HTTPException(status_code=404, detail=f"Template with ID '{template_id}' not found.")

    response = requests.get(img_url)
    if response.status_code == 200:
        output_dir = f"static/Face-swap/Templates/{template_id}"
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{template_id}.jpg")
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Template image for {template_id} downloaded successfully.")
    elif response.status_code == 404:
        raise HTTPException(status_code=404, detail=f"Template image for '{template_id}' not found at the provided URL.")
    else:
        raise HTTPException(status_code=500, detail="Failed to download the template image.")
    
    generation_id = str(uuid.uuid4())
    

    roop_globals.source_path = file_path
    
    print("Analyzing source image...")
    print(file_path)
    source_faces_data = extract_face_images(roop_globals.source_path, (False, 0))
    if not source_faces_data:
        print("Error: No face detected in the source image.")
    
    print(source_faces_data)

    # # Create and initialize the frame masker
    # frame_masker = Frame_Masking()
    # frame_masker.Initialize(plugin_options={'devicename': 'cpu'})

    # # Create the masked face
    # masked_frame = frame_masker.Run(img)
    
    # # Save the masked face
    # masked_face_dir = f"static/Face-swap/results/{generation_id}"
    # os.makedirs(masked_face_dir, exist_ok=True)
    # masked_face_path = os.path.join(masked_face_dir, f"{template_id}_{generation_id}_masked.jpg")
    # cv2.imwrite(masked_face_path, masked_frame)
    
    # Detect and save cropped faces
    detected_faces_dir = f"static/results/{generation_id}"
    os.makedirs(detected_faces_dir, exist_ok=True)
    
    masked_face_path = '/'
    detected_face_urls = []
    
    # faces = get_all_faces(img)
    # detected_face_urls = []
    # for i, face in enumerate(faces):
    #     face_img = extract_face_images(file_path, (False, 0))
    #     if face_img:
    #         for j, (face, face_img_data) in enumerate(face_img):
    #             cropped_face_path = os.path.join(detected_faces_dir, f"face_{i}_{j}_{generation_id}.jpg")
    #             cv2.imwrite(cropped_face_path, face_img_data)
    #             detected_face_urls.append(f"/{cropped_face_path}")

    return {
        "message": "Template uploaded successfully",
        "generation_id": generation_id,
        "template_id": template_id,
        "template_url": f"/static/Face-swap/Templates/{template_id}.jpg",
        "masked_face_url": f"/{masked_face_path}",
        "signed_masked_face_url": f"/{masked_face_path}?dummy_signed_url",
        "detected_face_urls": detected_face_urls,
        "signed_detected_face_urls": [f"{url}?dummy_signed_url" for url in detected_face_urls],
        "template_face_indices": list(range(len(detected_face_urls))),
        "template_face_count": len(detected_face_urls),
        "is_multi_face": len(detected_face_urls) > 1,
        "credits": 20,
        "status": "processing",
        "transcoding": None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
