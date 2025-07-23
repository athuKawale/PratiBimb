
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
from scripts.upload_template_func import process_and_save_faces
from scripts.upload_target_func import process_and_save_target_faces
from fastapi import File, UploadFile

app = FastAPI(
    title="Face Swap API",
    description="An API for performing face swaps and managing templates.",
    version="1.0.0"
)

UPLOAD_TEMPLATES_DIR = "static/uploads"
OUTPUT_DIR = "static/Face-swap/results"

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
    
    # Template download logic (unchanged)
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
    
    prepare_environment()
    roop_globals.source_path = file_path
    
    # Process faces and get URLs
    masked_face_url, detected_face_urls = process_and_save_faces(
        source_path=file_path,  # Use file_path directly
        generation_id=generation_id,
        template_id=template_id,
        output_dir=OUTPUT_DIR
    )
    
    # Handle cases where face processing fails
    if masked_face_url is None:
        masked_face_url = ""
    if not detected_face_urls:
        detected_face_urls = []
        
    
    
    return {
        "message": "Template uploaded successfully",
        "generation_id": generation_id,
        "template_id": template_id,
        "template_url": f"static/Face-swap/Templates/{template_id}/{template_id}.jpg",
        "masked_face_url": masked_face_url,
        "signed_masked_face_url": f"{masked_face_url}?dummy_signed_url" if masked_face_url else "",
        "detected_face_urls": detected_face_urls,
        "signed_detected_face_urls": [f"{url}?dummy_signed_url" for url in detected_face_urls],
        "template_face_indices": list(range(len(detected_face_urls))),
        "template_face_count": len(detected_face_urls),
        "is_multi_face": len(detected_face_urls) > 1,
        "credits": 20,
        "status": "processing",
        "transcoding": None
    }

@app.post("/upload_targets")
async def upload_targets(files: List[UploadFile] = File(...), user_id: str = Form(...), generation_id: str = Form(...)):
    
    # Check if the generation_id is valid by checking if the directory exists
    generation_dir = os.path.join(OUTPUT_DIR, generation_id)
    if not os.path.exists(generation_dir):
        raise HTTPException(
            status_code=400,
            detail={
                "status": 400,
                "error": "Bad Request",
                "message": "Invalid generation_id. Please upload template first and use the generation_id returned from the upload_template endpoint."
            }
        )

    target_urls, signed_target_urls, target_face_urls, signed_target_face_urls, target_face_indices = process_and_save_target_faces(
        files=files,
        user_id=user_id,
        generation_id=generation_id,
        output_dir=OUTPUT_DIR
    )

    return {
        "message": "Target images uploaded successfully",
        "generation_id": generation_id,
        "target_urls": target_urls,
        "signed_target_urls": signed_target_urls,
        "target_face_urls": target_face_urls,
        "signed_target_face_urls": signed_target_face_urls,
        "target_face_indices": target_face_indices,
        "target_face_count": len(target_face_urls),
        "status": "processing"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
