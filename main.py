
import os
import shutil
import subprocess
import uuid
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from typing import List, Dict, Any
import json
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI(
    title="Face Swap API",
    description="An API for performing face swaps and managing templates.",
    version="1.0.0"
)

BASE_URL = "http://localhost:8000/FaceSwap/results"

app.mount("/static", StaticFiles(directory="static"), name="static")

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


@app.post("/upload_targets")
async def upload_targets(
    files: List[UploadFile] = File(...),
    user_id: str = Form(...),
    generation_id: str = Form(...)
):
    # Create directory path, e.g. user_id/generation_id/
    upload_dir = Path(user_id) / generation_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_filenames = []
    for idx, image_file in enumerate(files):
        # Customize a filename pattern like in example (simulate UUID + target info)
        unique_part = str(uuid.uuid4())
        filename = f"{unique_part}_target_{idx}_{generation_id}.jpg"
        file_path = upload_dir / filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image_file.file, buffer)

        saved_filenames.append(filename)

    # Build URLs based on saved files and generation_id
    # Assume files are under: /Face-swap/results/{generation_id}/{filename}
    target_urls = [f"{BASE_URL}/{generation_id}/{fn}" for fn in saved_filenames]

    # Simulate signed URLs by appending dummy query params
    def sign_url(url):
        return url + "?AWSAccessKeyId=immersouser&Signature=dummySig&Expires=9999999999"

    signed_target_urls = [sign_url(u) for u in target_urls]

    # Simulate face URLs for each target as filename_face_0_ + generation_id.jpg pattern
    target_face_urls = [
        u.replace(".jpg", f"_face_0_{generation_id}.jpg") for u in target_urls
    ]
    signed_target_face_urls = [sign_url(u) for u in target_face_urls]

    target_face_indices = [0]  # example static, assuming one face per image
    target_face_count = len(saved_filenames)
    status = "processing"

    response = {
        "message": "Target images uploaded successfully",
        "generation_id": generation_id,
        "target_urls": target_urls,
        "signed_target_urls": signed_target_urls,
        "target_face_urls": target_face_urls,
        "signed_target_face_urls": signed_target_face_urls,
        "target_face_indices": target_face_indices,
        "target_face_count": target_face_count,
        "status": status,
    }

    return response

@app.post("/swap_face")
async def multiface_swap():
    pass
    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
