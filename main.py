import os
import shutil
import uvicorn
import subprocess
import uuid
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from typing import List, Dict, Any
import json
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from scripts.get_faces import extract_faces

app = FastAPI(
    title="Face Swap API",
    description="An API for performing face swaps and managing templates.",
    version="1.0.0"
)

BASE_URL = "http://localhost:8000"

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount(
    "/FaceSwap/results",
    StaticFiles(directory=os.path.abspath("FaceSwap/results")),
    name="results"
)

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


@app.post("/upload_target")
async def upload_target(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    generation_id: str = Form(...)
):
    # Save uploaded image
    upload_dir = Path("FaceSwap/results") / user_id / generation_id

    upload_dir.mkdir(parents=True, exist_ok=True)

    filename = "target.jpg"

    file_path = upload_dir / filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    face_paths = extract_faces(str(file_path), output_dir=upload_dir)

    target_url = f"{BASE_URL}/{upload_dir}/{filename}"

    signed_target_url = f"{target_url}?AWSAccessKeyId=immersouser&Signature=dummySig&Expires=9999999999"

    face_urls = [
        f"{BASE_URL}/{upload_dir}/{Path(p).name}" for p in face_paths
    ]

    signed_face_urls = [
        f"{url}?AWSAccessKeyId=immersouser&Signature=dummySig&Expires=9999999999"
        for url in face_urls
    ]

    return {
        "message": "Target image uploaded successfully",
        "generation_id": generation_id,
        "target_url": target_url,
        "signed_target_url": signed_target_url,
        "target_face_urls": face_urls,
        "signed_target_face_urls": signed_face_urls,
        "target_face_count": len(face_paths),
        "target_face_paths": face_paths,  
        "status": "processing",
    }

@app.post("/swap_face")
async def swap_face():
    pass
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
