
from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any

import json
import os
from fastapi import Form
import requests

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

import uuid

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
        file_path = os.path.join(output_dir, f"{user_id}_template.jpg")
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Template image for {template_id} downloaded successfully.")
    elif response.status_code == 404:
        raise HTTPException(status_code=404, detail=f"Template image for '{template_id}' not found at the provided URL.")
    else:
        raise HTTPException(status_code=500, detail="Failed to download the template image.")
    
    generation_id = str(uuid.uuid4())
    
    return {
        "message": "Template uploaded successfully",
        "generation_id": generation_id,
        "template_id": template_id,
        "template_url": f"/static/Face-swap/Templates/{template_id}.jpg",
        "masked_face_url": f"/static/Face-swap/results/{generation_id}/{template_id}_{generation_id}_masked.jpg",
        "signed_masked_face_url": f"/static/Face-swap/results/{generation_id}/{template_id}_{generation_id}_masked.jpg?dummy_signed_url",
        "detected_face_urls": [
            f"/static/results/{generation_id}/face_0_{generation_id}.jpg"
        ],
        "signed_detected_face_urls": [
            f"/static/results/{generation_id}/face_0_{generation_id}.jpg?dummy_signed_url"
        ],
        "template_face_indices": [0],
        "template_face_count": 1,
        "is_multi_face": False,
        "credits": 20,
        "status": "processing",
        "transcoding": None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
