
import os
import shutil
import subprocess
from fastapi import FastAPI, File, HTTPException, UploadFile
from typing import List, Dict, Any
import json
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="Face Swap API",
    description="An API for performing face swaps and managing templates.",
    version="1.0.0"
)

SOURCE_DIR = "static/multifaceswap"
TARGET_DIR = "static/multifaceswap"
OUTPUT_PATH = "output/output.jpg"
os.makedirs(SOURCE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

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


@app.get("/multifaceswap", response_class=HTMLResponse)
async def read_multifaceswap_page():
    file_path = os.path.join("static", "multifaceswap.html")
    with open(file_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/api/v1/multifaceswap")
async def multiface_swap(
    source_images: List[UploadFile] = File(...),
    target_image: UploadFile = File(...)
):
    # Save uploaded source images
    source_paths = []
    for idx, image in enumerate(source_images):
        source_path = os.path.join(SOURCE_DIR, f"s{idx+1}.jpg")
        with open(source_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
        source_paths.append(source_path)

    # Save uploaded target image
    target_path = os.path.join(TARGET_DIR, "target.jpg")
    with open(target_path, "wb") as f:
        shutil.copyfileobj(target_image.file, f)

    # Compose the CLI command
    cmd = [
        "python", "scripts/multi_face_swap.py",
        "-s", *source_paths,
        "-t", target_path,
        "-o", OUTPUT_PATH,
        "--swap-model", "InSwapper 128",
        "--enhancer", "GFPGAN",
        "--similarity-threshold", "0.5",
        "--blend-ratio", "0.7"
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        return JSONResponse(content={"error": e.stderr or "Face swap failed."}, status_code=500)

    if os.path.exists(OUTPUT_PATH):
        return FileResponse(OUTPUT_PATH, media_type="image/jpeg", filename="swapped_output.jpg")
    else:
        return JSONResponse(content={"error": "Output image not found"}, status_code=500)
    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
