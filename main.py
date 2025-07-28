from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any
import uvicorn
import json
import os
from fastapi import Form
from fastapi.staticfiles import StaticFiles
import requests
import uuid
from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any
from roop import FaceSet, globals as roop_globals
from pydantic import BaseModel

class SwapFaceRequest(BaseModel):
    generation_id: str
    source_indices: List[int]
    target_indices: List[int]
from roop.core import batch_process_regular
from roop.ProcessEntry import ProcessEntry
from scripts.upload_template_func import process_and_save_faces
from scripts.upload_target_func import process_and_save_target_faces
from fastapi import File, UploadFile

app = FastAPI(
    title="Face Swap API",
    description="An API for performing face swaps and managing templates.",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_TEMPLATES_DIR = "static/uploads"
OUTPUT_DIR = "static/Face-swap/results"

GENERATION_DATA = {} # Stores generation_id to template/target mapping

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
    
    roop_globals.target_path = file_path
    print(file_path)
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
        
    
    
    GENERATION_DATA[generation_id] = {
        "template_id": template_id,
        "template_path": file_path,
        "target_paths": [],
        "template_face_urls": detected_face_urls # Store detected face URLs for later use
    }
    
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
async def upload_targets(file: UploadFile = File(...), user_id: str = Form(...), generation_id: str = Form(...)):
    
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
        file=file,
        user_id=user_id,
        generation_id=generation_id,
        output_dir=OUTPUT_DIR
    )

    # Store target image paths in GENERATION_DATA
    if generation_id in GENERATION_DATA:
        GENERATION_DATA[generation_id]["target_paths"].extend([os.path.join(generation_dir, os.path.basename(url.lstrip("/"))) for url in target_urls])

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


@app.post("/swap_face")
async def swap_face(request: SwapFaceRequest):
    generation_id = request.generation_id
    roop_globals.source_indices = request.source_indices
    roop_globals.target_indices = request.target_indices
    generation_dir = os.path.join(OUTPUT_DIR, generation_id)
    
    generation_data = GENERATION_DATA.get(generation_id)
    
    if not generation_data:
        raise HTTPException(
            status_code=400,
            detail={
                "status": 400,
                "error": "Bad Request",
                "message": "Invalid generation_id. Please upload template and target images first."
            }
        )

    # Set roop_globals for face swapping
    roop_globals.target_path = generation_data["template_path"]
    roop_globals.reference_face_position = roop_globals.source_indices[0] if roop_globals.source_indices else 0 # Use the first source index, or 0 if not provided

    if not generation_data["target_paths"]:
        raise HTTPException(status_code=500, detail="No target images found for the given generation_id.")
    
    # Select the target image based on the first target_index provided
    target_image_path = generation_data["target_paths"][0]
    roop_globals.source_path = target_image_path
    
    output_filename = f"swapped_{generation_id}.jpg"
    roop_globals.output_path = os.path.join(generation_dir, output_filename)

    # Setting Indexes 
    
    temp_faceset = []
    for i in range(len(roop_globals.TARGET_FACES)):
        if i not in roop_globals.source_indices:
            temp_faceset.append(roop_globals.TARGET_FACES[i])
        else :
            temp_faceset.append(roop_globals.INPUT_FACESETS[roop_globals.target_indices[roop_globals.source_indices.index(i)]])

    roop_globals.INPUT_FACESETS = temp_faceset

    # Perform face swap
    
    list_files_process = [ProcessEntry(roop_globals.target_path, 0, 1, 0)]
    print(f"Target set to: {roop_globals.target_path}")
    
    # Set some more required globals
    roop_globals.execution_threads = roop_globals.CFG.max_threads
    roop_globals.max_memory = roop_globals.CFG.memory_limit if roop_globals.CFG.memory_limit > 0 else None

    
    try:
        batch_process_regular(
            swap_model=roop_globals.face_swapper_model,
            output_method="File",
            files=list_files_process,
            masking_engine=None,
            new_clip_text=None,
            use_new_method=True,
            imagemask=None,
            restore_original_mouth=False,
            num_swap_steps=1,
            progress=None,
            selected_index=0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face swap failed: {str(e)}")

    swapped_image_url = f"/{OUTPUT_DIR}/{generation_id}/{output_filename}"
    signed_swapped_image_url = f"{swapped_image_url}?dummy_signed_url"

    return {
        "message": "Face swap completed successfully",
        "generation_id": generation_id,
        "swapped_image_url": swapped_image_url,
        "signed_swapped_image_url": signed_swapped_image_url,
        "status": "completed"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)