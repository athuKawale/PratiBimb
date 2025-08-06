import os
import uuid
import json
import requests
from datetime import datetime
from pydantic import BaseModel
from roop.globals import BASE_URL
from roop import globals as roop_globals
from roop.ProcessEntry import ProcessEntry
from roop.core import batch_process_regular
from typing import List, Dict, Any, Optional
from scripts.upload_template_func import process_and_save_faces
from scripts.upload_target_func import process_and_save_target_faces
from fastapi import APIRouter, HTTPException, Form, File, UploadFile

class SwapFaceRequest(BaseModel):
    generation_id: str
    source_indices: List[int]
    target_indices: List[int]

router = APIRouter(
    prefix="",
    tags=["faceswap"]
)

TEMPLATES_DIR = "static/Face-swap/Templates"
OUTPUT_DIR = "static/Face-swap/results"

GENERATION_DATA = {} 

with open("static/templates.json", "r") as f:
    TEMPLATES_DATA = json.load(f)


@router.get("/health")
async def health() :

    return {
        "status": "healthy",
        "service": "image-face-swap",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": "1.0",
        "active_workers": 1,
        "cleanup_worker_active": True
    }

@router.get("/templates", response_model=Dict[str, Any])
async def get_available_templates():
    """
    Returns a list of available templates with their IDs, filenames, and pre-signed URLs.
    """
    if not TEMPLATES_DATA or not TEMPLATES_DATA["available_templates"]:
        raise HTTPException(status_code=404, detail="No templates found.")
    
    templates = TEMPLATES_DATA.get("available_templates", [])
    
    filtered_templates = []
    for template in templates:
        filtered_templates.append({
            "template_id": template.get("template_id"),
            "filename": template.get("filename"),
            "signed_template_url": template.get("signed_template_url"),
            "file_size": template.get("file_size"),
            "last_modified": "2025-07-31T05:16:19.874000+00:00"

        })
    
    return {"available_templates": filtered_templates}

@router.get("/templates/{template_id}/info", response_model=Dict[str, Any])
async def get_template_info(template_id: str):
    """
    Returns detailed information for a specific template.
    """
    for template in TEMPLATES_DATA.get("available_templates", []):
        if template.get("template_id") == template_id:
            return template
    raise HTTPException(status_code=404, detail=f"Template with ID '{template_id}' not found.")

@router.post("/upload_template")
async def upload_template(
    template_id: Optional[str] = Form(None),
    user_id: str = Form(...),           
    files: Optional[UploadFile] = File(None)
):    

    template_ids = []

    if template_id is not None and files is not None :
        raise HTTPException(
            status_code=400,
            detail="Cannot provide both template_id and files. Choose one option."
        )

    if template_id is None and files is None :
        raise HTTPException(
            status_code=400,
            detail="Specify either template_id or files"
        )
    
    if files is None :

        # Template download
        template_ids.append(template_id)

        for template in TEMPLATES_DATA.get("available_templates", []):
            if template.get("template_id") == template_id:
                img_url = template.get("template_url")
                break
        else:
            raise HTTPException(status_code=404, detail=f"Template with ID '{template_id}' not found.")
        
        response = requests.get(img_url)

        if response.status_code == 200:

            output_dir = f"{TEMPLATES_DIR}/{template_id}"

            os.makedirs(output_dir, exist_ok=True)

            file_path = os.path.join(output_dir, f"{template_id}.jpg")

            with open(file_path, 'wb') as f:
                f.write(response.content)

            print(f"Template image for {template_id} downloaded successfully.")

        elif response.status_code == 404:

            raise HTTPException(status_code=404, detail=f"Template image for '{template_id}' not found at the provided URL.")
        
        else:

            raise HTTPException(status_code=500, detail="Failed to download the template image.")
       
    else :

        os.makedirs(TEMPLATES_DIR, exist_ok=True) 

        file_path = os.path.join(TEMPLATES_DIR, files.filename)

        template_ids.append(files.filename)

        with open(file_path, "wb") as file_object:
            file_object.write(await files.read())

    roop_globals.target_path = file_path

    generation_id = str(uuid.uuid4())
    

    # Process faces and get URLs
    masked_face_url, detected_face_urls = process_and_save_faces(
        source_path=file_path,  # Use file_path directly
        generation_id=generation_id,
        template_id=template_id,
        output_dir=OUTPUT_DIR
    )
    
    if len(detected_face_urls) == 0 :
        raise HTTPException(
            status_code=404,
            detail="No face detected."
        )
    
    GENERATION_DATA[generation_id] = {
        "template_ids": template_ids,
        "template_path": file_path,
        "target_paths": [],
        "template_url": f"{BASE_URL}/{file_path}",
        "masked_face_url": masked_face_url,
        "signed_masked_face_url": f"{masked_face_url}?dummy_signed_url",
        "detected_face_urls": detected_face_urls,
        "signed_detected_face_urls": [f"{url}?dummy_signed_url" for url in detected_face_urls],
        "template_face_indices": list(range(len(detected_face_urls))),
        "template_face_count": len(detected_face_urls),
        "is_multi_face": len(detected_face_urls) > 1,
        "credits": 20,
        "status": "processing",
        
    }

    return {
        "message": "Template uploaded successfully",
        "generation_id": generation_id,
        "template_ids": GENERATION_DATA[generation_id]['template_ids'],
        "template_url": GENERATION_DATA[generation_id]["template_url"],
        "masked_face_url": GENERATION_DATA[generation_id]["masked_face_url"],
        "signed_masked_face_url": GENERATION_DATA[generation_id]["signed_masked_face_url"],
        "detected_face_urls": GENERATION_DATA[generation_id]["detected_face_urls"],
        "signed_detected_face_urls": GENERATION_DATA[generation_id]["signed_detected_face_urls"],
        "template_face_indices": GENERATION_DATA[generation_id]["template_face_indices"],
        "template_face_count": GENERATION_DATA[generation_id]["template_face_count"],
        "is_multi_face": GENERATION_DATA[generation_id]["is_multi_face"],
        "credits": 20,
        "status": GENERATION_DATA[generation_id]["status"],
    }

@router.post("/upload_targets")
async def upload_targets(files: List[UploadFile] = File(...), user_id: str = Form(...), generation_id: str = Form(...), file_url : str = Form(...)):
    
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

    target_urls, signed_target_urls, target_face_urls, signed_target_face_urls, target_face_indices, target_paths = process_and_save_target_faces(
        files=files,
        file_url=file_url,
        user_id=user_id,
        generation_id=generation_id,
        output_dir=OUTPUT_DIR
    )

    # Store target image paths in GENERATION_DATA
    if generation_id in GENERATION_DATA:
        GENERATION_DATA[generation_id]["target_paths"] = target_paths
        GENERATION_DATA[generation_id]["target_urls"] = target_urls
        GENERATION_DATA[generation_id]["signed_target_urls"] = signed_target_urls
        GENERATION_DATA[generation_id]["target_face_urls"] = target_face_urls
        GENERATION_DATA[generation_id]["signed_target_face_urls"] = signed_target_face_urls
        GENERATION_DATA[generation_id]["target_face_indices"] = target_face_indices
        GENERATION_DATA[generation_id]["target_face_count"] = len(target_face_urls)

    return {
        "message": "Target images uploaded successfully",
        "generation_id": generation_id,
        "target_urls": target_urls,
        "signed_target_urls": signed_target_urls,
        "target_face_urls": target_face_urls,
        "signed_target_face_urls": signed_target_face_urls,
        "target_face_indices": target_face_indices,
        "target_face_count": len(target_face_urls),
        "status": GENERATION_DATA[generation_id]["status"],
        "additional_info" : None
    }

@router.post("/swap_face")
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
    
    roop_globals.INPUT_FACESETS = []  
    roop_globals.TARGET_FACES = []
    
    return {
        "message": "Face swap completed successfully",
        "generation_id": generation_id,
        "swapped_image_url": swapped_image_url,
        "signed_swapped_image_url": signed_swapped_image_url,
        "status": "completed"
    }
