import os
import re
import uuid
import json
import glob
import asyncio
import logging
import requests
from PIL import Image
from metadata import version
from datetime import datetime
from roop.globals import BASE_URL
from schema import SwapFaceRequest
from roop import globals as roop_globals
from roop.ProcessEntry import ProcessEntry
from fastapi.responses import JSONResponse
from roop.core import batch_process_regular
from typing import List, Dict, Any, Optional
from contextlib import redirect_stderr, redirect_stdout
from scripts.save_to_json import save_generation_data_to_json
from scripts.upload_template_func import process_and_save_faces
from scripts.upload_target_func import process_and_save_target_faces
from fastapi import APIRouter, HTTPException, Form, File, UploadFile

"""Globals"""


router = APIRouter(
    prefix="",
    tags=["faceswap"]
)

OUTPUT_DIR = "static/Face-swap/results"
TEMPLATES_DIR = "static/Face-swap/Templates"
DATA_FILE = "data.json"
GENERATION_DATA = {} 

with open("static/templates.json", "r") as f:
    TEMPLATES_DATA = json.load(f)

"""Setup logging for face swap operations"""

# Dictionary to store loggers per generation_id
loggers = {}

def get_logger_for_generation(generation_id: str) -> logging.Logger:
    if generation_id in loggers:
        return loggers[generation_id]

    generation_dir = os.path.join(OUTPUT_DIR, generation_id)
    os.makedirs(generation_dir, exist_ok=True)
    log_file_path = os.path.join(generation_dir, "faceswap.log")

    logger = logging.getLogger(f"faceswap_{generation_id}")
    logger.setLevel(logging.INFO)

    # Clear existing handlers for this logger to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file_path, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    loggers[generation_id] = logger
    return logger

def log_and_print(generation_id: str, msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(timestamp, msg)
    logger = get_logger_for_generation(generation_id)
    logger.info(msg)


def extract_last_percentage(log_file: str) -> float:
    if not os.path.exists(log_file):
        print("file not found:", log_file)
        return 0.0

    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    percent = 0.0
    pattern = re.compile(r'Processing:\s+(\d{1,3})%')

    for line in reversed(lines):
        match = pattern.search(line)
        if match:
            percent = float(match.group(1))
            break

    return percent


"""Run face swap in background using asyncio"""

def run_face_swap_background(request):
    asyncio.run(run_face_swap(request))

async def run_face_swap(request : SwapFaceRequest):

    generation_id = request.generation_id

    generation_dir = os.path.join(OUTPUT_DIR, generation_id)
        
    log_file_path = os.path.join(generation_dir, "faceswap.log")

    log_and_print(generation_id, "Starting face swap process...")

    source_indices = request.source_indices
    target_indices = request.target_indices
    
    generation_data = GENERATION_DATA.get(generation_id)

    if not generation_data:
        raise HTTPException(status_code=400,detail="Invalid generation_id. Please upload template and target images first.")
    
    # Set roop_globals for face swapping
    roop_globals.target_path = generation_data["template_path"]

    if not generation_data["target_paths"]:
        raise HTTPException(status_code=500, detail="No target images found for the given generation_id.")
    
    # Select the target image based on the first target_index provided
    target_image_path = generation_data["target_paths"][0]
    roop_globals.source_path = target_image_path

    roop_globals.output_path = os.path.join(generation_dir, "swapped")
    
    Template = generation_data["template_path"]

    for i in range(len(source_indices)) :

        roop_globals.INPUT_FACESETS = [roop_globals.TEMP_FACESET[target_indices[i]]]

        if len(roop_globals.TARGET_FACES[0].faces) > 1 :
            # Move Embedding of faces to faces[0]  and setting faces embedding to None so that averageEmbeddings can be calculated.
            roop_globals.TARGET_FACES[0].faces[0].embedding = roop_globals.TARGET_FACES[0].embedding
            roop_globals.TARGET_FACES[0].embedding = None

            # getting i-th target face at front of list so that it is swapped
            temp = roop_globals.TARGET_FACES[0].faces[0]
            roop_globals.TARGET_FACES[0].faces[0] = roop_globals.TARGET_FACES[0].faces[source_indices[i]]
            roop_globals.TARGET_FACES[0].faces[source_indices[i]] = temp

            roop_globals.TARGET_FACES[0].AverageEmbeddings()

        # Perform face swap
        
        list_files_process = [ProcessEntry(Template, 0, 1, 0)]
        print(f"Target set to: {Template}")

        #This clears the log file
        with open(log_file_path, 'w'):
            pass

        try :

            with open(log_file_path, 'a') as f:
                with redirect_stdout(f), redirect_stderr(f):
                    batch_process_regular(
                        swap_model=roop_globals.face_swapper_model,
                        output_method="File",
                        files=list_files_process,
                        masking_engine=roop_globals.mask_engine,
                        new_clip_text=roop_globals.clip_text,
                        use_new_method=True,
                        imagemask=None,
                        restore_original_mouth=False,
                        num_swap_steps=1,
                        progress=None,
                        selected_index=0
                    )

        except Exception as e:

            print(f"Error during face swap processing: {e}\nIteration: {generation_data['iteration']}\n")

            generation_data["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            generation_data["status"] = "error"

            generation_data["details"] = f"Faceswap Failed : \n{e}"

            #Save GENRATION DATA to json
            save_generation_data_to_json(generation_data["user_id"], generation_id, GENERATION_DATA)
            
            log_and_print(generation_id, f"Face swap failed: {e}")

            roop_globals.INPUT_FACESETS = []
            roop_globals.TARGET_FACES = []

            return 
    
    
        generation_data["iterations"] += 1
        
        if os.path.exists(os.path.join(roop_globals.output_path, 'output.png')):
            os.remove(os.path.join(roop_globals.output_path, 'output.png'))

        Template = glob.glob(os.path.join(roop_globals.output_path, '*_*.png'))[0]

        if not Template:
            print("Error: No output file created during processing.")
            return
        
        os.rename(Template, os.path.join(roop_globals.output_path, "output.png"))
        Template = os.path.join(roop_globals.output_path, "output.png")

    
    file_path = glob.glob(os.path.join(roop_globals.output_path, '*.png'))[0]
    file_url = f"{BASE_URL}/{file_path}"
    signed_swap_url = f"{file_url}?dummy_signed_url"
    
    roop_globals.INPUT_FACESETS = []
    roop_globals.TARGET_FACES = []

    generation_data["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    generation_data["status"] = "finished"
    generation_data["file_url"] = file_url
    generation_data["signed_swap_url"] = signed_swap_url

    #Save GENRATION DATA to json
    save_generation_data_to_json(generation_data["user_id"], generation_id, GENERATION_DATA)
   
    print(f"[{generation_data['finished_at']}] Face swap completed.")


"""API ENDPOINTS"""

@router.get("/health")
async def health() :

    return {
        "status": "healthy",
        "service": "image-face-swap",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": version,
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

            os.makedirs(TEMPLATES_DIR, exist_ok=True)

            file_path = os.path.join(TEMPLATES_DIR, f"{template_id}.jpg")

            with open(file_path, 'wb') as f:
                f.write(response.content)

            print(f"Template image for {template_id} downloaded successfully.")

        elif response.status_code == 404:

            raise HTTPException(status_code=404, detail=f"Template image for '{template_id}' not found at the provided URL.")
        
        else:

            raise HTTPException(status_code=500, detail="Failed to download the template image.")
       
    else :

        os.makedirs(TEMPLATES_DIR, exist_ok=True) 
        
        file_path = os.path.join(TEMPLATES_DIR, str(files.filename))

        template_ids.append(files.filename)

        with open(file_path, "wb") as file_object:
            file_object.write(await files.read())

    roop_globals.target_path = file_path

    generation_id = str(uuid.uuid4())

    # Process faces and get URLs
    masked_face_url, detected_face_urls = process_and_save_faces(
        source_path=file_path, 
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
        "user_id" : user_id,
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
        "iterations" : 0
        
    }

    # Find Image dimensions and orientations 
    with Image.open(file_path) as img:
        width, height = img.size
        if width > height:
            orientation = "landscape"
        elif height > width:
            orientation = "portrait"
        else:
            orientation = "square"
        
        GENERATION_DATA[generation_id]["result_image_dimensions"] = {
            "width": width,
            "height": height,
            "orientation": orientation
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

@router.get("/users/{user_id}/previous_target_faces")
async def get_previous_target_faces(user_id: str):
    if not os.path.exists(DATA_FILE):
        raise HTTPException(status_code=404, detail="Data store unavailable.")

    with open(DATA_FILE, "r") as f:
        try:
            all_data = json.load(f)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Could not parse data file.")

    user_data = all_data.get(user_id)
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found.")

    # Aggregate target images and detected faces as required
    target_images = []
    detected_faces = []
    for generation_id, generation in user_data.items():

        # Target images (one entry per target image, with rich metadata)
        for idx, target_url in enumerate(generation.get("signed_target_urls", [])):
            # You can enrich with any extra metadata you store per image
            item = {
                "target_id": f"{generation_id}_{idx}",
                "s3_path": generation.get("target_urls", [])[idx] if len(generation.get("target_urls", [])) > idx else None,
                "signed_url": target_url,
                "generation_id": generation_id,
                "target_index": idx,
                "filename": os.path.basename(target_url.split("?")[0]) if target_url else None,
                "upload_timestamp": generation.get("upload_timestamp", "upload"),
                "last_modified": generation.get("last_modified", ""),
                "file_size": generation.get("file_size", 0),
                "image_type": "full_target"
            }
            target_images.append(item)

        # Detected faces (one entry per detected face, with rich metadata)
        for i, face_url in enumerate(generation.get("signed_target_face_urls", [])):
            item = {
                "face_id": f"{generation_id}_{i}_0",
                "s3_path": generation.get("target_face_urls", [])[i] if len(generation.get("target_face_urls", [])) > i else None,
                "signed_url": face_url,
                "generation_id": generation_id,
                "target_index": i,
                "face_index": 0,
                "filename": os.path.basename(face_url.split("?")[0]) if face_url else None,
                "upload_timestamp": generation.get("upload_timestamp", "upload"),
                "last_modified": generation.get("last_modified", ""),
                "file_size": generation.get("face_file_size", 0),
                "image_type": "detected_face"
            }
            detected_faces.append(item)

    response = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "target_images": target_images,
        "target_count": len(target_images),
        "detected_faces": detected_faces,
        "detected_count": len(detected_faces),
        "usage_info": {
            "target_images": "Use 'signed_url' from target_images for file_url parameter in upload_targets endpoint",
            "detected_faces": "Individual faces extracted from target images - for preview/selection purposes"
        }
    }
    return JSONResponse(content=response)

@router.post("/upload_targets")
async def upload_targets(files: List[UploadFile] = File(...), user_id: str = Form(...), generation_id: str = Form(...), file_url : str = Form(...)):
    
    if GENERATION_DATA[generation_id]["status"] == "error" or GENERATION_DATA[generation_id]["status"] == "finished":
        raise HTTPException(status_code=500, detail="Already Processed for the given ID.")

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
        output_dir=OUTPUT_DIR,
        generation_data=GENERATION_DATA[generation_id]
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
    
    if GENERATION_DATA[request.generation_id]["status"] == "error" or GENERATION_DATA[request.generation_id]["status"] == "finished":
        raise HTTPException(status_code=500, detail="Already Processed for the given ID.")
    
    # Launch face swap in background
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, run_face_swap_background, request)

    GENERATION_DATA[request.generation_id]["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    GENERATION_DATA[request.generation_id]["faces_to_swap"] = len(request.source_indices)

    return {
        "message": "Face swap request accepted and is being processed",
        "generation_id": request.generation_id,
        "status": "processing",
        "status_url": f"{BASE_URL}/faceswap/status/{request.generation_id}"
    }

@router.get("/faceswap/status/{generation_id}")
async def get_swap_status(generation_id: str):
    
    if generation_id not in GENERATION_DATA :
        raise HTTPException(status_code=404, detail="Invalid Generation ID")
    
    generation_dir = os.path.join(OUTPUT_DIR, generation_id)
    
    log_file_path = os.path.join(generation_dir, "faceswap.log")

    progress = extract_last_percentage(log_file_path)

    if GENERATION_DATA[generation_id]["status"] == "finished" :
             
        return {
            "generation_id": generation_id,
            "progress": 100,
            "status": "completed",
            "message": "Face swap completed successfully",
            "signed_swap_url": GENERATION_DATA[generation_id]["signed_swap_url"],
            "file_url": GENERATION_DATA[generation_id]["file_url"],
            "error": None,
            "result_image_dimensions":GENERATION_DATA[generation_id]["result_image_dimensions"],
            "content_id": "65c7803c-ccd2-4f8f-8202-6e5fa0eb2281",
            "ai_tool_id": "13bc568d-bd0f-4d2a-bd97-fe88c6d47689"
        }
    
    elif GENERATION_DATA[generation_id]["status"] == "error" :

        return {
            "generation_id": generation_id,
            "progress": None,
            "status": "Failed",
            "message": f"Face swap failed\n{GENERATION_DATA[generation_id]['details']}",
            "signed_swap_url": None,
            "file_url": None,
            "error": GENERATION_DATA[generation_id]['details'],
            "result_image_dimensions":GENERATION_DATA[generation_id]['result_image_dimensions'],
            "content_id": "65c7803c-ccd2-4f8f-8202-6e5fa0eb2281",
            "ai_tool_id": "13bc568d-bd0f-4d2a-bd97-fe88c6d47689"
        }

    else:               
        current_progress = GENERATION_DATA[generation_id]["iterations"]*100 + progress
        
        progress = current_progress/GENERATION_DATA[generation_id]["faces_to_swap"]

        return {
            "generation_id": generation_id,
            "progress": progress,
            "status": "processing",
            "message": "Processing face swap...",
            "signed_swap_url": None,
            "file_url": "null",
            "error": None,
            "result_image_dimensions": None,
            "content_id": None,
            "ai_tool_id": None
        }
