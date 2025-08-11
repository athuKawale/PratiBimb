import os
import uuid
import json
import asyncio
import requests
import traceback
from PIL import Image
from metadata import version
from datetime import datetime
from schema import SwapFaceRequest
from roop import globals as roop_globals
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from roop.globals import BASE_URL, DATA_FILE
from util.logger import extract_last_percentage
from util.async_operations import run_face_swap_background
from util.upload_target import process_and_save_target_faces
from util.upload_template import process_and_save_template_faces
from fastapi import APIRouter, HTTPException, Form, File, UploadFile

"""Globals"""

router = APIRouter(
    prefix="",
    tags=["faceswap"]
)

OUTPUT_DIR = "static/Face-swap/results"

TEMPLATES_DIR = "static/Face-swap/Templates"

GENERATION_DATA = {} 

with open("static/templates.json", "r") as f:
    TEMPLATES_DATA = json.load(f)


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
    try :

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
    except Exception as e:
        tb_str = traceback.format_exc()
        print("\nProblem in /templates endpoint:\n", tb_str)
        raise HTTPException(status_code=500, detail=f"Internal server error occurred.\n{e}")

    return {"available_templates": filtered_templates}

@router.get("/templates/{template_id}/info", response_model=Dict[str, Any])
async def get_template_info(template_id: str):
    """
    Returns detailed information for a specific template.
    """
    try :
            
        for template in TEMPLATES_DATA.get("available_templates", []):
            if template.get("template_id") == template_id:
                return template
    
        raise HTTPException(status_code=404, detail=f"Template with ID '{template_id}' not found.")

    except Exception as e:
        tb_str = traceback.format_exc()
        print("\nProblem in /template/{template_id}/info endpoint:\n", tb_str)
        raise HTTPException(status_code=500, detail=f"Internal server error occurred.\n{e}")

@router.post("/upload_template")
async def upload_template(
    template_id: Optional[str] = Form(None),
    user_id: str = Form(...),           
    files: Optional[UploadFile] = File(None)
):    

    try :
            
        template_ids = []

        if (template_id is not None and files is not None) or (template_id is None and files is None) :
            raise HTTPException(
                status_code=400,
                detail="Cannot provide both template_id and files. Choose one option."
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
        masked_face_url, detected_face_urls = process_and_save_template_faces(
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

    except Exception as e:
        tb_str = traceback.format_exc()
        print("\nProblem in /upload_template endpoint:\n", tb_str)
        raise HTTPException(status_code=500, detail=f"Internal server error occurred.\n{e}")

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

    try :
            
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

    except Exception as e:
        tb_str = traceback.format_exc()
        print("\nProblem in /users/{user_id}/previous_target_faces endpoint:\n", tb_str)
        raise HTTPException(status_code=500, detail=f"Internal server error occurred.\n{e}")

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
    
    try :

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
    
    except Exception as e:
        tb_str = traceback.format_exc()
        print("\nProblem in /upload_targets endpoint:\n", tb_str)
        raise HTTPException(status_code=500, detail=f"Internal server error occurred.\n{e}")

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
    try :

        if GENERATION_DATA[request.generation_id]["status"] == "error" or GENERATION_DATA[request.generation_id]["status"] == "finished":
            raise HTTPException(status_code=500, detail="Already Processed for the given ID.")
        
        # Launch face swap in background
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, run_face_swap_background, GENERATION_DATA, OUTPUT_DIR, request)

        GENERATION_DATA[request.generation_id]["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        GENERATION_DATA[request.generation_id]["faces_to_swap"] = len(request.source_indices)

    except Exception as e:
        tb_str = traceback.format_exc()
        print("\nProblem in /swap_face endpoint:\n", tb_str)
        raise HTTPException(status_code=500, detail=f"Internal server error occurred.\n{e}")

    return {
        "message": "Face swap request accepted and is being processed",
        "generation_id": request.generation_id,
        "status": "processing",
        "status_url": f"{BASE_URL}/faceswap/status/{request.generation_id}"
    }

@router.get("/faceswap/status/{generation_id}")
async def get_swap_status(generation_id: str):
        
    try :
    
        if generation_id not in GENERATION_DATA :
            raise HTTPException(status_code=404, detail="Invalid Generation ID")
        
        generation_dir = os.path.join(OUTPUT_DIR, generation_id)
        
        log_file_path = os.path.join(generation_dir, "faceswap.log")

        progress = extract_last_percentage(log_file_path)
    
    except Exception as e:
        tb_str = traceback.format_exc()
        print("\nProblem in /faceswap/status/{generation_id} endpoint:\n", tb_str)
        raise HTTPException(status_code=500, detail=f"Internal server error occurred.\n{e}")


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
