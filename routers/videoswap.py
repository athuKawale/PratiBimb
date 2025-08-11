import os
import cv2
import uuid
import json
import asyncio
import traceback
from datetime import datetime
from roop.FaceSet import FaceSet
from roop.globals import BASE_URL
from roop import utilities as util
from roop import globals as roop_globals
from roop.face_util import extract_face_images
from util.logger import extract_last_percentage
from util.async_operations import run_video_swap_background
from fastapi import APIRouter, File, Form, Request, UploadFile, HTTPException

"""Globals"""

router = APIRouter(
    prefix="/videoswap",
    tags=["videoswap"]
)

OUTPUT_DIR = "static/Video-swap"

TEMPLATES_DIR = "static/Video-swap/Templates"

GENERATION_DATA = {}

with open("static/video_templates.json", "r") as f:
    TEMPLATES_DATA = json.load(f)

"""API ENDPOINTS"""

@router.get("/health")
async def health_check():
    """
    Health check endpoint to verify if the service is running.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "database": "connected",
        "s3": "connected",
        "thread_pool": "active"
    }

@router.get("/video-templates")
async def get_video_templates():
    """
    Returns a list of available video templates.
    """
    if TEMPLATES_DATA is None:
        raise HTTPException(status_code=500, detail="No Template Data found.")
    
    return TEMPLATES_DATA

@router.get("/templates/{template_id}/info")
async def get_template_info(template_id: str):
    """
    Returns detailed information for a specific video template.
    """
    try :

        template = next((t for t in TEMPLATES_DATA["available_video_templates"] if t["template_id"] == template_id), None)
    
    except Exception as e:
        tb_str = traceback.format_exc()
        print("\nProblem in /templates/{template_id}/info endpoint:\n", tb_str)
        raise HTTPException(status_code=500, detail=f"Internal server error occurred.\n{e}")

    return template

@router.post("/uploadvideo")
async def upload_video(user_id: str = Form(...), template_id: str = Form(...)):
    """
    Endpoint to upload a video for face swapping.
    """

    try : 
        
        generation_id = f"{uuid.uuid4()}"

        GENERATION_DATA[generation_id] =  {
            "outputDir": OUTPUT_DIR,
            "generationId": "",
            "userId": "",
            "groupId": "",
            "templateId": "",
            "templatePath": "",
            "face_groups_detected" : {},
            "detected_faces_urls": {},
            "total_face_groups": 0,
            "faces_to_swap": 1,
            "thumbnail_url": "",
            "created_at": "",
            "finished_at": "",
            "iteration": 0,
            "status": "processing",
        }
        GENERATION_DATA[generation_id]["userId"] = user_id
        GENERATION_DATA[generation_id]["outputDir"] = f"{OUTPUT_DIR}/{generation_id}"
        GENERATION_DATA[generation_id]["templatePath"] = TEMPLATES_DIR + "/" + template_id + ".mp4"

        roop_globals.output_path = OUTPUT_DIR + "/" + generation_id + "/" + "output"

        os.makedirs(roop_globals.output_path, exist_ok=True)
        
        if roop_globals.CFG.clear_output:
            util.clean_dir(roop_globals.output_path)

        roop_globals.target_path = GENERATION_DATA[generation_id]["templatePath"]
        roop_globals.distance_threshold = 0.65
        roop_globals.blend_ratio = 1
        roop_globals.face_swap_mode = "selected"
        roop_globals.no_face_action = 0 # Use untouched original frame
        roop_globals.keep_frames = False
        roop_globals.wait_after_extraction = False
        roop_globals.vr_mode = False
        roop_globals.autorotate_faces = True
        roop_globals.subsample_size = 128
        roop_globals.mask_engine = 'None'
        roop_globals.clip_text = None
        roop_globals.execution_threads = roop_globals.CFG.max_threads
        roop_globals.video_encoder = roop_globals.CFG.output_video_codec
        roop_globals.video_quality = roop_globals.CFG.video_quality
        roop_globals.max_memory = roop_globals.CFG.memory_limit if roop_globals.CFG.memory_limit > 0 else None

        print("\n\nAnalyzing target video for faces...\n\n")

        target_face_data = extract_face_images(roop_globals.target_path, (True, 0))
        
        if not target_face_data:
            raise HTTPException(
                status_code=422,
                detail="No face detected in the image"
            )


        print(f"Found {len(target_face_data)} face(s) in the target video.\n\n")

        print("Creating face set for target video...\n\n")

        face_set = FaceSet()    
        for face_data in target_face_data:
            face = face_data[0]
            face.mask_offsets = (0,0,0,0,1,20)
            face_set.faces.append(face)

        if len(face_set.faces) > 1:
            face_set.AverageEmbeddings()
        
        if len(face_set.faces) <= 1:
            roop_globals.face_swap_mode = 'all_input'
            
        roop_globals.TARGET_FACES.append(face_set)

        print(f"Found {len(target_face_data)} face(s)\n\n")

        # Save cropped faces for target video
        for i, (face, face_image) in enumerate(target_face_data):
            try:
                face_filename = f"target_{i}_{generation_id}.jpg"
                face_path = GENERATION_DATA[generation_id]["outputDir"] + "/" + face_filename
                cv2.imwrite(str(face_path), face_image)

                GENERATION_DATA[generation_id]["detected_faces_urls"][i] = f"{BASE_URL}/{face_path}"

            except Exception as e:
                print(f"Error saving face {i}: {e}\n\n")
                continue

        GENERATION_DATA[generation_id]["status"] = "processing"
        GENERATION_DATA[generation_id]["templateId"] = template_id
        GENERATION_DATA[generation_id]["face_groups_detected"] = {
            "1": 277,
            "0": 290
        }
        GENERATION_DATA[generation_id]["total_face_groups"] = len(GENERATION_DATA[generation_id]["detected_faces_urls"])

        GENERATION_DATA[generation_id]["thumbnail_url"] = next((t["thumbnail_url"] for t in TEMPLATES_DATA["available_video_templates"] if t["template_id"] == template_id), None)

    except Exception as e:

        tb_str = traceback.format_exc()
        print("\nProblem in /uploadvideo :\n", tb_str)
        raise HTTPException(status_code=500, detail=f"Internal server error occurred.\n{e}")
    
    return {
        "generation_id": generation_id,
        "template_id": template_id,
        "face_groups_detected": GENERATION_DATA[generation_id]["face_groups_detected"],
        "detected_faces_urls": GENERATION_DATA[generation_id]["detected_faces_urls"],
        "total_face_groups": GENERATION_DATA[generation_id]["total_face_groups"],
        "status": GENERATION_DATA[generation_id]["status"],
        "credits": 50,
        "thumbnail_url": GENERATION_DATA[generation_id]["thumbnail_url"],
    }

@router.post("/uploadnewfaces/{generation_id}/{group_id}")
async def upload_new_faces(generation_id: str, group_id: str, file : UploadFile = File(...)):
    """
    Endpoint to upload new faces for a specific generation and group.
    """
    try :
            
        source_file_name = f"{group_id}_{generation_id}.jpg"
        os.makedirs(os.path.join(GENERATION_DATA[generation_id]["outputDir"], group_id), exist_ok=True)
        source_file_path = os.path.join(GENERATION_DATA[generation_id]["outputDir"], group_id,  source_file_name)

        roop_globals.source_path_video.append(source_file_path)

        with open(source_file_path, "wb") as buffer:
            buffer.write(file.file.read())

        print("Analyzing source image...\n\n")

        source_faces_data = extract_face_images(source_file_path, (False, 0))
        if not source_faces_data:
            raise HTTPException(
                status_code=422,
                detail="No face detected in the image"
            )

        face_set = FaceSet()
        face = source_faces_data[0][0]
        face.mask_offsets = (0,0,0,0,1,20)
        face_set.faces.append(face)
        roop_globals.VIDEO_INPUTFACES.append(face_set)
        
        # Put total faces to swap in GENERATION DATA
        GENERATION_DATA[generation_id]["faces_to_swap"] = len(roop_globals.VIDEO_INPUTFACES)

        print(f"Found {len(source_faces_data)} face(s), using the first one.\n\n")

        # Save cropped faces for source image
        for i, (face, face_image) in enumerate(source_faces_data):
            try:
                face_filename = f"source_{i}_{generation_id}.jpg"
                face_path = GENERATION_DATA[generation_id]["outputDir"] + "/" + group_id + "/" + face_filename
                cv2.imwrite(str(face_path), face_image)
                print(f"Saved source face {i} to {face_path}\n\n")
            except Exception as e:
                print(f"Error saving face {i}: {e}\n\n")
                continue
    
    except Exception as e:
        tb_str = traceback.format_exc()
        print("\nProblem in /uploadnewfaces endpoint:\n", tb_str)
        raise HTTPException(status_code=500, detail=f"Internal server error occurred.\n{e}")
    

    return {
        "message": "Face uploaded successfully",
        "generation_id": generation_id,
        "group_id": group_id,
        "base_url": f"{BASE_URL}/{source_file_path}",
        "status": "ready_for_swap"
    }

@router.post("/faceswap/{generation_id}")
async def faceswap(request: Request, generation_id : str):
    """
    Endpoint to perform face swap on the uploaded video.
    """
    try :

        json_data = await request.json()
        group_ids = json_data.get("group_ids", [])
        
        # Launch face swap in background
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, run_video_swap_background, group_ids, generation_id)

        GENERATION_DATA[generation_id]["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    except Exception as e:
        tb_str = traceback.format_exc()
        print("\nProblem in /faceswap/{generation_id} :\n", tb_str)
        raise HTTPException(status_code=500, detail=f"Internal server error occurred.\n{e}")

    return {
        "status": "processing",
        "generation_id": generation_id,
        "message": "Face swap processing started. Check status using /faceswap/status/{generation_id}",
        "estimated_time": "1-2 minutes"
    }

@router.get("/faceswap/status/{generation_id}")
async def get_swap_status(generation_id: str):
    
    try :
            
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
            "created_at": GENERATION_DATA[generation_id]["created_at"],
            "finished_at": GENERATION_DATA[generation_id]["finished_at"],
            "progress": 100.0,
            "status": GENERATION_DATA[generation_id]["status"],
            "message": "Face swap completed successfully"
        }
    
    elif GENERATION_DATA[generation_id]["status"] == "error" :

        return {
            "generation_id": generation_id,
            "created_at": GENERATION_DATA[generation_id]["created_at"],
            "finished_at": GENERATION_DATA[generation_id]["finished_at"],
            "progress": None,
            "status": GENERATION_DATA[generation_id]["status"],
            "message": "Error While Swapping Faces."
        }

    else:            
        
        current_progress = GENERATION_DATA[generation_id]["iteration"] *100 + progress

        progress = current_progress/ GENERATION_DATA[generation_id]["faces_to_swap"]
            
        return {
            "generation_id": generation_id,
            "created_at": GENERATION_DATA[generation_id]["created_at"],
            "finished_at": "Currently progressing",
            "progress": progress,
            "status": GENERATION_DATA[generation_id]["status"],
            "message": f"Face swap is {progress}% complete",
        }
