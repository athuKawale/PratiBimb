from datetime import datetime
import json
from fastapi import APIRouter, File, Form, Request, UploadFile
from roop import globals as roop_globals
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
import glob
import re
import cv2
# Add project root to path to allow relative imports
sys.path.append(os.getcwd())
import asyncio
from roop import globals as roop_globals
from roop.core import batch_process_regular
from roop.face_util import extract_face_images
from roop.ProcessEntry import ProcessEntry
from roop.FaceSet import FaceSet
from roop import utilities as util
from roop.capturer import get_video_frame_total
import logging

router = APIRouter(
    prefix="/videoswap",
    tags=["videoswap"]
)

OUTPUT_DIR = "static/Video-swap"
TEMPLATES_DIR = "static/Video-swap/Templates"

with open("static/video_templates.json", "r") as f:
    TEMPLATES_DATA = json.load(f)

GENERATION_DATA = {
    "outputDir": OUTPUT_DIR,
    "generationId": "",
    "userId": "",
    "groupId": "",
    "templateId": "",
    "templatePath": "",
    "templateData": TEMPLATES_DATA,
    "face_groups_detected" : {},
    "detected_faces_urls": {},
    "total_face_groups": 0,
    "thumbnail_url": "",
    "list_files_process": [],
    "target_url": "http://localhost:8000/",
    "created_at": "",
}

# Background task to run face swap

def run_face_swap_blocking(group_ids, generation_id):
    asyncio.run(run_face_swap(group_ids, generation_id))


async def run_face_swap(group_ids: list, generation_id: str):

    log_and_print("Starting face swap process...")  

    with open(log_file_path, 'a') as f:
        with redirect_stdout(f), redirect_stderr(f):
            batch_process_regular(
                swap_model="InSwapper 128",
                output_method="File",
                files=GENERATION_DATA["list_files_process"],
                masking_engine=roop_globals.mask_engine,
                new_clip_text=roop_globals.clip_text,
                use_new_method=True,
                imagemask=None,
                restore_original_mouth=False,
                num_swap_steps=1,
                progress=None,
                selected_index=0
            )

    list_of_files = glob.glob(os.path.join(roop_globals.output_path, '*.mp4'))

    if not list_of_files:
        print("Error: Processing finished, but no output file was created.")
        return

    latest_file = max(list_of_files, key=os.path.getctime)
    os.rename(latest_file, os.path.join(roop_globals.output_path, "output.mp4"))

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Face swap completed.")


# Setup logging
log_file_path = os.path.join(OUTPUT_DIR, "faceswap.log")
logger = logging.getLogger("faceswap")
logger.setLevel(logging.INFO)

# Avoid duplicate handlers if re-called
if not logger.handlers:
    file_handler = logging.FileHandler(log_file_path, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def log_and_print(msg: str):
    print(msg)
    logger.info(msg)

def extract_last_percentage(log_file: str) -> float:
    if not os.path.exists(log_file):
        print("file not found:", log_file)
        return 0.0

    with open(log_file, "r") as f:
        lines = f.readlines()

    percent = 0.0
    pattern = re.compile(r'Processing:\s+(\d{1,3})%')

    for line in reversed(lines):
        match = pattern.search(line)
        if match:
            percent = float(match.group(1))
            break

    return percent


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
    return GENERATION_DATA["templateData"]

@router.get("/templates/{template_id}/info")
async def get_template_info(template_id: str):
    """
    Returns detailed information for a specific video template.
    """
    template = next((t for t in GENERATION_DATA["templateData"]["available_video_templates"] if t["template_id"] == template_id), None)
    
    return template

@router.post("/uploadvideo")
async def upload_video(user_id: str = Form(...), template_id: str = Form(...)):
    """
    Endpoint to upload a video for face swapping.
    """
    GENERATION_DATA["generationId"] = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{user_id}"
    GENERATION_DATA["userId"] = user_id
    GENERATION_DATA["outputDir"] = f"{OUTPUT_DIR}/{GENERATION_DATA['generationId']}"
    GENERATION_DATA["templatePath"] = TEMPLATES_DIR + "/" + template_id + ".mp4"

    roop_globals.output_path = OUTPUT_DIR + "/" + GENERATION_DATA["generationId"] + "/" + "output"
    os.makedirs(roop_globals.output_path, exist_ok=True)
    
    if roop_globals.CFG.clear_output:
        util.clean_dir(roop_globals.output_path)

    roop_globals.target_face_index = 0 # Default target face index for video face swap
    roop_globals.target_path = GENERATION_DATA["templatePath"]
    roop_globals.selected_enhancer = "GFPGAN" 
    roop_globals.distance_threshold = 0.80
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

    print("Analyzing target video for faces...")

    target_face_data = extract_face_images(roop_globals.target_path, (True, 0))
    
    if not target_face_data:
        print("Error: No face detected in the source image.")
        sys.exit(1)

    face_set = FaceSet()
    for face_data in target_face_data:
        face = face_data[0]
        face.mask_offsets = (0,0,0,0,1,20)
        face_set.faces.append(face)
        roop_globals.TARGET_FACES.append(face_set)

    temp = face_set.faces[0]
    face_set.faces[0] = face_set.faces[roop_globals.target_face_index]
    face_set.faces[roop_globals.target_face_index] = temp
    
    face_set.AverageEmbeddings()
    print(f"Found {len(target_face_data)} face(s), using the first one.")


    # Save cropped faces for target video
    for i, (face, face_image) in enumerate(target_face_data):
        try:
            face_filename = f"target_{i}_{GENERATION_DATA['generationId']}.jpg"
            face_path = GENERATION_DATA["outputDir"] + "/" + face_filename
            cv2.imwrite(str(face_path), face_image)
            print(f"Saved target face {i} to {face_path}")
        except Exception as e:
            print(f"Error saving face {i}: {e}")
            continue

    # Prepare target file process entry
    list_files_process = []
    process_entry = ProcessEntry(GENERATION_DATA["templatePath"], 0, 0, 0)
    total_frames = get_video_frame_total(GENERATION_DATA["templatePath"])

    if total_frames is None or total_frames < 1:
        print(f"Warning: Could not read total frames from video {GENERATION_DATA['templatePath']}")
        total_frames = 1

    process_entry.endframe = total_frames
    list_files_process.append(process_entry)

    GENERATION_DATA["list_files_process"] = list_files_process

    print(f"Target set to: {GENERATION_DATA['templatePath']}")
    print(f"Target has {process_entry.endframe} frames.")


    GENERATION_DATA["templateId"] = template_id
    GENERATION_DATA["face_groups_detected"] = {
        "1": 277,
        "0": 290
    }
    GENERATION_DATA["detected_faces_urls"] = {
        "1": "https://sosnm1.shakticloud.ai:9024/immersobuk01/VFS/detected-faces/09acd6e6-d1b1-4cdb-93a5-02579c6f9876/cropped_faces/1/images_559.jpg?response-content-disposition=inline&response-content-type=image%2Fjpeg&AWSAccessKeyId=immersouser&Signature=C53OXWa16NuqXIxPMO9%2B%2BBn%2B2h0%3D&Expires=1753700477",
        "0": "https://sosnm1.shakticloud.ai:9024/immersobuk01/VFS/detected-faces/09acd6e6-d1b1-4cdb-93a5-02579c6f9876/cropped_faces/0/images_522.jpg?response-content-disposition=inline&response-content-type=image%2Fjpeg&AWSAccessKeyId=immersouser&Signature=SYuyJTgr%2FAWbOklH74Frp0zN%2FYM%3D&Expires=1753700477"
    }
    GENERATION_DATA["total_face_groups"] = 0

    GENERATION_DATA["thumbnail_url"] = next((t for t in GENERATION_DATA["templateData"]["available_video_templates"] if t["template_id"] == template_id), None)
    
    return {
        "generation_id": GENERATION_DATA["generationId"],
        "template_id": template_id,
        "face_groups_detected": GENERATION_DATA["face_groups_detected"],
        "detected_faces_urls": GENERATION_DATA["detected_faces_urls"],
        "total_face_groups": GENERATION_DATA["total_face_groups"],
        "status": "processing",
        "credits": 50,
        "thumbnail_url": GENERATION_DATA["thumbnail_url"],
    }

@router.post("/uploadnewfaces/{generation_id}/{group_id}")
async def upload_new_faces(generation_id: str, group_id: str, file : UploadFile = File(...)):
    """
    Endpoint to upload new faces for a specific generation and group.
    """
    source_file_name = f"{group_id}_{generation_id}.jpg"
    os.makedirs(os.path.join(GENERATION_DATA["outputDir"], group_id), exist_ok=True)
    source_file_path = os.path.join(GENERATION_DATA["outputDir"], group_id,  source_file_name)

    roop_globals.source_path_video.append(source_file_path)

    with open(source_file_path, "wb") as buffer:
        buffer.write(file.file.read())

    print("Analyzing source image...")

    source_faces_data = extract_face_images(source_file_path, (False, 0))
    if not source_faces_data:
        print("Error: No face detected in the source image.")
        sys.exit(1)
    face_set = FaceSet()
    face = source_faces_data[0][0]
    face.mask_offsets = (0,0,0,0,1,20)
    face_set.faces.append(face)
    roop_globals.INPUT_FACESETS.append(face_set)
    
    print(f"Found {len(source_faces_data)} face(s), using the first one.")

    # Save cropped faces for source image
    for i, (face, face_image) in enumerate(source_faces_data):
        try:
            face_filename = f"source_{i}_{GENERATION_DATA['generationId']}.jpg"
            face_path = GENERATION_DATA["outputDir"] + "/" + group_id + "/" + face_filename
            cv2.imwrite(str(face_path), face_image)
            print(f"Saved source face {i} to {face_path}")
        except Exception as e:
            print(f"Error saving face {i}: {e}")
            continue
    

    return {
        "message": "Face uploaded successfully",
        "generation_id": GENERATION_DATA["generationId"],
        "group_id": group_id,
        "target_url": GENERATION_DATA["target_url"] + source_file_path,
        "status": "ready_for_swap"
    }

@router.post("/faceswap/{generation_id}")
async def perform_face_swap(request: Request, generation_id : str):
    """
    Endpoint to perform face swap on the uploaded video.
    """
    
    json_data = await request.json()
    group_ids = json_data.get("group_ids", [])
    
    # Launch face swap in background
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, run_face_swap_blocking, group_ids, generation_id)

    GENERATION_DATA["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "status": "processing",
        "generation_id": GENERATION_DATA["generationId"],
        "message": "Face swap processing started. Check status using /faceswap/status/{uid}",
        "estimated_time": "1-2 minutes"
    }

@router.get("/faceswap/status/{generation_id}")
async def get_swap_status(generation_id: str):
    output_file = os.path.join(roop_globals.output_path, "output.mp4")
    log_file_path

    if os.path.exists(output_file):
        return {
            "generation_id": GENERATION_DATA["generationId"],
            "created_at": GENERATION_DATA["created_at"],
            "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "progress": 100.0,
            "status": "finished",
            "message": "Face swap completed successfully"
        }
    else:
        progress = extract_last_percentage(log_file_path)
        return {
            "generation_id": GENERATION_DATA["generationId"],
            "created_at": GENERATION_DATA["created_at"],
            "finished_at": None,
            "progress": progress,
            "status": "processing",
            "message": f"Face swap is {progress}% complete",
        }
