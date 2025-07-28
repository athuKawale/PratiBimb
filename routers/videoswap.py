from datetime import datetime
import json
from fastapi import APIRouter, File, Request, UploadFile

router = APIRouter(
    prefix="/videoswap",
    tags=["videoswap"]
)

with open("static/video_templates.json", "r") as f:
    TEMPLATES_DATA = json.load(f)

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
    return TEMPLATES_DATA

@router.get("/templates/{template_id}/info")
async def get_template_info(template_id: str):
    """
    Returns detailed information for a specific video template.
    """
    template = next((t for t in TEMPLATES_DATA["available_video_templates"] if t["template_id"] == template_id), None)
    return template

@router.post("/uploadvideo")
async def upload_video(template_id: str, user_id: str):
    """
    Endpoint to upload a video for face swapping.
    """
    return {
        "generation_id": "09acd6e6-d1b1-4cdb-93a5-02579c6f9876",
        "template_id": "1",
        "face_groups_detected": {
            "1": 277,
            "0": 290
        },
        "detected_faces_urls": {
            "1": "https://sosnm1.shakticloud.ai:9024/immersobuk01/VFS/detected-faces/09acd6e6-d1b1-4cdb-93a5-02579c6f9876/cropped_faces/1/images_559.jpg?response-content-disposition=inline&response-content-type=image%2Fjpeg&AWSAccessKeyId=immersouser&Signature=C53OXWa16NuqXIxPMO9%2B%2BBn%2B2h0%3D&Expires=1753700477",
            "0": "https://sosnm1.shakticloud.ai:9024/immersobuk01/VFS/detected-faces/09acd6e6-d1b1-4cdb-93a5-02579c6f9876/cropped_faces/0/images_522.jpg?response-content-disposition=inline&response-content-type=image%2Fjpeg&AWSAccessKeyId=immersouser&Signature=SYuyJTgr%2FAWbOklH74Frp0zN%2FYM%3D&Expires=1753700477"
        },
        "total_face_groups": 2,
        "status": "processing",
        "credits": 50,
        "thumbnail_url": "https://sosnm1.shakticloud.ai:9024/immersobuk01/VFS/Thumbnails/1.png?AWSAccessKeyId=immersouser&Signature=a7%2BUJRnfpsenjo%2BaCd3SNqeGiOI%3D&Expires=1753783279"
    }

@router.post("/uploadnewfaces/{generation_id}/{group_id}")
async def upload_new_faces(generation_id: str, group_id: str, file : UploadFile = File(...)):
    """
    Endpoint to upload new faces for a specific generation and group.
    """
    
    return {
        "message": "Face uploaded successfully",
        "generation_id": "c1784609-0098-449d-b5f4-e6225c693ff7",
        "group_id": "0",
        "target_url": "https://sosnm1.shakticloud.ai:9024/immersobuk01/VFS/uploads/c1784609-0098-449d-b5f4-e6225c693ff7/new_faces/0.jpg?AWSAccessKeyId=immersouser&Signature=tGQg11ArIUbYINuut8VMRzQUsAM%3D&Expires=1753783449",
        "status": "ready_for_swap"
    }

@router.post("/faceswap/{generation_id}")
async def perform_face_swap(generation_id : str, request: Request):
    """
    Endpoint to perform face swap on the uploaded video.
    """
    
    json_data = await request.json()
    group_ids = json_data.get("group_ids", [])
    return {
        "status": "processing",
        "generation_id": generation_id,
        "message": "Face swap processing started. Check status using /faceswap/status/{uid}",
        "estimated_time": "1-2 minutes"
    }

@router.get("/faceswap/status/{generation_id}")
async def get_swap_status(generation_id : str):
    
    return {
        "generation_id": generation_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": None,
        "progress": 0.0,
        "status": "processing",
        "message": "Face swap is being processed"
    }