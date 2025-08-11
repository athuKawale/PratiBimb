import os
import glob
import asyncio
import datetime
from fastapi import HTTPException
from roop.globals import BASE_URL
from schema import SwapFaceRequest
from util.logger import log_and_print
from roop.ProcessEntry import ProcessEntry
from roop.core import batch_process_regular
from roop.capturer import get_video_frame_total
from contextlib import redirect_stderr, redirect_stdout
from util.save_to_json import save_generation_data_to_json


"""Run face swap in background using asyncio"""

def run_face_swap_background(GENERATION_DATA, OUTPUT_DIR, request):
    asyncio.run(run_face_swap(GENERATION_DATA, OUTPUT_DIR, request))

async def run_face_swap(GENERATION_DATA, OUTPUT_DIR, request : SwapFaceRequest):

    generation_id = request.generation_id

    generation_dir = os.path.join(OUTPUT_DIR, generation_id)
        
    log_file_path = os.path.join(generation_dir, "faceswap.log")

    log_and_print(OUTPUT_DIR, generation_id, "Starting face swap process...")

    source_indices = request.source_indices
    target_indices = request.target_indices
    
    generation_data = GENERATION_DATA.get(generation_id)

    if not generation_data:
        raise HTTPException(status_code=400,detail="Invalid generation_id. Please upload template and target images first.")
    
    generation_data["globals"].target_path = generation_data["template_path"]

    if not generation_data["target_paths"]:
        raise HTTPException(status_code=500, detail="No target images found for the given generation_id.")
    
    # Select the target image based on the first target_index provided
    target_image_path = generation_data["target_paths"][0]
    generation_data["globals"].source_path = target_image_path

    generation_data["globals"].output_path = os.path.join(generation_dir, "swapped")
    
    Template = generation_data["template_path"]

    for i in range(len(source_indices)) :

        generation_data["globals"].INPUT_FACESETS = [generation_data["globals"].TEMP_FACESET[target_indices[i]]]

        if len(generation_data["globals"].TARGET_FACES[0].faces) > 1 :
            # Move Embedding of faces to faces[0]  and setting faces embedding to None so that averageEmbeddings can be calculated.
            generation_data["globals"].TARGET_FACES[0].faces[0].embedding = generation_data["globals"].TARGET_FACES[0].embedding
            generation_data["globals"].TARGET_FACES[0].embedding = None

            # getting i-th target face at front of list so that it is swapped
            temp = generation_data["globals"].TARGET_FACES[0].faces[0]
            generation_data["globals"].TARGET_FACES[0].faces[0] = generation_data["globals"].TARGET_FACES[0].faces[source_indices[i]]
            generation_data["globals"].TARGET_FACES[0].faces[source_indices[i]] = temp

            generation_data["globals"].TARGET_FACES[0].AverageEmbeddings()

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
                        swap_model=generation_data["globals"].face_swapper_model,
                        output_method="File",
                        files=list_files_process,
                        masking_engine=generation_data["globals"].mask_engine,
                        new_clip_text=generation_data["globals"].clip_text,
                        use_new_method=True,
                        imagemask=None,
                        restore_original_mouth=False,
                        num_swap_steps=1,
                        progress=None,
                        selected_index=0,
                        globals=generation_data["globals"]
                    )

        except Exception as e:

            print(f"Error during face swap processing: {e}\nIteration: {generation_data['iteration']}\n")

            generation_data["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            generation_data["status"] = "error"

            generation_data["details"] = f"Faceswap Failed : \n{e}"

            #Save GENRATION DATA to json
            save_generation_data_to_json(generation_data["user_id"], generation_id, GENERATION_DATA)
            
            log_and_print(OUTPUT_DIR, generation_id, f"Face swap failed: {e}")

            generation_data["globals"].INPUT_FACESETS = []
            generation_data["globals"].TARGET_FACES = []

            return 
    
    
        generation_data["iterations"] += 1
        
        if os.path.exists(os.path.join(generation_data["globals"].output_path, 'output.png')):
            os.remove(os.path.join(generation_data["globals"].output_path, 'output.png'))

        Template = glob.glob(os.path.join(generation_data["globals"].output_path, '*_*.png'))[0]

        if not Template:
            print("Error: No output file created during processing.")
            return
        
        os.rename(Template, os.path.join(generation_data["globals"].output_path, "output.png"))
        Template = os.path.join(generation_data["globals"].output_path, "output.png")

    
    file_path = glob.glob(os.path.join(generation_data["globals"].output_path, '*.png'))[0]
    file_url = f"{BASE_URL}/{file_path}"
    signed_swap_url = f"{file_url}?dummy_signed_url"
    
    generation_data["globals"].INPUT_FACESETS = []
    generation_data["globals"].TARGET_FACES = []

    generation_data["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    generation_data["status"] = "finished"
    generation_data["file_url"] = file_url
    generation_data["signed_swap_url"] = signed_swap_url

    #Save GENRATION DATA to json
    save_generation_data_to_json(generation_data["user_id"], generation_id, GENERATION_DATA)
   
    print(f"[{generation_data['finished_at']}] Face swap completed.")


"""Run Videoface swap in background using asyncio"""

def run_video_swap_background(GENERATION_DATA, OUTPUT_DIR, group_ids, generation_id):
    asyncio.run(run_video_swap(GENERATION_DATA, OUTPUT_DIR, group_ids, generation_id))

async def run_video_swap(GENERATION_DATA : dict, OUTPUT_DIR : str, group_ids: list, generation_id: str):

    generation_data = GENERATION_DATA.get(generation_id)

    if not generation_data:
        raise HTTPException(status_code=400,detail="Invalid generation_id. Please upload template and target images first.")
  
    generation_dir = os.path.join(OUTPUT_DIR, generation_id)
        
    log_file_path = os.path.join(generation_dir, "faceswap.log")

    log_and_print(OUTPUT_DIR, generation_id, "Starting face swap process...")  

    Template = generation_data["templatePath"]

    for i in group_ids:

        generation_data["globals"].INPUT_FACESETS = [generation_data["globals"].VIDEO_INPUTFACES[i]]

        if len(generation_data["globals"].TARGET_FACES[0].faces) > 1 :
            # Move Embedding of faces to faces[0]  and setting faces embedding to None so that averageEmbeddings can be calculated.
            generation_data["globals"].TARGET_FACES[0].faces[0].embedding = generation_data["globals"].TARGET_FACES[0].embedding
            generation_data["globals"].TARGET_FACES[0].embedding = None

            # getting i-th target face at front of list so that it is swapped
            temp = generation_data["globals"].TARGET_FACES[0].faces[0]
            generation_data["globals"].TARGET_FACES[0].faces[0] = generation_data["globals"].TARGET_FACES[0].faces[i]
            generation_data["globals"].TARGET_FACES[0].faces[i] = temp

            generation_data["globals"].TARGET_FACES[0].AverageEmbeddings()

        # Prepare target file process entry
        list_files_process = []
        process_entry = ProcessEntry(Template, 0, 0, 0)
        total_frames = get_video_frame_total(Template)

        if total_frames is None or total_frames < 1:
            print(f"Warning: Could not read total frames from video {Template}")
            total_frames = 1

        process_entry.endframe = total_frames
        list_files_process.append(process_entry)

        #This clears the log file
        with open(log_file_path, 'w'):
            pass

        try :

            with open(log_file_path, 'a') as f:
                with redirect_stdout(f), redirect_stderr(f):
                    batch_process_regular(
                        swap_model=generation_data["globals"].face_swapper_model,
                        output_method="File", 
                        files=list_files_process,
                        masking_engine=generation_data["globals"].mask_engine,
                        new_clip_text=generation_data["globals"].clip_text,
                        use_new_method=True,
                        imagemask=None,
                        restore_original_mouth=False,
                        num_swap_steps=1,
                        progress=None,
                        selected_index=0,
                        globals=generation_data["globals"]
                    )

        except Exception as e:
            print(f"Error during face swap processing: {e}\nIteration: {generation_data['iteration']}\n")
            
            generation_data["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            generation_data["status"] = "error"

            generation_data["details"] = f"Faceswap Failed : \n{e}"

            log_and_print(OUTPUT_DIR, generation_id, f"Face swap failed: {e}")

            generation_data["globals"].INPUT_FACESETS = []
            generation_data["globals"].TARGET_FACES = []
            generation_data["globals"].VIDEO_INPUTFACES = []

            return
        
        generation_data["iteration"] = generation_data["iteration"] + 1
        
        if os.path.exists(os.path.join(generation_data["globals"].output_path, 'output.mp4')):
            os.remove(os.path.join(generation_data["globals"].output_path, 'output.mp4'))

        Template = glob.glob(os.path.join(generation_data["globals"].output_path, '*.mp4'))[0]

        if not Template:
            print("Error: No output file created during processing.")
            return
        
        os.rename(Template, os.path.join(generation_data["globals"].output_path, "output.mp4"))
        Template = os.path.join(generation_data["globals"].output_path, "output.mp4")

    generation_data["globals"].INPUT_FACESETS = []
    generation_data["globals"].TARGET_FACES = []
    generation_data["globals"].VIDEO_INPUTFACES = []

    generation_data["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    generation_data["status"] = "finished"

    generation_data["iteration"] = 0

    print(f"[{generation_data['finished_at']}] Face swap completed.")
