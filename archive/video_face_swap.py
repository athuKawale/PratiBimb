import argparse
import os
import sys
import glob
import cv2
# Add project root to path to allow relative imports
sys.path.append(os.getcwd())

import roop.globals
from roop.core import batch_process_regular
from roop.face_util import extract_face_images
from roop.ProcessEntry import ProcessEntry
from roop.FaceSet import FaceSet
from roop import utilities as util
from roop.capturer import get_video_frame_total

def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Face swap in a video.")
    parser.add_argument('-s', '--source-img', dest='source_img', required=True, help='Path to the source image with the face.')
    parser.add_argument('-t', '--target-video', dest='target_video', required=True, help='Path to the target video file.')
    parser.add_argument('-o', '--output-file', dest='output_file', required=True, help='Path for the output video file.')
    
    return parser.parse_args()

def run():
    """Main execution function."""
    args = get_args()

    # Basic validation
    if not os.path.isfile(args.source_img):
        print(f"Error: Source image not found at {args.source_img}")
        sys.exit(1)
    if not os.path.isfile(args.target_video):
        print(f"Error: Target video not found at {args.target_video}")
        sys.exit(1)
    if not util.is_video(args.target_video):
        print(f"Error: Target must be a video file.")
        sys.exit(1)

    # Prepare environment and globals
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    roop.globals.output_path = output_dir
    if roop.globals.clear_output:
        util.clean_dir(roop.globals.output_path)

    roop.globals.source_path = args.source_img
    roop.globals.target_path = args.target_video
    roop.globals.face_swap_mode = "selected"
    roop.globals.clip_text = None
    roop.globals.execution_threads = roop.globals.max_threads
    roop.globals.max_memory = roop.globals.memory_limit if roop.globals.memory_limit > 0 else None

    # Load source face (for swapping)
    print("Analyzing source image...")

    source_faces_data = extract_face_images(args.source_img, (False, 0))
    if not source_faces_data:
        print("Error: No face detected in the source image.")
        sys.exit(1)
    face_set = FaceSet()
    face = source_faces_data[0][0]
    face.mask_offsets = (0,0,0,0,1,20)
    face_set.faces.append(face)
    roop.globals.INPUT_FACESETS.append(face_set)
    print(f"Found {len(source_faces_data)} face(s), using the first one.")

    target_face_data = extract_face_images(args.target_video, (True, 0))
    if not target_face_data:
        print("Error: No face detected in the source image.")
        sys.exit(1)

    face_set = FaceSet()
    for face_data in target_face_data:
        face = face_data[0]
        face.mask_offsets = (0,0,0,0,1,20)
        face_set.faces.append(face)
        roop.globals.TARGET_FACES.append(face_set)
    
    face_set.AverageEmbeddings()
    print(f"Found {len(target_face_data)} face(s), using the first one.")

    # Save cropped faces for source image
    for i, (face, face_image) in enumerate(source_faces_data):
        try:
            face_filename = f"source_{i}.jpg"
            face_path = output_dir + "/" + face_filename
            cv2.imwrite(str(face_path), face_image)
            print(f"Saved source face {i} to {face_path}")
        except Exception as e:
            print(f"Error saving face {i}: {e}")
            continue

    # Save cropped faces for target video
    for i, (face, face_image) in enumerate(target_face_data):
        try:
            face_filename = f"target_{i}.jpg"
            face_path = output_dir + "/" + face_filename
            cv2.imwrite(str(face_path), face_image)
            print(f"Saved target face {i} to {face_path}")
        except Exception as e:
            print(f"Error saving face {i}: {e}")
            continue

    # Prepare target file process entry
    list_files_process = []
    process_entry = ProcessEntry(args.target_video, 0, 0, 0)
    total_frames = get_video_frame_total(args.target_video)
    if total_frames is None or total_frames < 1:
        print(f"Warning: Could not read total frames from video {args.target_video}")
        total_frames = 1
    process_entry.endframe = total_frames
    list_files_process.append(process_entry)

    print(f"Target set to: {args.target_video}")
    print(f"Target has {process_entry.endframe} frames.")


    print("Starting face swap process...")
    
    batch_process_regular(
        swap_model=roop.globals.face_swapper_model,
        output_method="File", # "File", "Virtual Camera"
        files=list_files_process,
        masking_engine=roop.globals.mask_engine,
        new_clip_text=roop.globals.clip_text,
        use_new_method=True,
        imagemask=None,
        restore_original_mouth=False,
        num_swap_steps=1,
        progress=None,
        selected_index=0
    )

    # Find the generated output file and rename it
    list_of_files = glob.glob(os.path.join(output_dir, '*.mp4'))
    if not list_of_files:
        print("Error: Processing finished, but no output file was created.")
        sys.exit(1)

    latest_file = max(list_of_files, key=os.path.getctime)
    os.rename(latest_file, args.output_file)

    print(f"Processing complete. Output saved to {args.output_file}")


if __name__ == "__main__":
    run()
