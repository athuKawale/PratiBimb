
import argparse
import os
import sys
import glob

# Add project root to path to allow relative imports
sys.path.append(os.getcwd())

from roop import globals as roop_globals
from roop.core import batch_process_regular
from roop.face_util import extract_face_images
from roop.ProcessEntry import ProcessEntry
from roop.FaceSet import FaceSet
from roop import utilities as util
from prepare_env import prepare_environment
from roop.capturer import get_video_frame_total

def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Face swap in a video.")
    parser.add_argument('-s', '--source-img', dest='source_img', required=True, help='Path to the source image with the face.')
    parser.add_argument('-t', '--target-video', dest='target_video', required=True, help='Path to the target video file.')
    parser.add_argument('-o', '--output-file', dest='output_file', required=True, help='Path for the output video file.')
    
    # Simplified options from the UI
    parser.add_argument('--swap-model', dest='swap_model', default='InSwapper 128', choices=["InSwapper 128", "ReSwapper 128", "ReSwapper 256"], help='The face swapping model to use.')
    parser.add_argument('--enhancer', dest='enhancer', default='None', choices=["None", "Codeformer", "DMDNet", "GFPGAN", "GPEN", "Restoreformer++"], help='Face enhancer to use.')
    parser.add_argument('--similarity-threshold', dest='similarity_threshold', type=float, default=0.65, help='Lower values mean more similar faces.')
    parser.add_argument('--blend-ratio', dest='blend_ratio', type=float, default=0.65, help='How much of the original face to blend in.')
    parser.add_argument('--skip-audio', dest='skip_audio', action='store_true', help='Skip audio processing for videos.')

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
    
    prepare_environment()
    roop_globals.output_path = output_dir
    if roop_globals.CFG.clear_output:
        util.clean_dir(roop_globals.output_path)

    roop_globals.source_path = args.source_img
    roop_globals.target_path = args.target_video
    roop_globals.skip_audio = args.skip_audio
    roop_globals.selected_enhancer = args.enhancer
    roop_globals.distance_threshold = args.similarity_threshold
    roop_globals.blend_ratio = args.blend_ratio
    
    # Hardcoded globals for video swap
    roop_globals.face_swap_mode = "first" # Swap the first detected face
    roop_globals.no_face_action = 0 # Use untouched original frame
    roop_globals.keep_frames = False
    roop_globals.wait_after_extraction = False
    roop_globals.vr_mode = False
    roop_globals.autorotate_faces = True
    roop_globals.subsample_size = 128
    roop_globals.execution_providers=["CUDAExecutionProvider"]
    roop_globals.mask_engine = 'None'
    roop_globals.clip_text = None

    # Load source face
    print("Analyzing source image...")
    source_faces_data = extract_face_images(args.source_img, (False, 0))
    if not source_faces_data:
        print("Error: No face detected in the source image.")
        sys.exit(1)
    
    face_set = FaceSet()
    face = source_faces_data[0][0]
    face.mask_offsets = (0,0,0,0,1,20) # Default mask offsets
    face_set.faces.append(face)
    roop_globals.INPUT_FACESETS.append(face_set)
    print(f"Found {len(source_faces_data)} face(s), using the first one.")

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

    # Set some more required globals
    roop_globals.execution_threads = roop_globals.CFG.max_threads
    roop_globals.video_encoder = roop_globals.CFG.output_video_codec
    roop_globals.video_quality = roop_globals.CFG.video_quality
    roop_globals.max_memory = roop_globals.CFG.memory_limit if roop_globals.CFG.memory_limit > 0 else None

    print("Starting face swap process...")
    
    batch_process_regular(
        swap_model=args.swap_model,
        output_method="File",
        files=list_files_process,
        masking_engine=roop_globals.mask_engine,
        new_clip_text=roop_globals.clip_text,
        use_new_method=True,
        imagemask=None,
        restore_original_mouth=False,
        num_swap_steps=1,
        progress=None,
        selected_input_face_index=0
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
