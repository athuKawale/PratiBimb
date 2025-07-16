
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

def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Swap multiple faces in an image.")
    parser.add_argument('-s', '--source-imgs', dest='source_imgs', required=True, nargs='+', help='Paths to the source images with faces.')
    parser.add_argument('-t', '--target-img', dest='target_img', required=True, help='Path to the target image.')
    parser.add_argument('-o', '--output-file', dest='output_file', required=True, help='Path for the output image file.')
    
    # Simplified options from the UI
    parser.add_argument('--swap-model', dest='swap_model', default='InSwapper 128', choices=["InSwapper 128", "ReSwapper 128", "ReSwapper 256"], help='The face swapping model to use.')
    parser.add_argument('--enhancer', dest='enhancer', default='None', choices=["None", "Codeformer", "DMDNet", "GFPGAN", "GPEN", "Restoreformer++"], help='Face enhancer to use.')
    parser.add_argument('--similarity-threshold', dest='similarity_threshold', type=float, default=0.65, help='Lower values mean more similar faces.')
    parser.add_argument('--blend-ratio', dest='blend_ratio', type=float, default=0.65, help='How much of the original face to blend in.')

    return parser.parse_args()

def run():
    """Main execution function."""
    args = get_args()

    # Basic validation
    for src_img in args.source_imgs:
        if not os.path.isfile(src_img):
            print(f"Error: Source image not found at {src_img}")
            sys.exit(1)
    if not os.path.isfile(args.target_img):
        print(f"Error: Target image not found at {args.target_img}")
        sys.exit(1)
    if util.is_video(args.target_img):
        print(f"Error: Target must be an image, not a video.")
        sys.exit(1)

    # Prepare environment and globals
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    prepare_environment()
    roop_globals.output_path = output_dir
    if roop_globals.CFG.clear_output:
        util.clean_dir(roop_globals.output_path)

    roop_globals.target_path = args.target_img
    roop_globals.selected_enhancer = args.enhancer
    roop_globals.distance_threshold = args.similarity_threshold
    roop_globals.blend_ratio = args.blend_ratio
    
    # Hardcoded globals for multi-face image swap
    roop_globals.face_swap_mode = "all_input" # Swap all detected faces
    roop_globals.no_face_action = 0 # Use untouched original frame
    roop_globals.autorotate_faces = True
    roop_globals.subsample_size = 128
    roop_globals.mask_engine = 'None'
    roop_globals.clip_text = None
    roop_globals.execution_providers=["CUDAExecutionProvider"]

    # Load source faces
    print("Analyzing source images...")
    for src_img in args.source_imgs:
        source_faces_data = extract_face_images(src_img, (False, 0))
        if not source_faces_data:
            print(f"Warning: No face detected in source image {src_img}. Skipping.")
            continue
        
        for face_data in source_faces_data:
            face_set = FaceSet()
            face = face_data[0]
            face.mask_offsets = (0,0,0,0,1,20) # Default mask offsets
            face_set.faces.append(face)
            roop_globals.INPUT_FACESETS.append(face_set)
    
    if not roop_globals.INPUT_FACESETS:
        print("Error: No faces were detected in any of the source images.")
        sys.exit(1)
    print(f"Found a total of {len(roop_globals.INPUT_FACESETS)} faces to swap.")

    # Prepare target file process entry
    list_files_process = [ProcessEntry(args.target_img, 0, 1, 0)]
    print(f"Target set to: {args.target_img}")

    # Set some more required globals
    roop_globals.execution_threads = roop_globals.CFG.max_threads
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
        selected_index=0
    )

    # Find the generated output file and rename it
    list_of_files = glob.glob(os.path.join(output_dir, '*.*'))
    if not list_of_files:
        print("Error: Processing finished, but no output file was created.")
        sys.exit(1)

    latest_file = max(list_of_files, key=os.path.getctime)
    os.rename(latest_file, args.output_file)

    print(f"Processing complete. Output saved to {args.output_file}")


if __name__ == "__main__":
    run()
