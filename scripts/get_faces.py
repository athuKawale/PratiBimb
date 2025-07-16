
import argparse
import os
import sys
import cv2

sys.path.append(os.getcwd())

from roop.face_util import extract_face_images
from roop.utilities import has_image_extension

def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Detect and extract faces from an image.")
    parser.add_argument('-s', '--source', dest='source_path', required=True, help='Path to the source image.')
    parser.add_argument('-o', '--output-dir', dest='output_dir', default='output_faces', help='Directory to save the extracted faces.')
    return parser.parse_args()

def run():
    """Main execution function."""
    args = get_args()

    if not os.path.isfile(args.source_path) or not has_image_extension(args.source_path):
        print(f"Error: Invalid source image file: {args.source_path}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Detecting faces in {args.source_path}...")
    
    # The second argument to extract_face_images is a tuple for video processing options,
    # which we can set to (False, 0) for single image processing.
    extracted_faces = extract_face_images(args.source_path, (False, 0))

    if not extracted_faces:
        print("No faces detected in the source image.")
        return

    print(f"Found {len(extracted_faces)} face(s). Saving them to {args.output_dir}...")

    for i, face_data in enumerate(extracted_faces):
        face_image = face_data[0].face_image
        if face_image is not None:
            output_path = os.path.join(args.output_dir, f"face_{i+1}.png")
            try:
                # The image is in BGR format, so we convert it to RGB before saving
                cv2.imwrite(output_path, cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
                print(f"Saved face {i+1} to {output_path}")
            except Exception as e:
                print(f"Error saving face {i+1}: {e}")

if __name__ == "__main__":
    run()
