#!/usr/bin/env python3
"""
Professional Face Swapping Script using PratiBimb/roop

This script provides a clean, professional interface for face swapping between
source and target images using the roop face swapping framework.

Author: AI Assistant
Date: 2025-07-14
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Tuple
import cv2
import numpy as np

# Add the roop package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import roop.globals
    import roop.core
    import roop.utilities as util
    from roop.capturer import get_image_frame
    from roop.face_util import extract_face_images
    from roop.FaceSet import FaceSet
    from roop.ProcessOptions import ProcessOptions
except ImportError as e:
    print(f"Error importing roop modules: {e}")
    print("Please ensure the roop package is properly installed.")
    sys.exit(1)


class FaceSwapProcessor:
    """Professional face swapping processor using roop framework."""
    
    def __init__(self, 
                 execution_provider: str = "CPUExecutionProvider",
                 face_distance: float = 0.65,
                 blend_ratio: float = 0.5,
                 swap_mode: str = "first",
                 enhancer: Optional[str] = None,
                 swap_model: str = "inswapper_128.onnx"):
        """
        Initialize the face swap processor.
        
        Args:
            execution_provider: ONNX execution provider ("CPUExecutionProvider" or "CUDAExecutionProvider")
            face_distance: Face matching threshold (0.0-1.0, lower = stricter matching)
            blend_ratio: Face blending ratio (0.0-1.0, 0.0 = full replacement, 1.0 = no change)
            swap_mode: Face swap mode ("first", "all", "selected")
            enhancer: Face enhancement model ("GFPGAN", "Codeformer", None)
            swap_model: Face swap model to use
        """
        self.execution_provider = execution_provider
        self.face_distance = face_distance
        self.blend_ratio = blend_ratio
        self.swap_mode = swap_mode
        self.enhancer = enhancer
        self.swap_model = swap_model
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Initialize environment
        self._setup_environment()
    
    def _setup_environment(self) -> None:
        """Initialize the roop environment with configured settings."""
        self.logger.info("Setting up roop environment...")
        
        # Set execution providers
        if self.execution_provider == "CUDAExecutionProvider":
            roop.globals.execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            roop.globals.execution_providers = ["CPUExecutionProvider"]
        
        # Set processing options
        roop.globals.face_swap_mode = self.swap_mode
        roop.globals.blend_ratio = self.blend_ratio
        roop.globals.distance_threshold = self.face_distance
        roop.globals.selected_enhancer = self.enhancer
        
        # Download required models
        self.logger.info("Downloading required models (if not already present)...")
        if not roop.core.pre_check():
            raise RuntimeError("Failed to download required models")
        
        self.logger.info("Environment setup completed successfully")
    
    def _load_source_faces(self, source_image_path: str) -> FaceSet:
        """
        Load and extract faces from source image.
        
        Args:
            source_image_path: Path to source image
            
        Returns:
            FaceSet containing source faces
            
        Raises:
            ValueError: If no faces found in source image
            FileNotFoundError: If source image doesn't exist
        """
        if not os.path.exists(source_image_path):
            raise FileNotFoundError(f"Source image not found: {source_image_path}")
        
        self.logger.info(f"Extracting faces from source image: {source_image_path}")
        
        # Extract face data from source image
        face_data = extract_face_images(source_image_path, (False, 0))
        
        if not face_data:
            raise ValueError("No faces found in source image")
        
        # Create FaceSet with source faces
        faceset = FaceSet()
        for face_info in face_data:
            face_obj, face_image = face_info
            faceset.faces.append(face_obj)
            faceset.ref_images.append(face_image)
        
        # Average embeddings for better results if multiple faces
        if len(faceset.faces) > 1:
            faceset.AverageEmbeddings()
            self.logger.info(f"Averaged embeddings from {len(faceset.faces)} faces")
        
        self.logger.info(f"Successfully loaded {len(faceset.faces)} face(s) from source image")
        return faceset
    
    def _create_process_options(self) -> ProcessOptions:
        """
        Create ProcessOptions with configured settings.
        
        Returns:
            ProcessOptions object with current configuration
        """
        processors = {"faceswap": {}}
        
        # Add enhancer if selected
        if self.enhancer:
            if self.enhancer.lower() == "gfpgan":
                processors["gfpgan"] = {}
            elif self.enhancer.lower() == "codeformer":
                processors["codeformer"] = {}
        
        options = ProcessOptions(
            swap_model=self.swap_model,
            processordefines=processors,
            face_distance=self.face_distance,
            blend_ratio=self.blend_ratio,
            swap_mode=self.swap_mode,
            selected_index=0,  # Use first source face
            masking_text=None,
            imagemask=None,
            num_steps=1,
            subsample_size=128,
            show_face_area=False,
            restore_original_mouth=False
        )
        
        return options
    
    def swap_faces(self, source_image_path: str, target_image_path: str, 
                   output_path: str) -> np.ndarray:
        """
        Perform face swapping between source and target images.
        
        Args:
            source_image_path: Path to source image (face to be used)
            target_image_path: Path to target image (face to be replaced)
            output_path: Path where swapped image will be saved
            
        Returns:
            numpy.ndarray: Swapped image as BGR array
            
        Raises:
            FileNotFoundError: If source or target image doesn't exist
            ValueError: If no faces found in either image
            RuntimeError: If face swapping fails
        """
        # Validate inputs
        if not os.path.exists(source_image_path):
            raise FileNotFoundError(f"Source image not found: {source_image_path}")
        if not os.path.exists(target_image_path):
            raise FileNotFoundError(f"Target image not found: {target_image_path}")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.logger.info(f"Starting face swap process...")
        self.logger.info(f"Source: {source_image_path}")
        self.logger.info(f"Target: {target_image_path}")
        self.logger.info(f"Output: {output_path}")
        
        try:
            # Load source faces
            source_faceset = self._load_source_faces(source_image_path)
            roop.globals.INPUT_FACESETS = [source_faceset]
            
            # Load target image
            self.logger.info(f"Loading target image: {target_image_path}")
            target_frame = get_image_frame(target_image_path)
            if target_frame is None:
                raise RuntimeError("Failed to load target image")
            
            # Extract target faces for face matching
            target_face_data = extract_face_images(target_image_path, (False, 0))
            if not target_face_data:
                raise ValueError("No faces found in target image")
            
            roop.globals.TARGET_FACES = [face_info[0] for face_info in target_face_data]
            self.logger.info(f"Found {len(target_face_data)} face(s) in target image")
            
            # Create process options
            options = self._create_process_options()
            
            # Perform face swap
            self.logger.info("Performing face swap...")
            result_frame = roop.core.live_swap(target_frame, options)
            
            if result_frame is None:
                raise RuntimeError("Face swap failed - no result returned")
            
            # Save result
            self.logger.info(f"Saving result to: {output_path}")
            success = cv2.imwrite(output_path, result_frame)
            if not success:
                raise RuntimeError(f"Failed to save result image to {output_path}")
            
            self.logger.info("Face swap completed successfully!")
            return result_frame
            
        except Exception as e:
            self.logger.error(f"Face swap failed: {e}")
            raise


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Professional Face Swapping Script using PratiBimb/roop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python face_swap_script.py -s source.jpg -t target.jpg -o output.jpg
  python face_swap_script.py -s source.jpg -t target.jpg -o output.jpg --enhancer GFPGAN
  python face_swap_script.py -s source.jpg -t target.jpg -o output.jpg --gpu --blend-ratio 0.7
        """
    )
    
    parser.add_argument("-s", "--source", required=True, 
                       help="Path to source image (face to be used)")
    parser.add_argument("-t", "--target", required=True,
                       help="Path to target image (face to be replaced)")
    parser.add_argument("-o", "--output", required=True,
                       help="Path for output image")
    parser.add_argument("--face-distance", type=float, default=0.65,
                       help="Face matching threshold (0.0-1.0, default: 0.65)")
    parser.add_argument("--blend-ratio", type=float, default=0.5,
                       help="Face blending ratio (0.0-1.0, default: 0.5)")
    parser.add_argument("--swap-mode", choices=["first", "all", "selected"], default="first",
                       help="Face swap mode (default: first)")
    parser.add_argument("--enhancer", choices=["GFPGAN", "Codeformer"],
                       help="Face enhancement model to use")
    parser.add_argument("--gpu", action="store_true",
                       help="Use GPU acceleration (CUDA)")
    parser.add_argument("--swap-model", default="inswapper_128.onnx",
                       help="Face swap model to use (default: inswapper_128.onnx)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0.0 <= args.face_distance <= 1.0:
        parser.error("face-distance must be between 0.0 and 1.0")
    if not 0.0 <= args.blend_ratio <= 1.0:
        parser.error("blend-ratio must be between 0.0 and 1.0")
    
    # Determine execution provider
    execution_provider = "CUDAExecutionProvider" if args.gpu else "CPUExecutionProvider"
    
    try:
        # Initialize processor
        processor = FaceSwapProcessor(
            execution_provider=execution_provider,
            face_distance=args.face_distance,
            blend_ratio=args.blend_ratio,
            swap_mode=args.swap_mode,
            enhancer=args.enhancer,
            swap_model=args.swap_model
        )
        
        # Perform face swap
        result = processor.swap_faces(args.source, args.target, args.output)
        
        print(f"✅ Face swap completed successfully!")
        print(f"📁 Output saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()