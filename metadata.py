title = 'PratiBimb'

version = '7.8.0'

description = "Fast API app to perform image faceswap single and multiple faces and videofaceswap for single and multiple faces."

os_info = "Ubuntu 24.04"

cuda_toolkit_version = "12.4"

models = {
    "buffalo_l" : {
        # Face recognition and detection models by InsightFace; used for face detection, gender and age estimation, and face embedding extraction.
        "1k3d68.onnx" : "models/buffalo_l/1k3d68.onnx",
        "2d106det.onnx" : "models/buffalo_l/2d106det.onnx",
        "det_10g.onnx" : "models/buffalo_l/det_10g.onnx",
        "genderage.onnx": "models/buffalo_l/genderage.onnx",
        "w600k_r50.onnx" : "models/buffalo_l/w600k_r50.onnx"
    },
    "CLIP": {
        # CLIP: Connects images and text, enabling zero-shot image classification and text-image matching.
        "rd64-uni-refined": "models/CLIP/rd64-uni-refined.pth"
    },
    "CodeFormer": {
        # CodeFormer: High-quality face restoration and enhancement, effective on both real and AI-generated faces.
        "CodeFormerv0.1": "models/CodeFormer/CodeFormerv0.1.onnx"
    },
    "Frame": {
        # DeOldify Artistic: Colorizes and restores old grayscale images with artistic vibrancy.
        "deoldify_artistic": "models/Frame/deoldify_artistic.onnx",
        # DeOldify Stable: More stable colorization for grayscale images.
        "deoldify_stable": "models/Frame/deoldify_stable.onnx",
        # ISNet General Use: Accurate foreground segmentation and image mask generation.
        "isnet-general-use": "models/Frame/isnet-general-use.onnx",
        # Real-ESRGAN x4: Real-world image super-resolution with 4x upscale.
        "real_esrgan_x4": "models/Frame/real_esrgan_x4.onnx",
        # Real-ESRGAN x2: Real-world image super-resolution with 2x upscale.
        "real_esrgan_x2": "models/Frame/real_esrgan_x2.onnx",
        # LSDir x4: Lightweight super-resolution model with 4x upscale.
        "lsdir_x4": "models/Frame/lsdir_x4.onnx"
    },
    # GFPGAN v1.4: Generative Facial Prior GAN for blind face restoration in damaged or low-res photos.
    "GFPGANv1.4": "models/GFPGANv1.4.onnx",
    # GPEN-BFR-512: Blind Face Restoration model optimized for enhancing low-quality face images.
    "GPEN-BFR-512": "models/GPEN-BFR-512.onnx",
    # DMDNet: Deep Multi-scale Discriminator Network, used for high-quality image restoration and enhancement.
    "DMDNet": "models/DMDNet.pth",
    # Inswapper 128: Face swapping model at 128x128 resolution.
    "inswapper_128": "models/inswapper_128.onnx",
    # Reswapper 128: Face re-swapping model at 128x128 resolution.
    "reswapper_128": "models/reswapper_128.onnx",
    # Reswapper 256: Face re-swapping at higher 256x256 resolution.
    "reswapper_256": "models/reswapper_256.onnx",
    # RestoreFormer Plus Plus: Advanced restoration model for improving low-quality images.
    "restoreformer_plus_plus": "models/restoreformer_plus_plus.onnx",
    # XSeg: Segmentation model often used for facial components segmentation and mask extraction.
    "xseg": "models/xseg.onnx"
}
