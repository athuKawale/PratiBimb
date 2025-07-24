
# Roop Globals Defaults

# --- Execution ---
execution_provider: str = 'CPUExecutionProvider' # ['CUDAExecutionProvider', 'CPUExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'CoreMLExecutionProvider']
execution_queue_count: int = 1
max_threads: int = 1
force_fp32: bool = False

# --- Source and Target ---
source_path: str = ''
target_path: str = ''
output_path: str = ''

# --- Face Selector ---
face_selector_mode: str = 'reference' # ['reference', 'many']
reference_face_position: int = 0
reference_face_distance: float = 0.6
similar_face_distance: float = 0.85

# --- Face Analyser ---
face_analyser_order: str = 'left-right' # ['left-right', 'right-left', 'top-bottom', 'bottom-top', 'small-large', 'large-small', 'best-worst', 'worst-best']
face_analyser_age: str = None # ['child', 'teen', 'adult', 'senior']
face_analyser_gender: str = None # ['male', 'female']
face_detector_model: str = 'retinaface_1.0' # ['retinaface_1.0', 'yunet_2023']
face_detector_size: str = '640x640'
face_detector_score: float = 0.5
face_landmarker_model: str = '3d_68' # ['2d_106', '2d_5', '3d_68']
face_landmarker_score: float = 0.5
face_recognizer_model: str = 'arcface_inswapper' # ['arcface_blendswap', 'arcface_ghost', 'arcface_inswapper', 'arcface_simswap']

# --- Face Mask ---
face_mask_types: list[str] = ['box'] # ['box', 'occlusion', 'region']
face_mask_blur: float = 0.3
face_mask_padding: tuple[int, int, int, int] = (0, 0, 0, 0)
face_mask_regions: list[str] = ['skin', 'face', 'hair'] # ['background', 'skin', 'face', 'hair', 'beard', 'ear', 'neck', 'cloth']
face_masker_model: str = 'clip2seg' # ['clip2seg', 'isnet', 'u2net']

# --- Frame Processors ---
frame_processors: list[str] = ['face_swapper'] # ['face_debugger', 'face_enhancer', 'face_swapper', 'frame_colorizer', 'frame_enhancer']
face_swapper_model: str = 'inswapper_128' # ['inswapper_128', 'inswapper_128_fp16', 'inswapper_256', 'simswap_256', 'simswap_512_unofficial', 'simswap_uniface_1.1']
face_enhancer_model: str = 'gfpgan_1.4' # ['codeformer', 'dmdnet', 'gfpgan_1.2', 'gfpgan_1.3', 'gfpgan_1.4', 'gpen_bfr_256', 'gpen_bfr_512', 'restoreformer_plus_plus']
face_enhancer_blend: int = 80
frame_enhancer_model: str = 'realesrgan_x2plus' # ['realesrgan_x2plus', 'realesrgan_x4plus', 'realesrnet_x4plus']
frame_enhancer_blend: int = 80

# --- Output ---
output_video_encoder: str = 'libx264' # ['libx264', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc']
output_video_preset: str = 'ultrafast' # ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow']
video_quality: int = 80
keep_fps: bool = True
keep_frames: bool = False
no_audio: bool = False

# --- Misc ---
skip_download: bool = False
headless: bool = True
log_level: str = 'info' # ['error', 'warn', 'info', 'debug']
skip_nsfw: bool = False
proxy: str = None
system_memory_limit: int = 0
