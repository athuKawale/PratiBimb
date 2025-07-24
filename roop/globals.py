from settings import Settings
from typing import List

source_path = None
target_path = None
output_path = None
target_folder_path = None
startup_args = None

cuda_device_id = 0
frame_processors: List[str] = []
keep_fps = None
keep_frames = None
autorotate_faces = None
vr_mode = None
skip_audio = None
wait_after_extraction = None
many_faces = None
use_batch = None
source_face_index = 0
target_face_index = 0
face_position = None
video_encoder = None
video_quality = None
max_memory = None
execution_providers: List[str] = []
execution_threads = None
headless = None
log_level = 'error'
selected_enhancer = None
subsample_size = 128
face_swap_mode = None
blend_ratio = 0.5
distance_threshold = 0.65
default_det_size = True

no_face_action = 0

processing = False

g_current_face_analysis = None
g_desired_face_analysis = None

FACE_ENHANCER = None

INPUT_FACESETS = []
TARGET_FACES = []


IMAGE_CHAIN_PROCESSOR = None
VIDEO_CHAIN_PROCESSOR = None
BATCH_IMAGE_CHAIN_PROCESSOR = None

# Hard-coded Settings instance
CFG: Settings = Settings.__new__(Settings)
CFG.selected_theme = "Default"
CFG.server_name = ""
CFG.server_port = 0
CFG.server_share = False
CFG.output_image_format = "png"
CFG.output_video_format = "mp4"
CFG.output_video_codec = "libx264"
CFG.video_quality = 14
CFG.clear_output = True
CFG.max_threads = 2
CFG.memory_limit = 0
CFG.provider = "cuda"
CFG.force_cpu = False
CFG.output_template = "{file}_{time}"
CFG.use_os_temp_folder = False
CFG.output_show_video = True
CFG.launch_browser = True
# Add any other config values as needed


