from settings import Settings
from typing import List

source_path: str = ''
source_path_video: List[str] = []
target_path: str = ''
output_path: str = ''
target_folder_path = None
startup_args = None

mask_engine : str = 'None' 
clip_text : str = 'None'
cuda_device_id = 0
frame_processors: List[str] = []
keep_frames = False
autorotate_faces = True
vr_mode = None
skip_audio = None
wait_after_extraction = None
source_face_index = 0
target_face_index = 0
video_encoder = None
video_quality = None
max_memory = None
execution_providers: List[str] = ['CPUExecutionProvider']
force_fp32: bool = False
execution_queue_count: int = 1
execution_threads = 4
headless = None
log_level = 'error'
selected_enhancer = 'GPEN' # 'GFPGAN', 'Codeformer', None, 'DMDNet', 'Restoreformer++', 'GPEN'
subsample_size = 128
face_swap_mode = 'all_input' # 'first', 'all_input', 'all_female', 'all_male', 'all_random', 'all', 'selected'
face_swapper_model: str = 'InSwapper 128' # ['InSwapper 128', 'ReSwapper 128', 'ReSwapper 256']
blend_ratio = 1
distance_threshold: float = 0.80
default_det_size = True
no_face_action = 0

processing = False

g_current_face_analysis = None
g_desired_face_analysis = None

INPUT_FACESETS = []
TARGET_FACES = []
VIDEO_INPUTFACES : List = []
TEMP_FACESET : List = []


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
CFG.max_threads = 4
CFG.memory_limit = 0
CFG.provider = "cuda"
CFG.force_cpu = False
CFG.output_template = "{file}_{time}"
CFG.use_os_temp_folder = False
CFG.output_show_video = True
CFG.launch_browser = True

BASE_URL = "http://localhost:8002"
DATA_FILE = "data.json"
