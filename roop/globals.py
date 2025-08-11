import os
import gc
import sys
import torch
import shutil
import pathlib
import platform
import onnxruntime
from time import time
from typing import List
import roop.utilities as util
import roop.util_ffmpeg as ffmpeg
from roop.ProcessMgr import ProcessMgr
from roop.ProcessEntry import ProcessEntry
from roop.ProcessOptions import ProcessOptions
from roop.capturer import get_video_frame_total
from roop.core import limit_resources, release_resources


# Server settings
BASE_URL = "http://localhost:8002"
DATA_FILE = "data.json"

class GLOBALS :

    def __init__(self):
        
        self.processing = False 
        
        # Paths
        self.source_path: str = ''
        self.target_path: str = ''
        self.output_path: str = ''
        self.output_template = "{file}_{time}"

        # Face processing
        self.INPUT_FACESETS = []
        self.TARGET_FACES = []
        self.VIDEO_INPUTFACES : List = []
        self.TEMP_FACESET : List = []

        # Face swap settings
        self.face_swap_mode = 'all_input'  # 'first', 'all_input', 'all_female', 'all_male', 'all_random', 'all', 'selected'
        self.face_swapper_model: str = 'InSwapper 128'  # ['InSwapper 128', 'ReSwapper 128', 'ReSwapper 256']
        self.blend_ratio = 1
        self.distance_threshold: float = 0.80
        self.no_face_action = 0

        # Enhancement settings
        self.selected_enhancer = 'GPEN'  # 'GFPGAN', 'Codeformer', None, 'DMDNet', 'Restoreformer++', 'GPEN'
        self.subsample_size = 128
        self.autorotate_faces = True

        # Video encoding settings
        self.video_encoder = "libx264"
        self.output_video_format = "mp4"
        self.output_video_codec = "libx264"
        self.video_quality = 14
        self.output_image_format = "png"
        self.output_show_video = True

        # Performance & resource settings
        self.max_memory = None
        self.memory_limit = 0
        self.execution_providers: List[str] = ['CPUExecutionProvider']
        self.execution_threads = 4

        # Miscellaneous processing options
        self.mask_engine : str = 'None'
        self.clip_text : str = None
        self.keep_frames = False
        self.vr_mode = False
        self.skip_audio = False
        self.wait_after_extraction = False
        self.clear_output = True
        self.log_level = 'error'
        self.process_mgr = None

        # Globals for face analysis (used internally)
        self.g_current_face_analysis = None
        self.g_desired_face_analysis = None

        if 'ROCMExecutionProvider' in self.execution_providers:
            del torch

    def encode_execution_providers(execution_providers: List[str]) -> List[str]:
        return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]

    def decode_execution_providers(self, execution_providers: List[str]) -> List[str]:
        list_providers = [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), self.encode_execution_providers(onnxruntime.get_available_providers()))
                if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]
        
        try:
            for i in range(len(list_providers)):
                if list_providers[i] == 'CUDAExecutionProvider':
                    list_providers[i] = ('CUDAExecutionProvider', {'device_id': self.cuda_device_id})
                    torch.cuda.set_device(self.cuda_device_id)
                    break
        except:
            pass

        return list_providers

    def suggest_max_memory() -> int:
        if platform.system().lower() == 'darwin':
            return 4
        return 16

    def suggest_execution_providers(self) -> List[str]:
        return self.encode_execution_providers(onnxruntime.get_available_providers())

    def suggest_execution_threads(self) -> int:
        if 'DmlExecutionProvider' in self.execution_providers:
            return 1
        if 'ROCMExecutionProvider' in self.execution_providers:
            return 1
        return 8

    def limit_resources(self) -> None:
        # limit memory usage
        if self.max_memory:
            memory = self.max_memory * 1024 ** 3
            if platform.system().lower() == 'darwin':
                memory = self.max_memory * 1024 ** 6
            if platform.system().lower() == 'windows':
                import ctypes
                kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
                kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
            else:
                import resource
                resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))

    def release_resources() -> None:
        global process_mgr

        if process_mgr is not None:
            process_mgr.release_resources()
            process_mgr = None

        gc.collect()

    def pre_check(self) -> bool:
        if sys.version_info < (3, 9):
            update_status('Python version is not supported - please upgrade to 3.9 or higher.')
            return False
        
        download_directory_path = util.resolve_relative_path('../models')
        util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx'])
        util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/reswapper_128.onnx'])
        util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/reswapper_256.onnx'])
        util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/GFPGANv1.4.onnx'])
        util.conditional_download(download_directory_path, ['https://github.com/csxmli2016/DMDNet/releases/download/v1/DMDNet.pth'])
        util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/GPEN-BFR-512.onnx'])
        util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/restoreformer_plus_plus.onnx'])
        util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/xseg.onnx'])
        download_directory_path = util.resolve_relative_path('../models/CLIP')
        util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/rd64-uni-refined.pth'])
        download_directory_path = util.resolve_relative_path('../models/CodeFormer')
        util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/CodeFormerv0.1.onnx'])
        download_directory_path = util.resolve_relative_path('../models/Frame')
        util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/deoldify_artistic.onnx'])
        util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/deoldify_stable.onnx'])
        util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/isnet-general-use.onnx'])
        util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/real_esrgan_x4.onnx'])
        util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/real_esrgan_x2.onnx'])
        util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/lsdir_x4.onnx'])

        if not shutil.which('ffmpeg'):
            self.update_status('ffmpeg is not installed.')

        return True

    def update_status(message: str) -> None:
        print(message)

    def get_processing_plugins(self, masking_engine):
        processors = {  "faceswap": {}}
        if masking_engine is not None:
            processors.update({masking_engine: {}})
        
        if self.selected_enhancer == 'GFPGAN':
            processors.update({"gfpgan": {}})
        elif self.selected_enhancer == 'Codeformer':
            processors.update({"codeformer": {}})
        elif self.selected_enhancer == 'DMDNet':
            processors.update({"dmdnet": {}})
        elif self.selected_enhancer == 'GPEN':
            processors.update({"gpen": {}})
        elif self.selected_enhancer == 'Restoreformer++':
            processors.update({"restoreformer++": {}})
        return processors

    def live_swap(self, frame, options):
        global process_mgr

        if frame is None:
            return frame

        if process_mgr is None:
            process_mgr = ProcessMgr(None)

        process_mgr.initialize(self.INPUT_FACESETS, self.TARGET_FACES, options)
        newframe = process_mgr.process_frame(frame)
        if newframe is None:
            return frame
        return newframe

    def batch_process_regular(self, swap_model, output_method, files:list[ProcessEntry], masking_engine:str, new_clip_text:str, use_new_method, imagemask, restore_original_mouth, num_swap_steps, progress, selected_index = 0) -> None:

        release_resources()
        limit_resources()
        if process_mgr is None:
            process_mgr = ProcessMgr(progress)
        mask = imagemask["layers"][0] if imagemask is not None else None
        if len(self.INPUT_FACESETS) <= selected_index:
            selected_index = 0
        options = ProcessOptions(swap_model, self.get_processing_plugins(masking_engine), self.distance_threshold, self.blend_ratio,
                                self.face_swap_mode, selected_index, new_clip_text, mask, num_swap_steps,
                                self.subsample_size, False, restore_original_mouth)
        process_mgr.initialize(self, self.INPUT_FACESETS, self.TARGET_FACES, options)
        self.batch_process(output_method, files, use_new_method)
        return

    def batch_process_with_options(self, files:list[ProcessEntry], options, progress):
        release_resources()
        limit_resources()
        if process_mgr is None:
            process_mgr = ProcessMgr(progress)
        process_mgr.initialize(self, self.INPUT_FACESETS, self.TARGET_FACES, options)
        self.batch_process("Files", files, True)

    def batch_process(self, output_method, files:list[ProcessEntry], use_new_method) -> None:

        self.processing = True

        # limit threads for some providers
        max_threads = self.suggest_execution_threads()
        if max_threads == 1:
            self.execution_threads = 1

        imagefiles:list[ProcessEntry] = []
        videofiles:list[ProcessEntry] = []
            
        self.update_status('Sorting videos/images')


        for index, f in enumerate(files):
            fullname = f.filename
            if util.has_image_extension(fullname):
                destination = util.get_destfilename_from_path(fullname, self.output_path, f'.{self.output_image_format}')
                destination = util.replace_template(destination,globals=self, index=index)
                pathlib.Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)
                f.finalname = destination
                imagefiles.append(f)

            elif util.is_video(fullname) or util.has_extension(fullname, ['gif']):
                destination = util.get_destfilename_from_path(fullname, self.output_path, f'__temp.{self.output_video_format}')
                f.finalname = destination
                videofiles.append(f)



        if(len(imagefiles) > 0):
            self.update_status('Processing image(s)')
            origimages = []
            fakeimages = []
            for f in imagefiles:
                origimages.append(f.filename)
                fakeimages.append(f.finalname)

            process_mgr.run_batch(origimages, fakeimages, self.execution_threads)
            origimages.clear()
            fakeimages.clear()

        if(len(videofiles) > 0):
            for index,v in enumerate(videofiles):
                if not self.processing:
                    self.end_processing('Processing stopped!')
                    return
                fps = v.fps if v.fps > 0 else util.detect_fps(v.filename)
                if v.endframe == 0:
                    v.endframe = get_video_frame_total(v.filename)

                is_streaming_only = output_method == "Virtual Camera"
                if is_streaming_only == False:
                    self.update_status(f'Creating {os.path.basename(v.finalname)} with {fps} FPS...')

                start_processing = time()
                if is_streaming_only == False and self.keep_frames or not use_new_method:
                    util.create_temp(v.filename)
                    self.update_status('Extracting frames...')
                    ffmpeg.extract_frames(self, v.filename,v.startframe,v.endframe, fps)
                    if not self.processing:
                        self.end_processing('Processing stopped!')
                        return

                    temp_frame_paths = util.get_temp_frame_paths(self, v.filename)
                    process_mgr.run_batch(temp_frame_paths, temp_frame_paths, self.execution_threads)
                    if not self.processing:
                        self.end_processing('Processing stopped!')
                        return
                    if self.wait_after_extraction:
                        extract_path = os.path.dirname(temp_frame_paths[0])
                        util.open_folder(extract_path)
                        input("Press any key to continue...")
                        print("Resorting frames to create video")
                        util.sort_rename_frames(self, extract_path)                                    
                    
                    ffmpeg.create_video(self, v.filename, v.finalname, fps)
                    if not self.keep_frames:
                        util.delete_temp_frames(temp_frame_paths[0])
                else:
                    if util.has_extension(v.filename, ['gif']):
                        skip_audio = True
                    else:
                        skip_audio = self.skip_audio
                    process_mgr.run_batch_inmem(output_method, v.filename, v.finalname, v.startframe, v.endframe, fps,self.execution_threads)
                    
                if not self.processing:
                    self.end_processing('Processing stopped!')
                    return
                
                video_file_name = v.finalname
                if os.path.isfile(video_file_name):
                    destination = ''
                    if util.has_extension(v.filename, ['gif']):
                        gifname = util.get_destfilename_from_path(v.filename, self.output_path, '.gif')
                        destination = util.replace_template(gifname, index=index)
                        pathlib.Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)

                        self.update_status('Creating final GIF')
                        ffmpeg.create_gif_from_video(video_file_name, destination)
                        if os.path.isfile(destination):
                            os.remove(video_file_name)
                    else:
                        skip_audio = self.skip_audio
                        destination = util.replace_template(video_file_name, index=index)
                        pathlib.Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)

                        if not skip_audio:
                            ffmpeg.restore_audio(video_file_name, v.filename, v.startframe, v.endframe, destination)
                            if os.path.isfile(destination):
                                os.remove(video_file_name)
                        else:
                            shutil.move(video_file_name, destination)

                elif is_streaming_only == False:
                    self.update_status(f'Failed processing {os.path.basename(v.finalname)}!')
                elapsed_time = time() - start_processing
                average_fps = (v.endframe - v.startframe) / elapsed_time
                self.update_status(f'\nProcessing {os.path.basename(destination)} took {elapsed_time:.2f} secs, {average_fps:.2f} frames/s')
        
        self.end_processing('Finished')

    def end_processing(self, msg:str):
        self.update_status(msg)
        release_resources()

    def destroy(self) -> None:
        if self.target_path:
            util.clean_temp(self.target_path)
        release_resources()        
        sys.exit()


