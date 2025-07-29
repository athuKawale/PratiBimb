# Setup
```bash
conda create -n pratibimb python=3.11 -y

conda activate pratibimb

sh build.sh
```

# Execution

```bash
python main.py
```


# Cobebase overview : 

---

## Tools from `roop.utilities` (Our Helper Elves)

These are little helper tools that do useful jobs for us.

### 🧹 `clean_dir(path)`
- **What it does:** It's like a magic broom! You give it a folder, and it sweeps away all the files inside, making it clean and empty.
- **What you give it:** The address of the folder you want to clean.
- **What you get back:** Nothing! It just does the cleaning.

### 📂 `get_local_files_from_folder(folder)`
- **What it does:** Imagine you have a toy box. This tool looks inside and gives you a list of all the toys (files) in it.
- **What you give it:** The name of the toy box (the folder).
- **What you get back:** A list with the names of all the files.

### 🕵️ `has_image_extension(path)`
- **What it does:** This is a detective that checks if a file is a picture. It looks at the end of the file's name (like `.jpg` or `.png`) to be sure.
- **What you give it:** The file's name.
- **What you get back:** `True` (yes, it's a picture!) or `False` (no, it's something else).

### 🖼️ `is_image(path)` and 🎬 `is_video(path)`
- **What they do:** Just like the detective above, these tools check if a file is a picture or a video.
- **What you give them:** The file's name.
- **What you get back:** `True` or `False`.

### ✨ `convert_to_gradio(image)`
- **What it does:** This tool takes a picture and gets it ready to be shown on the screen in our app. It's like putting a pretty frame around your drawing.
- **What you give it:** A picture.
- **What you get back:** The picture, all ready to be displayed.

### 🚪 `open_folder(path)`
- **What it does:** This is like a magic key that opens a folder on your computer so you can see what's inside.
- **What you give it:** The folder's address.
- **What you get back:** Nothing! It just opens the folder for you.

### ⏱️ `detect_fps(filename)`
- **What it does:** A video is made of many pictures shown very fast. This tool counts how many pictures (frames) are in each second of the video.
- **What you give it:** The name of the video file.
- **What you get back:** A number, telling you the frames-per-second (FPS).

---

## Tools from `roop.face_util` (The Face Experts)

These tools are experts at finding and working with faces.

### 🧐 `extract_face_images(path, options)`
- **What it does:** This is a super smart face finder! You give it a picture or a video, and it finds all the faces in it. It carefully cuts out each face and gives it to you as a small picture.
- **What you give it:** The address of the picture/video and some settings.
- **What you get back:** A list of all the faces it found, along with a small picture of each one.

### 📄 `create_blank_image(width, height)`
- **What it does:** This is like getting a fresh, blank piece of paper to draw on. You tell it how big you want the paper to be.
- **What you give it:** How wide (`width`) and tall (`height`) the paper should be.
- **What you get back:** A new, empty picture.

---

## Tools from `roop.capturer` (The Picture Grabbers)

These tools are great at grabbing pictures from files.

### 🎞️ `get_video_frame(path, frame_number)`
- **What it does:** A video is like a flipbook with many pages (pictures). This tool lets you grab just one specific page (a "frame") from the video.
- **What you give it:** The video's name and the page number you want.
- **What you get back:** The picture from that exact spot in the video.

### 🔢 `get_video_frame_total(path)`
- **What it does:** This counts all the pictures (frames) in a whole video from start to finish.
- **What you give it:** The video's name.
- **What you get back:** The total number of pictures in the video.

### 🏞️ `get_image_frame(path)`
- **What it does:** This tool simply gets a picture from a file. It's like opening a photo album to look at a photo.
- **What you give it:** The picture's file name.
- **What you get back:** The picture itself.

---

## Tools from `roop.core` (The Magic Show)

These are the main tools that perform the face-swapping magic!

### 🎭 `live_swap(frame, options)`
- **What it does:** This is the star of the show! It takes a picture (`frame`) and swaps the face in it with a different face. The `options` are like the magic words that tell it exactly how to do the swap.
- **What you give it:** A picture and a list of rules for the swap.
- **What you get back:** A new picture with the face swapped!

### 🏭 `batch_process_regular(...)`
- **What it does:** This is for when you have a lot of work to do. Instead of swapping one face at a time, it does the face swap for a whole video or a big pile of pictures all at once. It's like a factory for face swapping!
- **What you give it:** A list of files to work on, the faces to use, and other rules.
- **What you get back:** It doesn't give anything back directly, but it saves all the finished pictures and videos in the output folder.

---

## The Blueprints (Classes)

These are not tools, but more like blueprints for making special containers to hold our things.

### 📦 `FaceSet()`
- **What it is:** A special box to hold one or more faces that belong to the same person. This helps keep everything organized.

### 📝 `ProcessEntry()`
- **What it is:** A little note that keeps track of a file we want to work on, like a video. It remembers where to start and stop.

### 📜 `ProcessOptions()`
- **What it is:** A list of all the rules and settings for how we want to do the face swap. It's like a recipe for the magic trick.
