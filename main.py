from lib_isa import LibIsa
from moviepy.editor import *

# Video settings
fps = 30
width = 1920
height = 1080
duration_per_image = 1.5

def concatenate(video_clip_paths, output_path, method="compose"):
    clips = [VideoFileClip(c) for c in video_clip_paths]
    if method == "reduce":
        min_height = min([c.h for c in clips])
        min_width = min([c.w for c in clips])
        clips = [c.resize(newsize=(min_width, min_height)) for c in clips]
        final_clip = concatenate_videoclips(clips)
    elif method == "compose":
        final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(output_path)

if __name__ == '__main__':
    os.makedirs('output', exist_ok=True)
    input_folders = ['input/1', 'input/2', 'input/3', 'input/4', 'input/5', 'input/6']
    video_clip_paths = [f"output/{input_folder.split('/')[-1]}.mp4" for input_folder in input_folders]

    for i, input_folder in enumerate(input_folders):
        LibIsa.convert_images_to_video(input_folder, video_clip_paths[i], fps, width, height, duration_per_image)
        break

    # concatenate(video_clip_paths, "output/concat.mp4", method="compose")