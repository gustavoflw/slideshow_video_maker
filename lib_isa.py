import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class LibIsa:
    @staticmethod
    def get_image_paths(folder, file_types):
        image_paths = [os.path.join(folder, file) for file in os.listdir(folder) if file.split('.')[-1] in file_types]
        image_paths.sort(key=os.path.getctime)
        print(f"len(image_paths): {len(image_paths)}")
        return image_paths

    @staticmethod
    def get_non_image_paths(folder, file_types):
        non_image_paths = [os.path.join(folder, file) for file in os.listdir(folder) if file.split('.')[-1] not in file_types]
        print(f"len(non_image_paths): {len(non_image_paths)}")
        [print(f"  {non_image_path}") for non_image_path in non_image_paths]
        return non_image_paths

    @staticmethod
    def get_duration_per_image(duration, image_paths):
        duration_per_image = duration / len(image_paths)
        print(f"duration_per_image: {duration_per_image}")
        return duration_per_image

    @staticmethod
    def get_frames_per_image(duration_per_image, fps):
        frames_per_image = max(int(duration_per_image * fps), 1)
        print(f"frames_per_image: {frames_per_image}")
        return frames_per_image

    @staticmethod
    def get_max_shape(frames):
        max_width = max([frame.shape[1] for frame in frames])
        max_height = max([frame.shape[0] for frame in frames])
        max_shape = (max_width, max_height)
        print(f"max_shape = ({max_shape})")
        return max_shape

    @staticmethod
    def get_unique_frames(file_paths, width, height):
        frames = []
        for image_path in file_paths:
            frame = cv2.imread(image_path)

            # Resize the image while maintaining its aspect ratio
            ratio = min(width / frame.shape[1], height / frame.shape[0])
            new_width = int(frame.shape[1] * ratio)
            new_height = int(frame.shape[0] * ratio)
            resized_frame = cv2.resize(frame, (new_width, new_height))
            
            # Calculate the position to center the image within the frame
            x_offset = (width - new_width) // 2
            y_offset = (height - new_height) // 2

            black_frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Place the resized image in the center of the black frame
            black_frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_frame
            
            # plt.imshow(cv2.cvtColor(black_frame, cv2.COLOR_BGR2RGB))
            # plt.show()

            frames.append(black_frame)
        print(f'Number of unique frames: {len(frames)}')
        return frames
    
    @staticmethod
    def convert_images_to_video(input_folder: str, output_file: str, fps: int, width: int, height: int, duration_per_image: float):
        print(f"\n*** {input_folder} ***")

        # Folder settings
        image_folders = [input_folder]
        image_types = ['jpg', 'JPG', 'jpeg', 'JPEG']

        # Image paths
        image_paths = sum([LibIsa.get_image_paths(image_folder, image_types) for image_folder in image_folders], [])
        not_image_paths = sum([LibIsa.get_non_image_paths(image_folder, image_types) for image_folder in image_folders], [])

        unique_frames = LibIsa.get_unique_frames(image_paths, width, height)
        # duration_per_image = LibIsa.get_duration_per_image(duration, image_paths)
        frames_per_image = LibIsa.get_frames_per_image(duration_per_image, fps)
        max_shape = LibIsa.get_max_shape(unique_frames)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
        video = cv2.VideoWriter(
            output_file,
            fourcc,
            fps,
            (max_shape[0], max_shape[1]))

        print(f"*** Writing frames to {output_file} ***")
        for i, frame in enumerate(unique_frames):
            percentage = int((i + 1) / len(unique_frames) * 100)
            print(f"--> Writing unique frame {i} to {output_file} [{percentage}%]")
            for j in range(frames_per_image):
                video.write(frame)

        print("Releasing video...")
        video.release()

        print("Destroying all windows...")
        cv2.destroyAllWindows()