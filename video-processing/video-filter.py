import ffmpeg
import numpy as np
import os
from image import cyberpunk
import cv2

if __name__ == '__main__':
    # 源视频
    video_path = 'sample-mp4-file.mp4'
    video_probe = ffmpeg.probe(video_path)
    video_info = next((stream for stream in video_probe['streams'] if stream['codec_type'] == 'video'), None)
    video_frames = int(video_info['nb_frames'])
    width = int(video_info['width'])
    height = int(video_info['height'])
    video_input = ffmpeg.input(video_path)
    in_process = (
        video_input.video.output('pipe:', format='rawvideo', pix_fmt='rgb24', r=30).run_async(pipe_stdout=True)
    )

    # 滤镜视频流
    tmp_path = 'tmp.mp4'
    tmp_process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), framerate=30)
            .output(tmp_path, pix_fmt='yuv420p', r=30)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )

    frame_index = 1

    # 视频帧处理
    while True:
        in_bytes = in_process.stdout.read(width * height * 3)
        if not in_bytes:
            break
        in_frame = (
            np
                .frombuffer(in_bytes, np.uint8)
                .reshape([height, width, 3])
        )

        # could split here

        # 渐变式局部滤镜视频，过渡时间 5 秒，帧率为 30，则此处设置的值为 150
        in_frame_bgr = cv2.cvtColor(in_frame, cv2.COLOR_RGB2BGR)
        current_width = int(width * (frame_index / 150))
        in_frame_bgr[:, 0:current_width, :] = cyberpunk(in_frame_bgr[:, 0:current_width, :])
        in_frame = cv2.cvtColor(in_frame_bgr, cv2.COLOR_BGR2RGB)

        tmp_process.stdin.write(
            in_frame
                .astype(np.uint8)
                .tobytes()
        )

        if frame_index < 150:
            frame_index += 1

    # 等待异步处理完毕
    tmp_process.stdin.close()
    in_process.wait()
    tmp_process.wait()

    # 将原始视频的音乐合并到新视频
    result_path = 'sample-mp4-file-new.mp4'
    (
        ffmpeg.input(tmp_path)
              .output(video_input.audio, result_path, r=30)
              .run(overwrite_output=True)
    )

    # 删除临时文件
    os.remove(tmp_path)
