#@ type: compute
#@ parents:
#@   - func1
#@ dependents:
#@   - func3
#@ corunning:
#@   mem1:
#@     trans: mem1
#@     type: rdma
#@   mem2:
#@     trans: mem2
#@     type: rdma
#@   mem3:
#@     trans: mem3
#@     type: rdma
#@   mem4:
#@     trans: mem4
#@     type: rdma

import pickle
import numpy as np
import base64
import disaggrt.buffer_pool_lib as buffer_pool_lib
from disaggrt.rdma_array import remote_array

import ffmpeg
import cv2

SERVER_NUM = 4

def cyberpunk(image):
    # 反转色相
    image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    image_hls = np.asarray(image_hls, np.float32)
    hue = image_hls[:, :, 0]
    hue[hue < 90] = 180 - hue[hue < 90]
    image_hls[:, :, 0] = hue

    image_hls = np.asarray(image_hls, np.uint8)
    image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2BGR)

    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    image_lab = np.asarray(image_lab, np.float32)
    
    # 提高像素亮度，让亮的地方更亮
    light_gamma_high = np.power(image_lab[:, :, 0], 0.8)
    light_gamma_high = np.asarray(light_gamma_high / np.max(light_gamma_high) * 255, np.uint8)

     # 降低像素亮度，让暗的地方更暗
    light_gamma_low = np.power(image_lab[:, :, 0], 1.2)
    light_gamma_low = np.asarray(light_gamma_low / np.max(light_gamma_low) * 255, np.uint8)

    # 调色至偏紫
    dark_b = image_lab[:, :, 2] * (light_gamma_low / 255) * 0.1
    dark_a = image_lab[:, :, 2] * (1 - light_gamma_high / 255) * 0.3

    image_lab[:, :, 2] = np.clip(image_lab[:, :, 2] - dark_b, 0, 255)
    image_lab[:, :, 2] = np.clip(image_lab[:, :, 2] - dark_a, 0, 255)

    image_lab = np.asarray(image_lab, np.uint8)
    return cv2.cvtColor(image_lab, cv2.COLOR_Lab2BGR)

def processFrame(count, width):
    mem_name = "mem" + str(count) + '.npy'
    with open(mem_name, 'rb') as f:
        in_frame = np.load(f)
        frame_num = len(in_frame)
        frame_index = 1 if count == 1 else frame_num
        frame_list = []
        for frame in in_frame:
            in_frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            current_width = int(width * (frame_index / frame_num))
            in_frame_bgr[:, 0:current_width, :] = cyberpunk(in_frame_bgr[:, 0:current_width, :])
            f = cv2.cvtColor(in_frame_bgr, cv2.COLOR_BGR2RGB)

            frame_list.append(f)

            if frame_index < frame_num:
                frame_index += 1
        
        out_frame = np.asarray(frame_list)
        out_name = "out" + str(count) + '.npy'
        with open(out_name, 'wb') as ff:
            np.save(ff, out_frame)


def main():
    video_path = 'sample-mp4-file.mp4'
    video_probe = ffmpeg.probe(video_path)
    video_info = next((stream for stream in video_probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_info['width'])

    for i in range(1, SERVER_NUM+1):
        processFrame(i, width)

if __name__ == '__main__':
    main()