#@ type: compute
#@ parents:
#@   - func2
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

import os
import pickle
import threading
import numpy as np
import base64
import disaggrt.buffer_pool_lib as buffer_pool_lib
from disaggrt.rdma_array import remote_array

import ffmpeg

SERVER_NUM = 4
LOCK2 = threading.Lock()

def fetch(count, map):
    mem_name = "out" + str(count) + '.npy'
    with open(mem_name, 'rb') as f:
        in_frame = np.load(f)
        LOCK2.acquire()
        map[mem_name] = in_frame
        LOCK2.release()

def main():
    # Load from local
    threads = []
    map = {}
    for i in range(1, SERVER_NUM+1):
        t = threading.Thread(target=fetch, args=(i, map))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # loading data
    video_path = 'sample-mp4-file.mp4'
    video_probe = ffmpeg.probe(video_path)
    video_info = next((stream for stream in video_probe['streams'] if stream['codec_type'] == 'video'), None)
    video_frames = int(video_info['nb_frames'])
    fps = int(video_info['r_frame_rate'][:-2])
    width = int(video_info['width'])
    height = int(video_info['height'])
    video_input = ffmpeg.input(video_path)

    tmp_path = 'tmp.mp4'
    tmp_process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), framerate=fps)
            .output(tmp_path, pix_fmt='yuv420p', r=fps)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )

    for i in range(1, SERVER_NUM+1):
        mem_name = "out" + str(i) + '.npy'
        for frame in map[mem_name]:
            tmp_process.stdin.write(
            frame
                .astype(np.uint8)
                .tobytes()
        )

    tmp_process.stdin.close()
    tmp_process.wait()

    result_path = 'sample-mp4-file-new.mp4'
    (
        ffmpeg.input(tmp_path)
              .output(video_input.audio, result_path, r=fps)
              .run(overwrite_output=True)
    )

    os.remove(tmp_path)

    return result_path

if __name__ == '__main__':
    main()