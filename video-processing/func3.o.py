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

LOCK1 = threading.Lock()
LOCK2 = threading.Lock()

def fetch(count, context_dict, action, map):
    mem_name = "mem" + str(count)
    LOCK1.acquire()
    trans = action.get_transport(mem_name, 'rdma')
    LOCK1.release()
    trans.reg(buffer_pool_lib.buffer_size)

    LOCK1.acquire()
    buffer_pool = buffer_pool_lib.buffer_pool({mem_name:trans}, context_dict["buffer_pool_metadata" + str(count)])
    LOCK1.release()
    load_frames_remote = remote_array(buffer_pool, metadata=context_dict["remote_output" + str(count)])
    in_frame = load_frames_remote.materialize()

    LOCK2.acquire()
    map[mem_name] = in_frame
    LOCK2.release()

def main(params, action):
    # Load from previous memory
    context_dict_in_b64 = params["func2"][0]['meta']
    context_dict_in_byte = base64.b64decode(context_dict_in_b64)
    context_dict = pickle.loads(context_dict_in_byte)

    threads = []
    map = {}
    for i in range(1, 5):
        t = threading.Thread(target=fetch, args=(i, context_dict, action, map))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # loading data
    video_path = 'sample-mp4-file.mp4'
    video_probe = ffmpeg.probe(video_path)
    video_info = next((stream for stream in video_probe['streams'] if stream['codec_type'] == 'video'), None)
    video_frames = int(video_info['nb_frames'])
    width = int(video_info['width'])
    height = int(video_info['height'])
    video_input = ffmpeg.input(video_path)

    tmp_path = 'tmp.mp4'
    tmp_process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), framerate=30)
            .output(tmp_path, pix_fmt='yuv420p', r=30)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )

    for i in range(1, 5):
        mem_name = "mem" + str(i)
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
              .output(video_input.audio, result_path, r=30)
              .run(overwrite_output=True)
    )

    os.remove(tmp_path)

    return result_path