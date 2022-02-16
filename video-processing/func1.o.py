#@ type: compute
#@ dependents:
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

import pickle
import threading
import numpy as np
import base64
import disaggrt.buffer_pool_lib as buffer_pool_lib
from disaggrt.rdma_array import remote_array

import ffmpeg

FRAME_NUM = 150
LOCK1 = threading.Lock()
LOCK2 = threading.Lock()

def processDataSlice(count, in_bytes, context_dict, action, height, width):
    mem_name = "mem" + str(count)
    LOCK1.acquire()
    trans = action.get_transport(mem_name, 'rdma')
    LOCK1.release()
    trans.reg(buffer_pool_lib.buffer_size)
    buffer_pool = buffer_pool_lib.buffer_pool({mem_name:trans})
    in_frame = (
        np
            .frombuffer(in_bytes, np.uint8)
            .reshape([-1, height, width, 3])
    )
    remote_input = remote_array(buffer_pool, input_ndarray=in_frame, transport_name=mem_name)
    # update context
    remote_input_metadata = remote_input.get_array_metadata()
    LOCK2.acquire()
    context_dict["remote_input" + str(count)] = remote_input_metadata
    context_dict["buffer_pool_metadata" + str(count)] = buffer_pool.get_buffer_metadata()
    LOCK2.release()

def main(context_dict, action):
    # loading data
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

    count = 1
    context_dict = {}
    threads = []
    while True:
        # setup
        in_bytes = in_process.stdout.read(width * height * 3 * FRAME_NUM)
        if not in_bytes:
            break
        
        t = threading.Thread(target=processDataSlice, args=(count, in_bytes, context_dict, action, height, width))
        t.start()
        threads.append(t)
        
        count += 1

    count = 0
    for t in threads:
        count += 1
        print(count)
        t.join()
    
    in_process.wait()
    context_dict_in_byte = pickle.dumps(context_dict)
    return {'meta': base64.b64encode(context_dict_in_byte).decode("ascii")}
