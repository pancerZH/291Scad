import ffmpeg

split = ffmpeg.input('sample-mp4-file.mp4').filter_multi_output('split', 'split=5')
split0 = split.stream(0) 
split1 = split[1]
split2 = split[2]
ffmpeg.concat(split0, split1, split2).output('out.mp4').run()