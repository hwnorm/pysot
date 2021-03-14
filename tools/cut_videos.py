import os
import pickle
from tqdm import tqdm
video_dir = '/home/pris1/Downloads/ava_videos'
video_file_list = os.listdir(video_dir)
video_to_file_dict = dict()
for videofile in video_file_list:
    video = videofile.split('.')[0]
    video_to_file_dict[video] = videofile

for tracker in ['deepsort', 'fairmot']:
    vp_dict = pickle.load(open('vp_dict_{}.pkl'.format(tracker), 'rb'))
    for vp, secs in tqdm(vp_dict.items()):
        video, person = vp.split('.')
        videofile = video_to_file_dict[video]
        videopath = os.path.join(video_dir, videofile)
        sec_min = min(secs) - 900
        sec_max = max(secs) + 1 - 900
        os.system('ffmpeg -accurate_seek -ss {:d} -i {:s} -t {:d} -codec copy -avoid_negative_ts 1 /home/pris1/Downloads/clips/{:s}%{:s}%{:s}%{:d}%to%{:d}.mp4'.format(sec_min, videopath, sec_max, tracker, video, person, sec_min+900, sec_max+900))