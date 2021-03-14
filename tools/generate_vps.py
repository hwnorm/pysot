import os
from tqdm import tqdm
from collections import defaultdict
import pickle


for tracker in ['deepsort', 'fairmot']:
    vp_dict = defaultdict(list)
    tracking_csv_file = '/home/pris1/tracking/pysot/DIO_track_{}.csv'.format(tracker)
    with open(tracking_csv_file, 'r') as f:
        for line in tqdm(f):
            video, sec, _, _, _, _, _, human_id = line.split(',')
            vp = video+'.'+str(int(human_id))
            sec = int(sec)
            vp_dict[vp].append(sec)
    pickle.dump(vp_dict, open('vp_dict_{}.pkl'.format(tracker), 'wb'))