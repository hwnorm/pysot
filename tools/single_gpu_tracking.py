from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob
import copy
import pickle
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

def calc_iou(bbox1, bbox2):
    if not isinstance(bbox1, np.ndarray):
        bbox1 = np.array(bbox1)
    if not isinstance(bbox2, np.ndarray):
        bbox2 = np.array(bbox2)
    
    xmin1, ymin1, xmax1, ymax1, = np.split(bbox1, 4, axis=-1)
    xmin2, ymin2, xmax2, ymax2, = np.split(bbox2, 4, axis=-1)

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    ymin = np.maximum(ymin1, np.squeeze(ymin2, axis=-1))
    xmin = np.maximum(xmin1, np.squeeze(xmin2, axis=-1))
    ymax = np.minimum(ymax1, np.squeeze(ymax2, axis=-1))
    xmax = np.minimum(xmax1, np.squeeze(xmax2, axis=-1))

    h = np.maximum(ymax - ymin, 0)
    w = np.maximum(xmax - xmin, 0)
    intersect = h * w

    union = area1 + np.squeeze(area2, axis=-1) - intersect
    return intersect / union

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--proposal_dict', type=str)
parser.add_argument('--vp_dict', type=str)
parser.add_argument('--vp_list', type=str)
parser.add_argument('--human_tracker', type=str)
parser.add_argument("--range",
                    nargs=2,
                    type=int,
                    default=[0, -1],
                    help="")
args = parser.parse_args()

def main():
    os.makedirs('outputs/', exist_ok=True)
    video_dir = '/home/pris1/Downloads/clips'
    video_file_list = os.listdir(video_dir)
    vp_to_file_dict = dict()
    for videofile in video_file_list:
        tracker, video, person, sec_start, _, sec_end = os.path.splitext(videofile)[0].split('%')
        if tracker == args.human_tracker:
            vp = video+'.'+person
            vp_to_file_dict[vp] = [videofile, int(sec_start)]

    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = True
    device = torch.device('cuda')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    vp_list = pickle.load(open(args.vp_list, 'rb'))
    proposal_dict = pickle.load(open(args.proposal_dict, 'rb'))
    vp_dict = pickle.load(open(args.vp_dict, 'rb'))
    if args.range[-1] == -1:
        args.range[-1] == len(vp_dict)

    vp_range = vp_list[args.range[0]:args.range[1]]
    for vp_idx, vp in enumerate(vp_range):
        videofile, sec_start = vp_to_file_dict[vp]
        video, person = vp.split('.')
        start_vf = video+'.'+str(sec_start)
        if start_vf in proposal_dict:
            start_length = len(proposal_dict[start_vf])
            for obj_idx in range(start_length):
                print('vp: {} / {}, obj: {} / {}'.format(vp_idx+1, len(vp_range), obj_idx+1, start_length))
                track_id = '%'.join([video, person, str(obj_idx)])
                track_dict = dict()
                start_rect = copy.deepcopy(proposal_dict[start_vf][obj_idx])
                videopath = os.path.join(video_dir, videofile)
                cap = cv2.VideoCapture(videopath)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                frame_idx = 0
                first_frame = True
                while True:
                    ret, frame = cap.read()
                    if ret:
                        sec = frame_idx // fps + sec_start
                        remained = frame_idx % fps
                        
                        if remained == 0:
                            if first_frame:
                                init_rect = copy.deepcopy(start_rect) # xywh
                                track_dict[sec] = copy.deepcopy(init_rect)
                                init_rect[2] -= init_rect[0]
                                init_rect[3] -= init_rect[1]
                                tracker.init(frame, init_rect)
                                first_frame = False
                            else:
                                outputs = tracker.track(frame)
                                bbox = outputs['bbox'] # xywh
                                bbox[2] += bbox[0]
                                bbox[3] += bbox[1]
                                vf = video+'.'+str(sec)
                                if vf in proposal_dict:
                                    proposals = copy.deepcopy(proposal_dict[vf])
                                    track_results = np.expand_dims(np.array(bbox), axis=0)
                                    iou_mat = calc_iou(track_results, proposals)[0]
                                    max_proposal_idx = np.argmax(iou_mat)
                                    proposal_adjust = copy.deepcopy(proposals[max_proposal_idx])
                                    track_dict[sec] = copy.deepcopy(proposal_adjust)
                                    proposal_adjust[2] -= proposal_adjust[0]
                                    proposal_adjust[3] -= proposal_adjust[1]
                                    tracker.init(frame, proposal_adjust)
                                else:
                                    track_dict[sec] = copy.deepcopy(bbox)
                        else:
                            outputs = tracker.track(frame)

                        frame_idx += 1
                    else:
                        break
                pickle.dump(track_dict, open('outputs/{}.pkl'.format(track_id), 'wb'))
        else:
            pass
if __name__ == '__main__':
    main()
