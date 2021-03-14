from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import pickle
import cv2
import torch
import numpy as np
from glob import glob
import copy
from custom_multiprocessing import process_pool

def parse_args():
    parser = argparse.ArgumentParser(description='tracking demo')
    parser.add_argument('--config', type=str, help='config file')
    parser.add_argument('--snapshot', type=str, help='model name')
    parser.add_argument('--proposal_dict', type=str)
    parser.add_argument('--vp_dict', type=str)
    parser.add_argument('--vp_list', type=str)
    parser.add_argument('--human_tracker', type=str)
    parser.add_argument("--gpu_list", type=str)
    parser.add_argument("--range",
                        nargs=2,
                        type=int,
                        default=[0, -1],
                        help="")
    args = parser.parse_args()
    return args

def split_range(num_parts, start_in, end_in):
    a = np.arange(start_in, end_in)
    res = np.array_split(a, num_parts)
    end = list(np.add.accumulate([len(x) for x in res]))
    start = [0] + end[:-1]
    ix = list(zip(start, end))
    return ix

def multiproc(args, gpu_list, data_length):
    cmd = ('CUDA_VISIBLE_DEVICES={gpu} python -u {binary} '
            '--config {config} --snapshot {snapshot} --proposal_dict {proposal_dict} --vp_dict {vp_dict} '
            '--vp_list {vp_list} --human_tracker {human_tracker} --range {start} {end}' )
    print(args.range)
    range_list = split_range(len(gpu_list), args.range[0], args.range[1])
    cmd_cwd_list = [(cmd.format(binary='tools/single_gpu_tracking.py', gpu=gpu, config=args.config, snapshot=args.snapshot, start=range_list[gpu_idx][0], end=range_list[gpu_idx][1], proposal_dict=args.proposal_dict, vp_dict=args.vp_dict, vp_list=args.vp_list, human_tracker=args.human_tracker), '.') for gpu_idx, gpu in enumerate(gpu_list)]

    print('processes num: {:d}, data length: {:d}...'.format(len(cmd_cwd_list), data_length))

    pool = process_pool()
    pool.apply(cmd_cwd_list)
    pool.wait()

if __name__ == '__main__':
    args = parse_args()
    vp_length = len(pickle.load(open(args.vp_list, 'rb')))
    gpu_list = args.gpu_list.split(',')
    if args.range[1] == -1:
        args.range[1] = vp_length

    multiproc(args, gpu_list, vp_length)

