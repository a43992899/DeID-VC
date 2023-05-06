from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import numpy as np
import argparse
import json
import pickle
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import MAX_WAV_VALUE
from models import Generator
import librosa as lr

h = None
device = None

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(input_pkl_dir='output/results.pkl', output_dir='hifigan_output', cp_path='/home/yx/hifi-gan/pretrained/VCTK_V1/g_03280000'):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(cp_path, device)
    generator.load_state_dict(state_dict_g['generator'])

    spect_vc = pickle.load(open(input_pkl_dir, 'rb'))

    os.makedirs(output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, spect in enumerate(spect_vc):
            name = spect[0]
            x = spect[1].T
            x = torch.FloatTensor(x).unsqueeze(0).to(device)
            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            output_file = os.path.join(output_dir, name + '.wav')
            write(output_file, h.sampling_rate, audio)
            print(output_file)


def call(a):
    print('Initializing Inference Process..')

    output_dir = a.output_wavs_dir
    cp_path = '/home/yx/hifi-gan/pretrained/VCTK_V1/g_03280000'
    # cp_path = '/home/yx/hifi-gan/pretrained/UNIVERSAL_V1/g_02500000'

    config_file = os.path.join(os.path.split(cp_path)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a.input_pkl_dir, output_dir, cp_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl_dir", required=True)
    parser.add_argument("--output_wavs_dir", required=True)
    a = parser.parse_args()
    call(a)
