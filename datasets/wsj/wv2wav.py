# if it doesn't work, export PATH=$PATH:<path where you installed ffmpeg> 
# export PATH=$PATH:/tmp/ruibiny/data/speech/ffmpeg-4.4.1-amd64-static/
from tqdm import tqdm
import os
import pickle
from pydub import AudioSegment
from pydub import effects

# PROJECT_ROOT = '/afs/andrew.cmu.edu/usr12/ruibiny/11751/id-deid/ID-DEID/'
PROJECT_ROOT = '/home/ubuntu/mnt/yrb/ID-DEID/'
DATA_PATH = os.path.join(PROJECT_ROOT, 'data/speech/')
AUTOVC_ROOT = os.path.join(PROJECT_ROOT, 'deid/autovc/')

os.chdir(PROJECT_ROOT+'datasets/wsj')

# this script reads the output 'csr_senn.csv', 'csr_1_senn.csv' from parse.py, turns .wv1 into .wav in PROJECT_ROOT/data/wav
csvs = ['csr_senn.csv', 'csr_1_senn.csv']
parsed_data = {"csr_senn.csv": {}, "csr_1_senn.csv": {}}
parsed_paths = set()
skip_cnt = 0
for csv in csvs:
    with open(os.path.join(PROJECT_ROOT,'datasets/wsj/',csv), 'r') as csvfp:
        lines = csvfp.readlines()
    for line in tqdm(lines):
        audio_path, speaker_id, text = line.strip().split('|')
        if audio_path in parsed_paths:
            skip_cnt += 1
            continue
        parsed_paths.add(audio_path)
        audio = AudioSegment.from_file(audio_path, "nistsphere")
        audio = effects.normalize(audio)
        export_dir = os.path.join(PROJECT_ROOT, "data", "wsj_wavs", speaker_id)
        os.makedirs(export_dir, exist_ok=True)
        export_path = os.path.join(export_dir, os.path.split(audio_path)[-1])
        audio.export(export_path+".wav", format="wav")
        if parsed_data[csv].get(speaker_id, None):
            parsed_data[csv][speaker_id].update({(export_path+".wav").replace(PROJECT_ROOT, ''): text})
        else:
            parsed_data[csv][speaker_id] = {audio_path: text}
        if csv == csvs[1]:
            assert parsed_data[csvs[0]].get(speaker_id, None) == None, "ERROR: overlapped speaker id"
    with open(os.path.join(PROJECT_ROOT, "data", "wsj_wavs", 'parsed_wsj.pkl'), 'wb') as ff:
        pickle.dump(parsed_data, ff)
    print('skip cnt', skip_cnt)