# if it doesn't work, export PATH=$PATH:<path where you installed ffmpeg> 
# export PATH=$PATH:/tmp/ruibiny/data/speech/ffmpeg-4.4.1-amd64-static/
from tqdm import tqdm
import os
import pickle
from pydub import AudioSegment
from pydub import effects

# PROJECT_ROOT = '/afs/andrew.cmu.edu/usr12/ruibiny/11751/id-deid/ID-DEID/'
PROJECT_ROOT = '/home/ubuntu/mnt/yrb/ID-DEID/'
DATA_PATH = os.path.join(PROJECT_ROOT, 'data/')
AUTOVC_ROOT = os.path.join(PROJECT_ROOT, 'deid/autovc/')

os.chdir(PROJECT_ROOT+'datasets/vctk')

vctk_flac_path = os.path.join(DATA_PATH, 'vctk', 'wav48_silence_trimmed/')
vctk_txt_path = os.path.join(DATA_PATH, 'vctk', 'txt/')
speaker_ids = os.listdir(vctk_flac_path)
speaker_ids = [i for i in speaker_ids if 'log.txt' not in i]
speaker_ids.sort()

parsed_data = {"vctk": {}}
try:
    with open(os.path.join(PROJECT_ROOT, "data", "vctk_wavs", "parsed_vctk.pkl"), 'rb') as ff:
        parsed_data = pickle.load(ff)
except FileNotFoundError:
    pass
for spkr_id in tqdm(speaker_ids, position=1):
    spkr_audio_folder = os.path.join(vctk_flac_path, spkr_id)
    spkr_txt_folder = os.path.join(vctk_txt_path, spkr_id)
    # only need single channel
    audios = os.listdir(spkr_audio_folder)
    audios = [i for i in audios if 'mic1' in i]
    audios.sort()
    for audio_name in tqdm(audios, position=0):
        audio_path = os.path.join(spkr_audio_folder, audio_name)
        export_dir = os.path.join(PROJECT_ROOT, "data", "vctk_wavs", spkr_id)
        os.makedirs(export_dir, exist_ok=True)
        shorter_export_name = os.path.splitext(audio_name)[0].replace('_mic1', '')
        export_path = os.path.join(export_dir, shorter_export_name) + '.wav'

        if parsed_data["vctk"].get(spkr_id, None):
            if parsed_data["vctk"][spkr_id].get(export_path.replace(PROJECT_ROOT, ''), None):
                continue

        # flac to wav
        audio = AudioSegment.from_file(audio_path, "flac").set_frame_rate(16000)
        audio = effects.normalize(audio)
        audio.export(export_path, format="wav")

        # txt to pickle
        txt_path = os.path.join(spkr_txt_folder, shorter_export_name + ".txt")
        try:
            with open(txt_path, 'r') as ff:
                lines = ff.readlines()
            assert len(lines) == 1
            ref = lines[0].strip()
        except FileNotFoundError:
            ref = ''
        if parsed_data["vctk"].get(spkr_id, None) is None:
            parsed_data["vctk"][spkr_id] = {}
        update_dict = {export_path.replace(PROJECT_ROOT, ''): ref}
        parsed_data["vctk"][spkr_id].update(update_dict)
    with open(os.path.join(PROJECT_ROOT, "data", "vctk_wavs", "parsed_vctk.pkl"), 'wb') as ff:
        pickle.dump(parsed_data, ff)
