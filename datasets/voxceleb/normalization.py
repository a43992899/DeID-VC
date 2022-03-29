import pydub
import os
from tqdm import tqdm

vox1_path = "/home/yrb/code/ID-DEID/data/voxceleb/vox1"
vox2_path = "/home/yrb/code/ID-DEID/data/voxceleb/vox2/wav"

def normalize_audio(path, save_path):
    """
    Normalize audio file.
    """
    audio = pydub.AudioSegment.from_wav(path)
    normalized_audio = audio.normalize()
    normalized_audio.export(save_path, format="wav")

def find_all_files_with_extension(path, extension):
    """
    Walk and find all files with a given extension in a directory.
    """
    files = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(os.path.join(root, filename))
    return files

def normalize_all_audio(path):
    """
    Normalize all audio files in a directory.
    """
    files = find_all_files_with_extension(path, ".wav")
    for f in tqdm(files):
        normalize_audio(f, f)

print("Normalizing vox1...")
normalize_all_audio(vox1_path)
print("Normalizing vox2...")
normalize_all_audio(vox2_path)
