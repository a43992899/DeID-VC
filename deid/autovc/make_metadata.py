"""
Generate speaker embeddings and metadata for training
"""
import os
import pickle
from hparams import DATA_PATH
from model_bl import D_VECTOR
from collections import OrderedDict
import numpy as np
import torch
from tqdm import tqdm

torch.set_num_threads(4)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().to(device)
c_checkpoint = torch.load(DATA_PATH+'3000000-BL.ckpt', map_location=device)
new_state_dict = OrderedDict()
for key, val in c_checkpoint['model_b'].items():
    new_key = key[7:]
    new_state_dict[new_key] = val
C.load_state_dict(new_state_dict)
num_uttrs = 10
len_crop = 128

# Directory containing mel-spectrograms
rootDir = DATA_PATH+'./spmel'
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)


speakers = []
for speaker in tqdm(sorted(subdirList), position=1):
    # print('Processing speaker: %s' % speaker)
    utterances = []
    utterances.append(speaker)
    _, _, fileList = next(os.walk(os.path.join(dirName,speaker)))
    
    # make speaker embedding
    # each speaker pick 10 corpped utters and extract mean embedding
    assert len(fileList) >= num_uttrs
    idx_uttrs = np.random.choice(len(fileList), size=num_uttrs, replace=False)
    embs = []
    for i in tqdm(range(num_uttrs), position=0):
        tmp = np.load(os.path.join(dirName, speaker, fileList[idx_uttrs[i]]))
        candidates = np.delete(np.arange(len(fileList)), idx_uttrs)
        # choose another utterance if the current one is too short
        while tmp.shape[0] < len_crop:
            idx_alt = np.random.choice(candidates)
            tmp = np.load(os.path.join(dirName, speaker, fileList[idx_alt]))
            candidates = np.delete(candidates, np.argwhere(candidates==idx_alt))
        assert tmp.shape[0]-len_crop >= 0, "wav too short"
        left = np.random.randint(0, tmp.shape[0]-len_crop) if tmp.shape[0] > len_crop else 0
        melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :]).to(device)
        emb = C(melsp)
        embs.append(emb.detach().squeeze().cpu().numpy())
    utterances.append(np.mean(embs, axis=0))
    
    # create file list
    for fileName in sorted(fileList):
        utterances.append(os.path.join(speaker,fileName))
    speakers.append(utterances)
    
with open(os.path.join(rootDir, 'train.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)

