# Voice Activity Detection (VAD) with LibriParty
This folder contains scripts for training a VAD with the [LibriParty dataset](https://drive.google.com/file/d/1--cAS5ePojMwNY5fewioXAv9YlYAWzIJ/view?usp=sharing).
LibriParty contains sequences of 1 minute compose of speech sentences (sampled from LibriSpeech) corrupted by noise and reverberation.
Data augmentation with open_rir, musan, CommonLanguge is used as well. Make sure you download all the datasets before staring the experiment:
- LibriParty: https://drive.google.com/file/d/1--cAS5ePojMwNY5fewioXAv9YlYAWzIJ/view?usp=sharing
- Musan: https://www.openslr.org/resources/17/musan.tar.gz
- CommonLanguage: https://zenodo.org/record/5036977/files/CommonLanguage.tar.gz?download=1


# Training a RNN-based VAD
Run the following command to train the model:
`python train.py hparams/train.yaml --data_folder=/localscratch/LibriParty/dataset/ --musan_folder=/localscratch/musan/ --commonlanguage_folder=/localscratch/common_voice_kpd`
(change the paths with your local ones)


# Results
| Release | hyperparams file | Test Precision | Test Recall. | Test F-Score | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| -----:| --------:| -----------:| -----------:|
| 2021-09-09 | train.yaml |  0.9518 | 0.9437 | 0.9477 | [Model](https://drive.google.com/drive/folders/1YLYGuiyuTH0D7fXOOp6cMddfQoM74o-Y?usp=sharing) | 1xV100 16GB |


# Training Time
About 12 minutes for each epoch with a TESLA V100.

# Inference
The pre-trained model + easy inference is available on HuggingFace:
- https://huggingface.co/speechbrain/vad-crdnn-libriparty

Basically, you can run inference with only a few lines of code:

```python
from speechbrain.pretrained import VAD

VAD = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="pretrained_models/vad-crdnn-libriparty")
boundaries = VAD.get_speech_segments("speechbrain/vad-crdnn-libriparty/example_vad.wav")

# Print the output
VAD.save_boundaries(boundaries)
```


# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/

# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```

