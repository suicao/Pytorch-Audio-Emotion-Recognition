# 1st Place Public Leaderboard Solution for [ERC2019](https://erc2019.com/top-10#top_10)

## Overview
Although the competition is for recognizing emotion from audio data. You can treat this codebase for a baseline for audio classification in general, I did not make any assumption about the provided data.
This is is pipeline
## Key features
### Preprocessing by converting the audio to Mel Spectrogram.
I used librosa with this config:
- sampling_rate = 16000
- duration = 2  # sec
- hop_length = 125 * duration 
- n_mels = 128

Basically, each 128x128 image represents 2 second of audio.
### Fully CNN for audio classification
Recently we won a gold medal in Kaggle's [Freesound Audio Tagging 2019](https://www.kaggle.com/c/freesound-audio-tagging-2019/leaderboard) and thus most of the architectures were borrowed from there.

The final submission was an ensemble of 4 models. 3 of them were ```Classifier_M0```, ```Classifier_M2``` and ```Classifier_M3``` from our technical report [[1]](https://easychair.org/publications/preprint_open/MpKs):
![](https://i.imgur.com/Z6syhc6.png)

Here's what ```Classifier_M3``` looks like:
![](https://i.imgur.com/T8vq1pv.png)

The other model came from the [7th place solution](https://www.kaggle.com/hidehisaarai1213/freesound-7th-place-solution)

### Mixup + SpecAugment (SpecMix)

The most important part of this solution is the augmentation method, as the dataset is very small and pretraining is not allowed.

Augmenting options spectrogram are very limited due to the nature of the data (they are not ordinary images e.g rotating a spectrogram makes no sense). In this work I ultilized Mixup [[2]](https://arxiv.org/abs/1710.09412) and SpecAugment [[3]](https://arxiv.org/abs/1904.08779).
This [repo](https://github.com/ebouteillon/freesound-audio-tagging-2019#SpecMix-1) by Eric Bouteillon showed a nice explantion of the method:
![](https://raw.githubusercontent.com/ebouteillon/freesound-audio-tagging-2019/master/images/all_augmentations.png)

## Training

### Preprocessing 
To reproduce the Mels data, run the following command:
```python preprocess.py --train_df_path <path-to>/train_label.csv --train_dir  <path-to>/Train --test_dir  <path-to>/Public_Test --train_output_path ./data/mels_train.pkl --test_output_path ./data/mels_test.pkl```

### Training 
To reproduce the models, run the following commands:

```python train_full.py --train_df_path <path-to>/train_label.csv --test_dir <path-to>/Public_Test/ --model m0 --logdir models_m0 --output_name preds_m0.npy```
```python train_full.py --train_df_path <path-to>/train_label.csv --test_dir <path-to>/Public_Test/ --model m2 --logdir models_m2 --output_name preds_m2.npy```
```python train_full.py --train_df_path <path-to>/train_label.csv --test_dir <path-to>/Public_Test/ --model m3 --logdir models_m3 --output_name preds_m3.npy```
```python train_full.py --train_df_path <path-to>/train_label.csv --test_dir <path-to>/Public_Test/ --model dcase --logdir models_dcase --output_name preds_dcase.npy```



