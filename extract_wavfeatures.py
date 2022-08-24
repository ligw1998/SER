from h5py import File
import librosa
import pandas as pd
from tqdm import tqdm
import argparse
import os
import python_speech_features as ps
import numpy as np
import pickle

emotion_dict = {'ang': 0, 'hap': 1, 'exc': 2, 'sad': 3, 'fru': 4, 'fea': 5, 'sur': 6, 'neu': 7, 'xxx': 8, 'oth': 8}
total_Sess = {'Ses01': 'Session1', 'Ses02': 'Session2', 'Ses03': 'Session3', 'Ses04': 'Session4', 'Ses05': 'Session5'}
eps = 1e-5


def main(args):
    with open(args.ds_sum, 'rb') as d:
        mean1, std1, mean2, std2, mean3, std3 = pickle.load(d)
    labels_df = pd.read_csv(args.df_file)
    with tqdm(total=len(labels_df)) as pbar, File(args.opt) as opt:
        for i in range(0, len(labels_df)):
            wav_file = labels_df.iloc[i]['wav_file']
            start_time = labels_df.iloc[i]['start_time']
            end_time = labels_df.iloc[i]['end_time']
            label = emotion_dict[labels_df.iloc[i]['emotion']]
            for key in total_Sess:
                if key in emotion_dict:
                    cur_datadir = args.data_dir.replace("Session0", total_Sess[key])
                    break
            wav, _sr = librosa.load(f'{cur_datadir}/{wav_file}.wav', sr=args.sr)
            mel_spec = ps.logfbank(wav, _sr, nfilt=40)
            delta1 = ps.delta(mel_spec, 2)
            delta2 = ps.delta(delta1, 2)
            time = mel_spec.shape[0]
            if time <= 300:
                part = mel_spec
                delta11 = delta1
                delta21 = delta2
                part = np.pad(part, ((0, 300 - part.shape[0]), (0, 0)), 'constant', constant_values=0)
                delta11 = np.pad(delta11, ((0, 300 - delta11.shape[0]), (0, 0)), 'constant', constant_values=0)
                delta21 = np.pad(delta21, ((0, 300 - delta21.shape[0]), (0, 0)), 'constant', constant_values=0)
            else:
                begin = int((time - 300) / 2)
                end = begin + 300
                part = mel_spec[begin:end, :]
                delta11 = delta1[begin:end, :]
                delta21 = delta2[begin:end, :]
            opt[f"{wav_file}/wav"] = wav
            opt[f"{wav_file}/spec"] = (part - mean1) / (std1 + eps)
            opt[f"{wav_file}/delta1"] = (delta11 - mean2) / (std2 + eps)
            opt[f"{wav_file}/delta2"] = (delta21 - mean3) / (std3 + eps)
            opt[f"{wav_file}/label"] = label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--df_file', type=str)
    parser.add_argument('--opt', type=str)
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--ds_sum', type=str, default='./zscore40.pkl')
    args = parser.parse_args()
    main(args)
