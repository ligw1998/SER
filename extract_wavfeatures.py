from h5py import File
import librosa
import pandas as pd
from tqdm import tqdm
import argparse
import os
import python_speech_features as ps
import numpy as np
import pickle
import math

emotion_dict = {'ang': 0, 'hap': 1, 'exc': 2, 'sad': 3, 'fru': 4, 'fea': 5, 'sur': 6, 'neu': 7, 'xxx': 8, 'oth': 8}
total_Sess = {'Ses01': 'Session1', 'Ses02': 'Session2', 'Ses03': 'Session3', 'Ses04': 'Session4', 'Ses05': 'Session5'}
eps = 1e-5


def main(args):
    # with open(args.ds_sum, 'rb') as d:
        # mean1, std1, mean2, std2, mean3, std3 = pickle.load(d)
    labels_df = pd.read_csv(args.df_file)
    with tqdm(total=len(labels_df)) as pbar, File(args.opt,'w') as opt:
        for i in range(0, len(labels_df)):
            wav_file = labels_df.iloc[i]['wav_file'][:-5]
            wav_file_df = labels_df.iloc[i]['wav_file']
            start_time = labels_df.iloc[i]['start_time']
            end_time = labels_df.iloc[i]['end_time']
            label = emotion_dict[labels_df.iloc[i]['emotion'] if (labels_df.iloc[i]['emotion'] in emotion_dict ) else 'xxx']
            for seskey in total_Sess:
                if(seskey in wav_file):
                    cur_datadir = args.data_dir.replace("Session0", total_Sess[seskey])
                    cur_ses =  wav_file[:6]
                    # cur_gender = wav_file[5:6]
                    break

            wav, _sr = librosa.load(f'{cur_datadir}/{wav_file}.wav', sr=args.sr)
            start_frame = math.floor(start_time * _sr)
            end_frame = math.floor(end_time * _sr)
            truncated_wav_vector = wav[start_frame:end_frame + 1]
            mel_spec = ps.logfbank(truncated_wav_vector, _sr, nfilt=40)
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
            opt[f"{cur_ses}/{wav_file_df}/wav"] = truncated_wav_vector
            opt[f"{cur_ses}/{wav_file_df}/spec"] = part
            opt[f"{cur_ses}/{wav_file_df}/delta1"] = delta11
            opt[f"{cur_ses}/{wav_file_df}/delta2"] = delta21
            opt[f"{cur_ses}/{wav_file_df}/label"] = label
            pbar.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,default='/home/liguangwei/data/IEMOCAP_full_release/Session0/dialog/wav')
    parser.add_argument('--df_file', type=str,default='/home/liguangwei/emotion_detection/emotion_recognition/data/pre_processed/df_iemocap.csv')
    parser.add_argument('--opt', type=str,default='./data/spec.h5')
    parser.add_argument('--sr', type=int, default=16000)
    # parser.add_argument('--ds_sum', type=str, default='./zscore40.pkl')
    args = parser.parse_args()
    main(args)
