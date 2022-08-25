import re
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)

start_times, end_times, wav_file_names, emotions, vals, acts, doms = [], [], [], [], [], [], []

with tqdm(total=5) as pbar:
    for sess in range(1, 6):
        emo_evaluation_dir = '/home/liguangwei/data/IEMOCAP_full_release/Session{}/dialog/EmoEvaluation/'.format(sess)
        evaluation_files = [l for l in os.listdir(emo_evaluation_dir) if ('Ses' in l)]
        for file in evaluation_files:
            print(f'{emo_evaluation_dir+file} reading...')
            with open(emo_evaluation_dir + file) as f:
                content = f.read()
            # print(f'{emo_evaluation_dir+file} read!')
            info_lines = re.findall(info_line, content)
            for line in info_lines[1:]:  # the first line is a header
                start_end_time, wav_file_name, emotion, val_act_dom = line.strip().split('\t')
                start_time, end_time = start_end_time[1:-1].split('-')
                val, act, dom = val_act_dom[1:-1].split(',')
                val, act, dom = float(val), float(act), float(dom)
                start_time, end_time = float(start_time), float(end_time)
                start_times.append(start_time)
                end_times.append(end_time)
                wav_file_names.append(wav_file_name)
                emotions.append(emotion)
                vals.append(val)
                acts.append(act)
                doms.append(dom)
        pbar.update()

df_iemocap = pd.DataFrame(columns=['start_time', 'end_time', 'wav_file', 'emotion', 'val', 'act', 'dom'])

df_iemocap['start_time'] = start_times
df_iemocap['end_time'] = end_times
df_iemocap['wav_file'] = wav_file_names
df_iemocap['emotion'] = emotions
df_iemocap['val'] = vals
df_iemocap['act'] = acts
df_iemocap['dom'] = doms

df_iemocap.tail()

df_iemocap.to_csv('/home/liguangwei/emotion_detection/emotion_recognition/data/pre_processed/df_iemocap.csv',index=False,sep=',')
