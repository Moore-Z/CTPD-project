from scipy import stats
import os
import pandas as pd
from tqdm import tqdm  # 导入tqdm库
import re
import json
from nltk import sent_tokenize, word_tokenize
import warnings
warnings.filterwarnings("ignore")

SECTION_TITLES = re.compile(
    r'('
    r'ABDOMEN AND PELVIS|CLINICAL HISTORY|CLINICAL INDICATION|COMPARISON|COMPARISON STUDY DATE'
    r'|EXAM|EXAMINATION|FINDINGS|HISTORY|IMPRESSION|INDICATION'
    r'|MEDICAL CONDITION|PROCEDURE|REASON FOR EXAM|REASON FOR STUDY|REASON FOR THIS EXAMINATION'
    r'|TECHNIQUE'
    r'):|FINAL REPORT',
    re.I | re.M)

def pattern_repl(matchobj):
    return ' '.rjust(len(matchobj.group(0)))

def find_end(text):
    ends = [len(text)]
    patterns = [
        re.compile(r'BY ELECTRONICALLY SIGNING THIS REPORT', re.I),
        re.compile(r'\n {3,}DR.', re.I),
        re.compile(r'[ ]{1,}RADLINE ', re.I),
        re.compile(r'.*electronically signed on', re.I),
        re.compile(r'M\[0KM\[0KM')
    ]
    for pattern in patterns:
        matchobj = pattern.search(text)
        if matchobj:
            ends.append(matchobj.start())
    return min(ends)

def split_heading(text):
    start = 0
    for matcher in SECTION_TITLES.finditer(text):
        end = matcher.start()
        if end != start:
            section = text[start:end].strip()
            if section:
                yield section
        start = end
        end = matcher.end()
        if end != start:
            section = text[start:end].strip()
            if section:
                yield section
        start = end
    end = len(text)
    if start < end:
        section = text[start:end].strip()
        if section:
            yield section

def clean_text(text):
    text = re.sub(r'\[\*\*.*?\*\*\]', pattern_repl, text)
    text = re.sub(r'_', ' ', text)
    start = 0
    end = find_end(text)
    new_text = ''
    if start > 0:
        new_text += ' ' * start
    new_text = text[start:end]
    if len(text) - end > 0:
        new_text += ' ' * (len(text) - end)
    return new_text

def preprocess_mimic(text):
    for sec in split_heading(clean_text(text)):
        for sent in sent_tokenize(sec):
            text = ' '.join(word_tokenize(sent))
            yield text.lower()

df = pd.read_csv("/Users/haochengyang/Desktop/Research/CTPD/MMMSPG-014C/EHR_dataset/mimiciii/1.4/NOTEEVENTS.csv")
df.CHARTDATE = pd.to_datetime(df.CHARTDATE)
df.CHARTTIME = pd.to_datetime(df.CHARTTIME)
df.STORETIME = pd.to_datetime(df.STORETIME)
df2 = df[df.SUBJECT_ID.notnull()]
df2 = df2[df2.HADM_ID.notnull()]
df2 = df2[df2.CHARTTIME.notnull()]
df2 = df2[df2.TEXT.notnull()]
df2 = df2[['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'TEXT']]
del df

def filter_for_first_hrs(dataframe, _days=2):
    min_time = dataframe.CHARTTIME.min()
    return dataframe[dataframe.CHARTTIME < min_time + pd.Timedelta(days=_days)]

def getText(t):
    return " ".join(list(preprocess_mimic(t)))

def getSentences(t):
    return list(preprocess_mimic(t))

print(df2.groupby('HADM_ID').count().describe())

dataset_path = "/Users/haochengyang/Desktop/Research/CTPD/MMMSPG-014C/EHR_dataset/mimiciii_benchmark/root/test"
all_files = os.listdir(dataset_path)
all_folders = list(filter(lambda x: x.isdigit(), all_files))
output_folder = "/Users/haochengyang/Desktop/Research/CTPD/MMMSPG-014C/EHR_dataset/mimiciii_benchmark/root/test_text_fixed"

suceed = 0
failed = 0
failed_exception = 0
sentence_lens = []
hadm_id2index = {}

for folder in tqdm(all_folders, desc="Processing Folders"):  # 添加tqdm进度条
    try:
        patient_id = int(folder)
        sliced = df2[df2.SUBJECT_ID == patient_id]
        if sliced.shape[0] == 0:
            print("No notes for PATIENT_ID : {}".format(patient_id))
            failed += 1
            continue
        sliced.sort_values(by='CHARTTIME', inplace=True)
        stays_path = os.path.join(dataset_path, folder, 'stays.csv')
        stays_df = pd.read_csv(stays_path)
        hadm_ids = list(stays_df.HADM_ID.values)
        for ind, hid in enumerate(hadm_ids):
            hadm_id2index[str(hid)] = str(ind)
            hid_sliced = sliced[sliced.HADM_ID == hid]
            data_json = {}
            for index, row in hid_sliced.iterrows():
                data_json["{}".format(row['CHARTTIME'])] = getSentences(row['TEXT'])
            with open(os.path.join(output_folder, folder + '_' + str(ind+1)), 'w') as f:
                json.dump(data_json, f)
        suceed += 1
    except:
        import traceback
        traceback.print_exc()
        print("Failed with Exception FOR Patient ID: %s", folder)
        failed_exception += 1

print("Sucessfully Completed: %d/%d" % (suceed, len(all_folders)))
print("No Notes for Patients: %d/%d" % (failed, len(all_folders)))
print("Failed with Exception: %d/%d" % (failed_exception, len(all_folders)))

with open(os.path.join(output_folder, 'test_hadm_id2index'), 'w') as f:
    json.dump(hadm_id2index, f)
