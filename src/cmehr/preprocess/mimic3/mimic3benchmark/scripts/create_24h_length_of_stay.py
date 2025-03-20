import os
import argparse
import numpy as np
import pandas as pd
import random
random.seed(49297)
from tqdm import tqdm

"""
基于前24小时的ICU住院数据预测剩余住院时长
"""

def process_partition(args, partition, eps=1e-6, n_hours=24.0):
    output_dir = os.path.join(args.output_path, partition)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    xty_triples = []
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path, partition))))
    for patient in tqdm(patients, desc='迭代处理{}中的患者'.format(partition)):
        patient_folder = os.path.join(args.root_path, partition, patient)
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))

        for ts_filename in patient_ts_files:
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                lb_filename = ts_filename.replace("_timeseries", "")
                label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))

                # 空标签文件
                if label_df.shape[0] == 0:
                    print("\n\t(空标签文件)", patient, ts_filename)
                    continue

                los = 24.0 * label_df.iloc[0]['Length of Stay']  # 转换为小时
                if pd.isnull(los):
                    print("\n\t(住院时长缺失)", patient, ts_filename)
                    continue

                # 如果总住院时长小于24小时，则跳过
                if los < n_hours - eps:
                    continue

                ts_lines = tsfile.readlines()
                header = ts_lines[0]
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]

                # 只保留前24小时内的数据
                ts_lines = [line for (line, t) in zip(ts_lines, event_times)
                            if -eps < t < n_hours + eps]
                event_times = [t for t in event_times
                               if -eps < t < n_hours + eps]

                # ICU中没有测量数据
                if len(ts_lines) == 0:
                    print("\n\t(ICU中无事件) ", patient, ts_filename)
                    continue

                output_ts_filename = patient + "_" + ts_filename
                with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                    outfile.write(header)
                    for line in ts_lines:
                        outfile.write(line)

                # 计算剩余住院时长（总时长减去前24小时）
                remaining_los = los - n_hours
                if remaining_los < 0:
                    remaining_los = 0  # 确保剩余时间不为负

                # 使用固定的24小时作为预测点
                xty_triples.append((output_ts_filename, n_hours, remaining_los))

    print("创建的样本数量:", len(xty_triples))
    if partition == "train":
        random.shuffle(xty_triples)
    if partition == "test":
        xty_triples = sorted(xty_triples)

    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write('stay,period_length,y_true\n')
        for (x, t, y) in xty_triples:
            listfile.write('{},{:.6f},{:.4f}\n'.format(x, t, y))


def main():
    '''
    使用方法:
    python -m mimic3benchmark.scripts.create_24h_length_of_stay [root_path] [output_path]
    
    例如:
    python -m mimic3benchmark.scripts.create_24h_length_of_stay /Users/haochengyang/Desktop/research/CTPD/MMMSPG-014C/EHR_dataset/mimiciii_benchmark/root /Users/haochengyang/Desktop/research/CTPD/MMMSPG-014C/EHR_dataset/mimiciii_benchmark/los/
    '''
    parser = argparse.ArgumentParser(description="创建基于前24小时数据的住院时长预测任务数据。")
    parser.add_argument('root_path', type=str, help="包含训练集和测试集的根文件夹路径。")
    parser.add_argument('output_path', type=str, help="创建的数据应存储的目录。")
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process_partition(args, "test")
    process_partition(args, "train")


if __name__ == '__main__':
    main() 