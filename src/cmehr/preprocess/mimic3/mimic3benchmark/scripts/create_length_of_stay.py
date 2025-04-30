import os
import argparse
import numpy as np
import pandas as pd
import random
random.seed(49297)
from tqdm import tqdm

"""
LOS是完整的ICU住院时长
"""

def process_partition(args, partition, sample_rate=1.0, shortest_length=4.0, eps=1e-6):
    # Build and Make Directory
    output_dir = os.path.join(args.output_path, partition)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    xty_triples = []
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path, partition))))
    for patient in tqdm(patients, desc='Iterating over patients in {}'.format(partition)):
        patient_folder = os.path.join(args.root_path, partition, patient)
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))

        for ts_filename in patient_ts_files:
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                lb_filename = ts_filename.replace("_timeseries", "")
                label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))

                # empty label file
                if label_df.shape[0] == 0:
                    print("\n\t(empty label file)", patient, ts_filename)
                    continue

                los = 24.0 * label_df.iloc[0]['Length of Stay']  # in hours
                if pd.isnull(los):
                    print("\n\t(length of stay is missing)", patient, ts_filename)
                    continue

                ts_lines = tsfile.readlines()
                # Header : names of categories
                header = ts_lines[0]
                # TS : values of these categories
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]

                ts_lines = [line for (line, t) in zip(ts_lines, event_times)
                            if -eps < t < los + eps]
                event_times = [t for t in event_times
                               if -eps < t < los + eps]

                # no measurements in ICU
                if len(ts_lines) == 0:
                    print("\n\t(no events in ICU) ", patient, ts_filename)
                    continue
                """
                Let's assume:
                
                los = 48.0 (Length of Stay: 48 hours)
                
                eps = 1e-6 (a very small number to avoid rounding errors)
                
                sample_rate = 1.0 (hourly samples)
                
                shortest_length = 4.0 (minimum 4 hours after admission)
                
                event_times = [1.5, 3.2, 7.8, 12.5, 24.3, 36.1] (when measurements were actually taken)
                """
                # Step 1: Create a regular time grid
                sample_times = np.arange(0.0, los + eps, sample_rate)

                # Step 2: Remove early timepoints
                sample_times = list(filter(lambda x: x > shortest_length, sample_times))

                # Step 3: Make sure we start after first measurement
                # At least one measurement
                sample_times = list(filter(lambda x: x > event_times[0], sample_times))

                output_ts_filename = patient + "_" + ts_filename
                with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                    outfile.write(header)
                    for line in ts_lines:
                        outfile.write(line)

                for t in sample_times:
                    xty_triples.append((output_ts_filename, t, los - t)) # “剩余住院时长”：los - t）。

    print("Number of created samples:", len(xty_triples))
    if partition == "train":
        random.shuffle(xty_triples)
    if partition == "test":
        xty_triples = sorted(xty_triples)

    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write('stay,period_length,y_true\n')
        for (x, t, y) in xty_triples:
            listfile.write('{},0,{:.4f}\n'.format(x, t, y))


def main():
    parser = argparse.ArgumentParser(description="Create data for length of stay prediction task.")
    parser.add_argument('root_path', type=str, help="Path to root folder containing train and test sets.")
    parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process_partition(args, "test")
    process_partition(args, "train")


if __name__ == '__main__':
    main()
