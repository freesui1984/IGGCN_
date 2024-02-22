from glob2 import glob
import numpy as np
import mne
import os

file_path = 'E:\\Datasets\\TUSZ\\edf\\*\\*\\*\\*\\*\\*.edf'
edf_file_list = glob(file_path)

# extract subject IDs from the file path, create python set to extract unique elements from list, convert to list again
unique_epilepsy_patient_ids = list(set([x.split("\\")[-1].split("_")[0] for x in edf_file_list]))

subjects_name_file_path = r'E:\Datasets\TUSZ\corpus_subjects.txt'
with open(subjects_name_file_path, 'w') as file_handler:
    for item in unique_epilepsy_patient_ids:
        file_handler.write("{}\n".format(item))

edf_file_list = glob(file_path)
channel_configs = [x.split("\\")[6] for x in edf_file_list]

f = open(subjects_name_file_path, 'r')
unique_epilepsy_patient_ids = f.readlines()
unique_epilepsy_patient_ids = [x.strip() for x in unique_epilepsy_patient_ids]

SAMPLING_FREQ = 200.0
WINDOW_LENGTH_SECONDS = 12.0
WINDOW_LENGTH_SAMPLES = int(WINDOW_LENGTH_SECONDS * SAMPLING_FREQ)
raw_label = {"fnsz": 0, "spsz": 1, "cpsz": 2, "gnsz": 3, "absz": 4, "tnsz": 5, "tcsz": 6, "mysz": 7}
label_mapping = {"cfsz": 0, "gnsz": 1, "absz": 2, "ctsz": 3, "no_seiz": 4}
label_count = {label: 0 for label in label_mapping.keys()}

dataset_index_rows = []

def get_seizure_times(file_name):
    seizure_times = []
    with open(file_name) as f:
        for line in f.readlines():
            for seiz_label in raw_label.keys():
                if seiz_label in line:
                    parts = line.strip().split(" ")
                    if seiz_label in ["fnsz", "spsz", "cpsz"]:
                        seiz_label = "cfsz"
                    elif seiz_label in ["tnsz", "tcsz", "mysz"]:
                        seiz_label = "ctsz"
                    seizure_times.append([
                        float(parts[0]),
                        float(parts[1]),
                        seiz_label
                    ])
    return seizure_times


for idx, patient_id in enumerate(unique_epilepsy_patient_ids):
    labels = []
    print(f"{patient_id} : {idx + 1}/{len(unique_epilepsy_patient_ids)}")

    patient_edf_file_list = glob(f"E:\\hy\\Datasets\\TUSZ\\edf\\*\\*\\*\\*\\*\\{patient_id}_*.edf")
    assert len(patient_edf_file_list) >= 1

    print(len(patient_edf_file_list))

    for raw_file_path in patient_edf_file_list:
        raw_data = mne.io.read_raw_edf(raw_file_path, verbose=False, preload=False)
        file_name = raw_file_path.split('.edf')[0] + ".tse"
        seizure_times = get_seizure_times(file_name)

        for start_sample_index in range(0, int(int(raw_data.times[-1]) * SAMPLING_FREQ), WINDOW_LENGTH_SAMPLES):
            end_sample_index = start_sample_index + (WINDOW_LENGTH_SAMPLES - 1)

            if end_sample_index > int(int(raw_data.times[-1]) * SAMPLING_FREQ):
                break
            label = "no_seiz"
            for t in seizure_times:
                start_t, end_t, seiz_label = t
                if not ((end_sample_index <= start_t * SAMPLING_FREQ) or (start_sample_index >= end_t * SAMPLING_FREQ)):
                    label = seiz_label
                    break

            labels.append(label)
            label_count[label] += 1

            row = {
                "patient_ID": str(patient_id),
                "raw_file_path": raw_file_path,
                "record_length_seconds": raw_data.times[-1],
                "sampling_freq": SAMPLING_FREQ,
                "channel_config": raw_file_path.split("\\")[6],
                "start_sample_index": start_sample_index,
                "end_sample_index": end_sample_index,
                "label": label,
                "numeric_label": label_mapping[label]
            }
            dataset_index_rows.append(row)

    print(len(labels))
    print(labels)

df = pd.DataFrame(dataset_index_rows, columns=["patient_ID", "raw_file_path", "record_length_seconds",
                                               "sampling_freq", "channel_config", "start_sample_index",
                                               "end_sample_index", "label", "numeric_label"])
df.to_csv("F:\\IGECN\\data\\tusz_eeg\\4_classcorpus_window_index_yu.csv", index=False, encoding='utf_8_sig')
print("\nSucceed!!!")
print(label_count["no_seiz"])
print(label_count["cfsz"])
print(label_count["gnsz"])
print(label_count["absz"])
print(label_count["ctsz"])

