import glob
import wfdb
import numpy as np
import pandas as pd
import utils
from scipy import signal


# ECG-ID
def load_ecg_id(non_fixed=False):
    """
    Fuction: get a list of ECG segments(heart beat) of ECG-ID database and corresponding target ndarray, \
    the R peak data is from manual .csv file\n
    :return: list(np.array(), np.array(), ...), np.array(0, 0, ...,89)
    """
    wildcard_file_path_list = ["ecg-id-database/Person_0" + str(i + 1) + "/*.dat" for i in range(9)]
    wildcard_file_path_list += ["ecg-id-database/Person_" + str(i + 10) + "/*.dat" for i in range(81)]
    cnt = 0  # counter
    person_heart_beats = []
    person_targets = []
    for one_wildcard_file_path in wildcard_file_path_list:
        one_directory_file_path = glob.glob(one_wildcard_file_path)
        one_directory_file_path = utils.remove_suffix(one_directory_file_path)
        for one_file in one_directory_file_path:
            ecg_data = wfdb.rdsamp(one_file)
            ecg_data = ecg_data[0]  # Two lead data, use the first
            # raw_data = ecg_data[:, 0]  # Raw ECG I
            filtered_ecg_data = ecg_data[:, 1]  # Filtered ECG I
            # Denoise again, notion: R peak that manual mark is base on the ecg data after wavelet transform denoise
            filtered_ecg_data = utils.wavelet_transform_denoise(filtered_ecg_data)
            r_peak = np.loadtxt(one_file + ".csv", dtype=int, delimiter=",")
            sampling_rate = 500

            segments = []
            if non_fixed:
                for k in range(len(r_peak)):
                    if k == 0 or k == len(r_peak) - 1:
                        continue
                    s = r_peak[k + 1] - r_peak[k]
                    left_border = int(r_peak[k] - (25 / (25 + 38)) * s)
                    right_border = int(r_peak[k] + (38 / (25 + 38)) * s)
                    segments.append(filtered_ecg_data[left_border: right_border])
                # remove abnormal signal
                segments = [i for i in segments if sampling_rate * 0.1 + 200 < len(i) < sampling_rate * 1.2]
            else:
                # Abandon first and last R peak, because it may not be able to acquire a integrated heart beat
                left_range = int(250 * sampling_rate / 1000)  # 250ms
                right_range = int(380 * sampling_rate / 1000)  # 380ms
                segments = [filtered_ecg_data[(r_peak[k + 1] - left_range):(r_peak[k + 1] + right_range)]
                            for k in range(len(r_peak) - 2)]
                # There are some length of segment not equal to others, remove them
                segments = [seg for seg in segments if len(seg) == left_range + right_range]
            person_heart_beats += segments
            person_targets += [cnt] * (len(segments))
        cnt += 1
    person_targets = np.array(person_targets)
    return person_heart_beats, person_targets


def load_mit_db(non_fixed=False):
    """
    Fuction: get a list of ECG segments(heart beat) of MIT-DB database and corresponding target ndarray
    :return: list(np.array(), np.array(), ...), np.array(0, 0, ...,47)
    """
    #     Total data contain annotation, the annotation is as follows:
    #     {'fs': 360, 'sig_len': 650000, 'n_sig': 2, 'base_date': None, 'base_time': None, 'units': ['mV', 'mV'],
    #      'sig_name': ['MLII', 'V5'], 'comments': ['69 M 1085 1629 x1', 'Aldomet, Inderal']})
    one_directory_file_path = glob.glob("MIT_BIH/*.dat")
    one_directory_file_path = utils.remove_suffix(one_directory_file_path)
    cnt = 0  # counter
    person_heart_beats = []
    person_targets = []
    for one_file in one_directory_file_path:
        ecg_data = wfdb.rdsamp(one_file)
        ecg_data = ecg_data[0]  # Two lead data
        ecg_data = ecg_data[:, 0]  # 'MLII' lead data
        annotation = wfdb.rdann(one_file, "atr")
        sampling_rate = 360
        # Since the database is not good, use part of data
        begin = 3
        # end = 103
        end = 43
        segments = []
        if non_fixed:
            for k in range(end - begin):
                if k == 0 or k == end - begin - 1:
                    continue
                s = annotation.sample[k + 1] - annotation.sample[k]
                left_border = int(annotation.sample[k] - (25 / (25 + 38)) * s)
                right_border = int(annotation.sample[k] + (38 / (25 + 38)) * s)
                segments.append(ecg_data[left_border: right_border])
            # remove abnormal signal
            segments = [i for i in segments if sampling_rate * 0.1 + 200 < len(i) < sampling_rate * 1.2]
        else:
            left_range = int(250 * sampling_rate / 1000)  # 250ms
            right_range = int(380 * sampling_rate / 1000)  # 380ms
            segments = [ecg_data[annotation.sample[i] - left_range: annotation.sample[i] + right_range]
                        for i in range(begin, end)]
        person_heart_beats += segments
        person_targets += [cnt] * (end - begin)
        cnt += 1
    person_targets = np.array(person_targets)
    return person_heart_beats, person_targets


# MIT-BIH
def load_mit_db_resample(sampling_rate=500):
    person_heart_beats, person_targets = load_mit_db()
    left_range = int(250 * sampling_rate / 1000)  # 250ms
    right_range = int(380 * sampling_rate / 1000)  # 380ms
    for index in range(len(person_heart_beats)):
        # print(person_heart_beats[index])
        person_heart_beats[index] = signal.resample(person_heart_beats[index], right_range + left_range)
        for inner_index in range(len(person_heart_beats[index])):
            person_heart_beats[index][inner_index] = round(person_heart_beats[index][inner_index], 3)
        # print(person_heart_beats[index])
        # break
    return person_heart_beats, person_targets


def usst_db_data_to_npy_file(non_fixed=False):
    """
    Function: generate usst_db ecg_data_list and target_list and save as .npy file\n
    """
    db_data = pd.read_excel("USSTDB/usst_db.xlsx")
    r_info = pd.read_excel("USSTDB/usst_R_peaks.xlsx", header=None, names=["pos_index", "R_peak_index"])
    r_info_length = len(r_info)
    person_heart_beats = []
    person_targets = []

    for i in range(r_info_length):
        # look progress bar
        progress = int(100 * i / r_info_length)
        if i % 50 == 0:
            print("#" * progress, str(progress) + "%")

        pos_index = r_info["pos_index"][i].split(",")
        pos_index = [int(i) for i in pos_index]
        available_section_str = db_data["available section"][pos_index[0]].split(", ")[pos_index[1]]
        available_section_str = available_section_str[1: -1].split(",")
        available_section = [int(i) for i in available_section_str]
        single_data = utils.read_usst_db_data("./USSTDB/sData/" + db_data["ecg_data_file"][pos_index[0]])[available_section[0]:available_section[1]]
        r_index = r_info["R_peak_index"][i].split(",")
        r_index = [int(i) for i in r_index]

        single_data = utils.wavelet_transform_denoise(single_data)
        sampling_rate = 500
        segments = []
        if non_fixed:
            for k in range(len(r_index)):
                if k == 0 or k == len(r_index) - 1:
                    continue
                s = r_index[k + 1] - r_index[k]
                left_border = int(r_index[k] - (25 / (25 + 38)) * s)
                right_border = int(r_index[k] + (38 / (25 + 38)) * s)
                segments.append(single_data[left_border: right_border])
            # remove abnormal signal
            segments = [i for i in segments if sampling_rate * 0.1 + 200 < len(i) < sampling_rate * 1.2]
        else:
            left_range = int(250 * sampling_rate / 1000)  # 250ms
            right_range = int(380 * sampling_rate / 1000)  # 380ms
            segments = [single_data[r_index[k + 1] - left_range: r_index[k + 1] + right_range] for k in range(len(r_index) - 2)]
        person_heart_beats += segments
        person_targets += [int(db_data["bed_number"][pos_index[0]])] * len(segments)
    print("#" * 100, "100%")

    np.save("./wx_usst_db_ecgdata", person_heart_beats)
    np.save("./wx_usst_db_target", person_targets)


def load_exercise_usst_db():
    """
    Fuction: get a list of ECG segments(heart beat) of USST-DB database from .npy file and corresponding target ndarray
    :return: list(np.array(), np.array(), ...), np.array(0, 0, ...,116)
    """
    usst_db_ecgdata = np.load("./wx_exercise_usst_db_ecgdata.npy", allow_pickle=True)
    usst_db_target = np.load("./wx_exercise_usst_db_target.npy", allow_pickle=True)
    id = -1
    flag = 0
    encode_target = []
    for i in usst_db_target:
        if i != flag:
            flag = i
            id += 1
        encode_target += [id]
    usst_db_ecgdata = list(usst_db_ecgdata)
    return usst_db_ecgdata, np.array(encode_target)


# USSTDB
def load_usst_db_filtering():
    a, b = load_exercise_usst_db()
    x = []
    y = []
    # 46 can't calculate, and remove the signal whose quality is poor
    arr = [0, 1, 4, 15, 16, 17, 18, 20, 27, 28, 29, 30, 36, 37, 38, 39, 42, 45, 46, 47, 48, 49, 54,
           60, 61, 63, 65, 67, 74, 75, 78, 81, 83, 84, 85, 86, 88, 92, 96, 98]
    # arr = []
    print("length of arr: ", len(arr))
    for ecg, target in zip(a, b):
        if target not in arr:
            x.append(ecg)
            y.append(target)
    #-----------------------------------------
    print(len(x), len(y))
    id = -1
    flag = -1
    encode_target = []
    for i in y:
        if i != flag:
            flag = i
            id += 1
        encode_target += [id]
    usst_db_ecgdata = list(x)
    return usst_db_ecgdata, np.array(encode_target)


def make_mix_data(dataset1, dataset2):
    d1, t1 = dataset1
    d2, t2 = dataset2
    id = -1
    flag = -1
    encode_target = []
    for i in t1:
        if i != flag:
            flag = i
            id += 1
        encode_target += [id]
    for i in t2:
        if i != flag:
            flag = i
            id += 1
        encode_target += [id]
    ecg_data = d1 + d2
    return ecg_data, np.array(encode_target)


def load_dataset_with_start_end(dataset, start, end):
    ecgdata, target = dataset
    res_ecgdata = []
    res_target = []
    for a, b in zip(ecgdata, target):
        if b >= end:
            break
        elif b >= start:
            res_ecgdata.append(a)
            res_target.append(b)
    return res_ecgdata, res_target


if __name__ == '__main__':
    # a, b = make_mix_data(load_ecg_id_with_start_end(0, 60), load_mit_db_resample_with_start_end(0, 32))
    a, b = load_usst_db_filtering()
    print(a)
    print(b)
    print(len(a))
    print(len(b))