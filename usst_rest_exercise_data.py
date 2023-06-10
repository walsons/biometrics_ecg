import numpy as np
import pandas as pd
import utils


def usst_db_data_to_npy_file(non_fixed=False):
    """
    Function: generate usst_db ecg_data_list and target_list and save as .npy file\n
    """
    db_data = pd.read_excel("USSTDB/usst_db.xlsx")
    r_info = pd.read_excel("USSTDB/usst_R_peaks.xlsx", header=None, names=["pos_index", "R_peak_index"])
    r_info_length = len(r_info)
    person_heart_beats = []
    person_targets = []

    previous_segment = None
    rest_person_heart_beats = []
    rest_person_targets = []

    last_bed_number = -1
    last_pos_index = -1

    for i in range(r_info_length):
        # look progress bar
        progress = int(100 * i / r_info_length)
        if i % 50 == 0:
            print("#" * progress, str(progress) + "%")

        pos_index = r_info["pos_index"][i].split(",")
        pos_index = [int(i) for i in pos_index]


        continue_flag = False
        if int(db_data["bed_number"][pos_index[0]]) != last_bed_number:
            last_bed_number = int(db_data["bed_number"][pos_index[0]])
            last_pos_index = pos_index[0]
            continue_flag = True
        elif pos_index[0] == last_pos_index:
            continue_flag = True
        # if continue_flag:
        #     continue

        available_section_str = db_data["available section"][pos_index[0]].split(", ")[pos_index[1]]
        available_section_str = available_section_str[1: -1].split(",")
        available_section = [int(i) for i in available_section_str]
        single_data = utils.read_usst_db_data("./USSTDB/sData/" + db_data["ecg_data_file"][pos_index[0]])[available_section[0]:available_section[1]]
        r_index = r_info["R_peak_index"][i].split(",")
        r_index = [int(i) for i in r_index]

        single_data = utils.wavelet_transform_denoise(single_data)
        sampling_rate = 500
        segment = []
        if non_fixed:
            for k in range(len(r_index)):
                if k == 0 or k == len(r_index) - 1:
                    continue
                s = r_index[k + 1] - r_index[k]
                left_border = int(r_index[k] - (25 / (25 + 38)) * s)
                right_border = int(r_index[k] + (38 / (25 + 38)) * s)
                segment.append(single_data[left_border: right_border])
            # remove abnormal signal
            segment = [i for i in segment if sampling_rate * 0.1 + 200 < len(i) < sampling_rate * 1.2]
        else:
            left_range = int(250 * sampling_rate / 1000)  # 250ms
            right_range = int(380 * sampling_rate / 1000)  # 380ms
            segment = [single_data[r_index[k + 1] - left_range: r_index[k + 1] + right_range] for k in range(len(r_index) - 2)]
        if continue_flag:
            previous_segment = segment
            continue
        person_heart_beats += segment
        person_targets += [int(db_data["bed_number"][pos_index[0]])] * len(segment)

        rest_person_heart_beats += previous_segment
        rest_person_targets += [int(db_data["bed_number"][pos_index[0]])] * len(previous_segment)
    print("#" * 100, "100%")

    np.save("./xxx_exercise_usst_db_ecgdata", person_heart_beats)
    np.save("./xxx_exercise_usst_db_target", person_targets)

    np.save("./xxx_rest_usst_db_ecgdata", rest_person_heart_beats)
    np.save("./xxx_rest_usst_db_target", rest_person_targets)


# "./xxx_exercise_usst_db_ecgdata.npy"
# "./xxx_exercise_usst_db_target.npy"
# "./xxx_rest_usst_db_ecgdata.npy"
# "./xxx_rest_usst_db_target.npy"
def load_usst_db(ecgdata_path, target_path):
    """
    Fuction: get a list of ECG segments(heart beat) of USST-DB database from .npy file and corresponding target ndarray
    :return: list(np.array(), np.array(), ...), np.array(0, 0, ...,116)
    """
    usst_db_ecgdata = np.load(ecgdata_path, allow_pickle=True)
    usst_db_target = np.load(target_path, allow_pickle=True)
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


def load_usst_db_filtering(ecgdata_path, target_path):
    a, b = load_usst_db(ecgdata_path, target_path)
    x = []
    y = []
    # 46 can't calculate, and remove the signal whose quality is poor
    arr = [0, 1, 4, 15, 16, 17, 18, 20, 27, 28, 29, 30, 36, 37, 38, 39, 42, 45, 46, 47, 48, 49, 54,
           60, 61, 63, 65, 67, 74, 75, 78, 81, 83, 84, 85, 86, 88, 92, 96, 98]
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


def load_usst_db_filtering_with_num(capacity, ecgdata_path, target_path):
    ecgdata, target = load_usst_db_filtering(ecgdata_path, target_path)
    res_ecgdata = []
    res_target = []
    for a, b in zip(ecgdata, target):
        if b >= capacity:
            break
        else:
            res_ecgdata.append(a)
            res_target.append(b)
    return res_ecgdata, res_target


def load_usst_db_filtering_with_start_end(start, end, ecgdata_path, target_path):
    ecgdata, target = load_usst_db_filtering(ecgdata_path, target_path)
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
    # "./xxx_exercise_usst_db_ecgdata.npy"
    # "./xxx_exercise_usst_db_target.npy"
    # "./xxx_rest_usst_db_ecgdata.npy"
    # "./xxx_rest_usst_db_target.npy"

    # usst_db_data_to_npy_file()
    # a, b = load_usst_db_filtering_with_num(11)
    # a, b = load_usst_db_filtering_with_start_end(10, 20)
    # print(len(a), b)
    a, b = load_usst_db_filtering("./xxx_exercise_usst_db_ecgdata.npy", "./xxx_exercise_usst_db_target.npy")
    # print(a)
    print(b)
    print(len(a))
    print(len(b))
    a, b = load_usst_db_filtering("./xxx_rest_usst_db_ecgdata.npy", "./xxx_rest_usst_db_target.npy")
    # print(a)
    print(b)
    print(len(a))
    print(len(b))