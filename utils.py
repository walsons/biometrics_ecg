import pywt
import numpy as np


def remove_suffix(file_path_list):
    """
    Function: get rid of every file suffix(e.g. "*dat") in file_path_list\n
    :param file_path_list: a list of file path
    :return: a list contain paths of all files that were get rid of suffix;
    """
    res = []
    for path in file_path_list:
        path = path.rsplit('.', maxsplit=1)[0]
        res.append(path)
    return res


def wavelet_transform_denoise(ecg):
    """
    Function: denoise via wavelet transform\n
    :param ecg: the ecg that need to denoise
    :return: the ecg which is denoised
    """
    coeffs = pywt.wavedec(ecg, wavelet="db5", level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    cD1.fill(0)
    cD2.fill(0)
    cA9.fill(0)
    cD9.fill(0)

    data = pywt.waverec(coeffs=coeffs, wavelet="db5")
    return data


def read_usst_db_data(file_path):
    """
    Function: read ECG data of USSTDB from file\n
    :param file_path: the path of the file that is read
    :return: data list of lead I
    """
    f = open(file_path, 'rb')
    ecg_string = f.read()
    f.close()
    ecg_data = []

    # Real ECG data segment from 4096
    for i in range(4096, len(ecg_string)):
        if i % 2 == 0:

            # The following 5 rows are the formula of value of ECG data point(num), no reason
            num = ecg_string[i + 1] % 16 * 256 + ecg_string[i]
            ecg_data.append(num)
    ecg_data = (np.array(ecg_data) - 2048) / 241
    # 8 lead, the first is II lead, the second is III lead, others are lead of chest(disconnected)
    ecg_data = ecg_data.reshape(-1, 8)
    # I lead equal to II lead subtract III lead
    lead1 = ecg_data[:, 0] - ecg_data[:, 1]

    return lead1