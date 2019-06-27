import csv
import utm
import numpy as np
from sklearn.model_selection import train_test_split
from numpy import array
from numpy import argmax
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from geopy.distance import geodesic

left_lat = 328770
left_lon = 3462224

BS_DICT = {}
"""
key -   (str)RNCID + (str)CellID
val -   ['Latitude', 'Longitude']
"""
IMEI_DICT = {}
IMSI_DICT = {}
RNCID_DICT = {}
CELLID_DICT = {}


def get_csv_rows(filepath):
    rows = []
    with open(filepath, 'r') as file:
        for row in csv.reader(file):
            rows.append(row)
    return rows


def write_csv(filepath: str, content: list):
    with open(filepath, 'a+') as file:
        writer = csv.writer(file)
        writer.writerow(content)


def trans_set_to_dict(from_set, to_dict):
    from_set = list(from_set)
    # index = [i for i in range(len(from_set))]
    # from_set_encoded = to_categorical(array(index), len(index))
    for i, ele in enumerate(from_set):
        to_dict[str(ele)] = i
        # to_dict[str(ele)] = from_set_encoded[i]
        # to_dict[str(ele)] = argmax(from_set_encoded[i])


def generate_global_bs_dict():
    print('…… generate_global_bs_dict ……')
    RNCID_set = set()
    CellID_set = set()
    with open('data/gongcan.csv', 'r') as file:
        for row in csv.reader(file):
            if row[0] == 'RNCID':
                continue
            RNCID_set.add(str(row[0]))
            CellID_set.add(str(row[1]))
            key = str(row[0]) + str(row[1])
            BS_DICT[key] = gps_to_utm(row[3], row[2])
    trans_set_to_dict(RNCID_set, RNCID_DICT)
    trans_set_to_dict(CellID_set, CELLID_DICT)


def generate_global_mr_dict():
    print('…… generate_global_mr_dict ……')
    IMEI_set = set()
    IMSI_set = set()
    with open('data/train_2g.csv', 'r') as file:
        for row in csv.reader(file):
            if row[1] == 'IMEI':
                continue
            IMEI_set.add(str(row[1]))
            IMSI_set.add(str(row[2]))
    trans_set_to_dict(IMEI_set, IMEI_DICT)
    trans_set_to_dict(IMSI_set, IMSI_DICT)


def get_bs_location(rncid, cellid):
    """
    获取基站经纬度信息
    :param rncid
    :param cellid
    :return: 经纬度
    """
    if len(BS_DICT.keys()) <= 0:
        generate_global_bs_dict()
    return BS_DICT.get(str(rncid) + str(cellid))


def get_IMEI_code(imei):
    if (len(IMEI_DICT.keys())) <= 0:
        generate_global_mr_dict()
    return IMEI_DICT.get(str(imei))


def get_IMSI_code(imsi):
    if (len(IMSI_DICT.keys())) <= 0:
        generate_global_mr_dict()
    return IMSI_DICT.get(str(imsi))


def get_RNCID_code(rncid):
    if (len(RNCID_DICT.keys())) <= 0:
        generate_global_bs_dict()
    return RNCID_DICT.get(str(rncid))


def get_CELLID_code(cellid):
    if (len(CELLID_DICT.keys())) <= 0:
        generate_global_bs_dict()
    return CELLID_DICT.get(str(cellid))


def get_test_set(start=0, end=-1, add_speed=True, add_bs_location=True):
    print('…… get_test_set ……')
    test_set = []
    with open('data/test_2g.csv', 'r') as file:
        for row in csv.reader(file):
            for i, ele in enumerate(row):
                if ele == '-999' or ele == '-1':
                    row[i] = '-1'
            if row[1] != '-1':
                row[1] = get_IMEI_code(row[1])
            if row[2] != '-1':
                row[2] = get_IMSI_code(row[2])

            for j in range(29, 0, -5):
                if add_bs_location:     # 添加经纬度
                    if row[j] == '-1' or row[j+1] == '-1':
                        row.insert(j+2, '-1')
                        row.insert(j+3, '-1')
                    else:
                        location = get_bs_location(row[j], row[j+1])
                        if location is None:
                            print("----- error : cannot get base station's location")
                            continue
                        row.insert(j+2, location[0])
                        row.insert(j+3, location[1])
                if row[j] != '-1':
                    row[j] = get_RNCID_code(row[j])
                if row[j+1] != '-1':
                    row[j+1] = get_CELLID_code(row[j+1])

            data = row[start:end]
            if add_speed:
                data.insert(4, row[-1])
            test_set.append(data)
    return test_set[1:]


def get_train_set(start=0, end=-3, add_speed=True, add_bs_location=True):
    """
        train_2g.csv 文件数据，可选使用哪几列
        数据格式：（python list）
                ['IMEI', 'IMSI', 'MRTime',
                    'RNCID_1', 'CellID_1', 'Dbm_1', 'AsuLevel_1', 'SignalLevel_1',
                    'RNCID_2', ...,
                'Longitude', 'Latitude', 'Speed']
    :param add_bs_location:
    :param add_speed:
    :param start:
    :param end:
    :return:
    """
    print('…… get_train_set ……')
    train_set = []
    if start > 3 or end < -5:
        add_bs_location = False
    with open('data/train_2g.csv', 'r') as file:
        for row in csv.reader(file):
            if row[0] == 'TrajID':
                continue
            # if '-999' in row or '-1' in row:
            #     continue
            count = 0
            for i, ele in enumerate(row):
                if ele == '-999' or ele == '-1':
                    row[i] = '-1'
                    count += 1
            if count >= 10:
                continue
            if row[1] != '-1':
                row[1] = get_IMEI_code(row[1])
            if row[2] != '-1':
                row[2] = get_IMSI_code(row[2])
            utm_location = gps_to_utm(row[-5], row[-4])
            row[-5] = utm_location[0]
            row[-4] = utm_location[1]

            for j in range(29, 0, -5):
                if add_bs_location:     # 添加经纬度
                    if row[j] == '-1' or row[j+1] == '-1':
                        row.insert(j+2, '-1')
                        row.insert(j+3, '-1')
                    else:
                        location = get_bs_location(row[j], row[j+1])
                        if location is None:
                            print("----- error : cannot get base station's location")
                            continue
                        row.insert(j+2, location[0])
                        row.insert(j+3, location[1])
                if row[j] != '-1':
                    row[j] = get_RNCID_code(row[j])
                if row[j+1] != '-1':
                    row[j+1] = get_CELLID_code(row[j+1])

            data = row[start:end]
            if add_speed:
                data.insert(4, row[-1])
            train_set.append(data)
    return train_set[1:]


def split_dataset(features, labels):
    print('…… split_dataset ……')
    parts = []
    group = []
    for i in range(4):
        parts.append([[], []])
    a_features, b_features, a_labels, b_labels = \
        train_test_split(features, labels, test_size=0.5, random_state=0)
    parts[0][0], parts[1][0], parts[0][1], parts[1][1] = \
        train_test_split(a_features, a_labels, test_size=0.5, random_state=0)
    parts[2][0], parts[3][0], parts[2][1], parts[3][1] = \
        train_test_split(b_features, b_labels, test_size=0.5, random_state=0)

    for i in range(4):
        g = {
            'train_features': [],
            'train_labels': [],
            'test_features': [],
            'test_labels': []
        }
        for j in range(4):
            if i == j:
                g['test_features'] = list(parts[i][0])
                g['test_labels'] = list(parts[i][1])
            else:
                g['train_features'] += list(parts[j][0])
                g['train_labels'] += list(parts[j][1])
        group.append(g)
    return group


def generate_pred_csv(q, locations):
    filepath = 'out/q' + str(q) + '/pred.csv'
    with open(filepath, 'w+') as file:
        writer = csv.writer(file)
        writer.writerow(['Longitude', 'Latitude'])
        for point in locations:
            writer.writerow([point[0], point[1]])


def visualize(q, location):
    filepath = 'out/q' + str(q) + '/pred.js'
    with open(filepath, 'w+') as file:
        file.write('pred = [')
        for i, point in enumerate(location):
            file.write('[' + str(point[0]) + ',' + str(point[1]) + ']')
            if i+1 != len(location):
                file.write(',')
        file.write('];')


def get_distance(lon_1, lat_1, lon_2, lat_2):
    latlon_1 = (lat_1, lon_1)
    latlon_2 = (lat_2, lon_2)
    return geodesic(latlon_1, latlon_2).meters


def get_metrics(predictions, truth):
    metric = {}
    errors = []
    for i in range(len(truth)):
        if predictions[i] is None:
            continue
        errors.append(get_distance(predictions[i][0], predictions[i][1], truth[i][0], truth[i][1]))
    errors.sort()
    metric['mean'] = np.mean(np.asarray(errors))
    metric['middle'] = errors[int(len(errors)/2)]
    metric['p90'] = errors[int(len(errors) * 0.9)]
    metric['p80'] = errors[int(len(errors) * 0.8)]
    metric['p70'] = errors[int(len(errors) * 0.7)]
    plt.hist(errors, bins='auto', rwidth=0.9)
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.show()
    print(metric)
    return metric


def utm_to_gird(longitude, latitude):
    lat = int(latitude) - left_lat
    lon = int(longitude) - left_lon
    lat /= 10
    lon /= 10
    y = int(lat)
    x = int(lon)
    x = 0 if x < 0 else x
    y = 0 if y < 0 else y

    if lat == int(lat) and y >= 1:
        y -= 1
    if lon == int(lon) and x >= 1:
        x -= 1
    return x, y


def grid_to_gps(lon, lat):
    lon = left_lon + int(lon) * 10 - 5
    lat = left_lat + int(lat) * 10 - 5
    gps = utm.to_latlon(lat, lon, 51, 'U')
    return gps[1], gps[0]


def gps_to_utm(longitude, latitude):
    utm_point = utm.from_latlon(float(latitude), float(longitude))
    return round(utm_point[1]), round(utm_point[0])


def utm_to_gps(lon, lat):
    try:
        gps = utm.to_latlon(lat, lon, 51, 'U')
    except:
        print('error:', lon, lat)
        return None
    return gps[1], gps[0]


def utm_labels_to_gps(labels):
    for i, pred in enumerate(labels):
        labels[i] = utm_to_gps(pred[0], pred[1])
    return labels


def transform_gps_labels(labels, reverse=False):
    if not reverse:
        for i, pred in enumerate(labels):
            lon = (pred[0] - 121) * 10000
            lat = (pred[1] - 31) * 10000
            labels[i] = [lon, lat]
    else:
        for i, pred in enumerate(labels):
            lon = (pred[0] / 10000) + 121
            lat = (pred[1] / 10000) + 31
            labels[i] = [lon, lat]
    return labels

# def gps_labels_to_utm(labels):
#     for i, pred in enumerate(labels):
#         labels[i] = utm_to_gps(pred[0], pred[1])
#     return labels
