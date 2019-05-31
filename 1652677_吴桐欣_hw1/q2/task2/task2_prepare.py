import json
import time

from sklearn.decomposition import PCA
from math import cos, sin, atan2, sqrt, radians, degrees

CATEGORIES = {}

CATE_LIST = []

FULL_LIST = []


def get_dataset(input_filepath):
    with open(input_filepath, 'r') as f:
        dataset = json.load(f)
    return dataset


def get_test_set():
    dataset = get_dataset('./data/user_business_223699.json')
    count = 0
    part = {}
    for k in dataset.keys():
        part[k] = dataset.get(k)
        count += 1
        if count >= 5000:
            break
    with open('./data/user_business_test.json', 'w') as f:
        json.dump(part, f)


def find_geographic_center(locations):
    x = 0
    y = 0
    z = 0
    length = len(locations)
    for lon, lat in locations:
        lon = radians(float(lon))
        lat = radians(float(lat))
        x += cos(lat) * cos(lon)
        y += cos(lat) * sin(lon)
        z += sin(lat)
        x = float(x / length)
        y = float(y / length)
        z = float(z / length)
        return degrees(atan2(y, x)), degrees(atan2(z, sqrt(x * x + y * y)))


def get_category_list():
    with open('./data/category.txt') as f:
        for line in f.readlines():
            line = line.strip(' \n')
            if line != '':
                CATE_LIST.append(line)
                FULL_LIST.append(line)
                CATEGORIES[line] = line
    with open('./data/yelp_category_list.txt') as f:
        current = ''
        for line in f.readlines():
            line = line.strip(' \n')
            if line in CATE_LIST:
                current = line
            else:
                FULL_LIST.append(line)
                CATEGORIES[line] = current


def generate_user_data_pca():
    time1 = time.time()
    user_data = []
    get_category_list()
    business = get_dataset('./data/business_163665.json')
    user_business = get_dataset('./data/user_business_test.json')

    for business_list in user_business.values():
        category_list = []
        locations = []
        for bid in business_list:
            locations.append((business.get(bid).get('longitude'), business.get(bid).get('latitude')))
            for category in business.get(bid).get('categories'):
                category_list.append(category)
                category = CATEGORIES.get(category)
                if category:
                    category_list.append(category)
        user = [0] * (len(FULL_LIST) + 2)
        user_center = find_geographic_center(locations)
        user[0] = round(user_center[0], 2)
        user[1] = round(user_center[1], 2)
        for cate in category_list:
            try:
                index = FULL_LIST.index(cate)
                user[index + 2] += 1
            except ValueError:
                pass
        for i in range(len(FULL_LIST)):
            user[i + 2] = round(user[i + 2] / len(category_list) * 10, 2)
        user_data.append(user)
    print(time.time() - time1)

    pca = PCA(n_components=26)
    pca.fit(user_data)
    with open('./data/user_data_26.json', 'w') as f:
        data = {
            'data': pca.fit_transform(user_data).tolist()
        }
        json.dump(data, f)


def generate_user_data():
    print('generate_user_data', end=' ')
    time1 = time.time()
    user_data = []
    get_category_list()
    business = get_dataset('./data/business_163665.json')
    user_business = get_dataset('./data/user_business_test.json')

    for business_list in user_business.values():
        category_list = []
        locations = []
        for bid in business_list:
            locations.append((business.get(bid).get('longitude'), business.get(bid).get('latitude')))
            for category in business.get(bid).get('categories'):
                category = CATEGORIES.get(category)
                if category:
                    category_list.append(category)
        user = [0 for i in range(len(CATE_LIST)+2)]
        user_center = find_geographic_center(locations)
        user[0] = round(user_center[0], 2)
        user[1] = round(user_center[1], 2)
        for cate in category_list:
            index = CATE_LIST.index(cate)
            user[index+2] += 1
        for i in range(len(CATE_LIST)):
            user[i+2] = round(user[i+2]/len(category_list)*10, 2)
        user_data.append(user)
    print(time.time()-time1)
    with open('./data/user_data.json', 'w') as f:
        data = {
            'data': user_data
        }
        json.dump(data, f)


def main():
    # get_test_set()
    generate_user_data_pca()
    pass


if __name__ == '__main__':
    main()
