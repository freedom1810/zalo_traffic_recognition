#load data
import pandas as pd
import numpy as np
import cv2
import time

# kFold
import random

#unzip data
import gc
import os


def unzip_data(path = ''):

    '''
    return anh đen chữ trắng    
    '''

    indices=[0,1,2,3]
    images = []
    count_image = 0

    if not os.path.exists(path + '/images'):
        os.mkdir(path + '/images')

    for i in indices:
        df =  pd.read_parquet(path + 'train_image_data_{}.parquet'.format(i)) 
        HEIGHT = 137
        WIDTH = 236
        images = df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)
        del df
        gc.collect()

        image_name = pd.read_csv(path + 'train.csv')['image_id'].values

        for img in images:
            cv2.imwrite(path + 'images/' + image_name[count_image] + '.png', 255 - img)

            count_image += 1

        del images
        gc.collect()

def kFold(path = ''):

    '''
    load dữ liệu vào sau đó shuffle luôn 
    vì dữ liệu có thể do nhiều người làm và vào thời điểm khác nhau nên phải shuffle luôn
    '''

    if not os.path.exists('fold'):
        os.mkdir('fold')

    train = pd.read_csv(path + 'train.csv')
    new_rows = [i for i in range(len(train))]
    random.seed(2020)
    random.shuffle(new_rows)
    train = train.iloc[new_rows]

    train_labels = train[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values
    train_paths = train['image_id'].values

    '''
    label đang được chia làm 3 nhãn khác nhau, gộp lại thành 1 nhãn 
    sau đó chia làm 5 fold
    '''
    labels = {}
    labels_path = {}

    for i, path in enumerate(train_paths):
        label = str(train_labels[i][0]) + '_' + str(train_labels[i][1]) + '_' + str(train_labels[i][2])

        if label not in labels:
            labels[label] = 1
            labels_path[label] = [path]
        else:
            labels[label] += 1
            labels_path[label].append(path)
            
    # chia làm 5 fold
    for fold in range(5):

        train = {'image_id': [], 'grapheme_root': [],'vowel_diacritic': [],'consonant_diacritic': []}

        test= {'image_id': [], 'grapheme_root': [],'vowel_diacritic': [],'consonant_diacritic': []}

        for i in labels:

            i_ = [int(x) for x in i.split('_')]
            n_dataset = len(labels_path[i])

            x_train = labels_path[i][0: int(n_dataset*0.2*fold)] + labels_path[i][int(n_dataset*0.2*(fold + 1)):]
            train['image_id'] += x_train

            for _ in range(len(x_train)):

                train['grapheme_root'].append(i_[0])
                train['vowel_diacritic'].append(i_[1])
                train['consonant_diacritic'].append(i_[2])

            x_test = labels_path[i][int(n_dataset*0.2*fold): int(n_dataset*0.2*(fold + 1))]
            test['image_id'] += x_test

            for _ in range(len(x_test)):

                test['grapheme_root'].append(i_[0])
                test['vowel_diacritic'].append(i_[1])
                test['consonant_diacritic'].append(i_[2])


        df_train = pd.DataFrame(train)
        new_rows = [i for i in range(len(df_train))]

        random.seed(2020)
        random.shuffle(new_rows)
        df_train = df_train.iloc[new_rows]
        df_train.to_csv('train_fold_{}.csv'.format(fold), index = False)

        df_test = pd.DataFrame(test)
        new_rows = [i for i in range(len(df_test))]

        random.seed(2020)
        random.shuffle(new_rows)
        df_test = df_test.iloc[new_rows]
        df_test.to_csv('test_fold_{}.csv'.format(fold), index = False)


def main():
    unzip_data(path = '/home/asilla/sonnh/k/ben/')
    kFold(path = '/home/asilla/sonnh/k/ben/')
    
if __name__ == "__main__":
    main()


