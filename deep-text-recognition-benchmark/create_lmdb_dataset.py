""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

import fire
import os
import lmdb
import cv2

import numpy as np


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(inputPath, gtFile, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    # env = lmdb.open(outputPath, map_size=1099511627776) # 1TB
    # env = lmdb.open(outputPath, map_size=10737418240) # 10GB
    # env = lmdb.open(outputPath, map_size=5368709120) # 5GB
    # env = lmdb.open(outputPath, map_size=35433480192) # 33GB
    # env = lmdb.open(outputPath, map_size=16106127360) # 15GB
    # env = lmdb.open(outputPath, map_size=18253611008) # 17GB
    env = lmdb.open(outputPath, map_size=3221225472) # 3GB
    # env = lmdb.open(outputPath, map_size=2147483648) # 2GB
    # env = lmdb.open(outputPath, map_size=1073741824) # 1GB
    # env = lmdb.open(outputPath, map_size=30064771072) # 30GB
    # env = lmdb.open(outputPath, map_size=4294967296) # 4GB
    # env = lmdb.open(outputPath, map_size=23622320128) # 22GB
    
    
    

    
    cache = {}
    cnt = 1

    try:
        with open(gtFile, 'r', encoding='utf-8') as data:
            datalist = data.readlines()
    except:
        with open(gtFile, 'r') as data:
            datalist = data.readlines()
    nSamples = len(datalist)
    for i in range(nSamples):
        imagePath, label = datalist[i].strip('\n').split('\t')
        imagePath = os.path.join(inputPath, imagePath)

        # # only use alphanumeric data
        # if re.search('[^a-zA-Z0-9]', label):
        #     continue

        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except:
                print('error occured', i)
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('%s-th image data occured error\n' % str(i))
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    fire.Fire(createDataset)
