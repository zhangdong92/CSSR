import torch.utils.data as data
from PIL import Image
import os
import os.path
import random


class MriSliceDataset(data.Dataset):
    """dataset for CSSR"""


    def __init__(self, datasetRootPath, transform=None, originSize=None, cropTargetSize=None, isTrainOrVal=True):

        self.datasetRootPath=datasetRootPath
        imagePathList = getAllImageList(datasetRootPath)
        self.imagePathList = imagePathList
        self.transform = transform
        self.cropTargetSize = cropTargetSize
        self.train = isTrainOrVal
        self.originSize=originSize
        self.cropTargetSize=cropTargetSize


    def __getitem__(self, index):
        """ 
        generate random i, return list(S0 S1 Si) and i
        """
        resultSliceList = []
        sliceCount=len(self.imagePathList[index])
        
        if self.train:
            cropX = random.randint(0, self.originSize[0] - self.cropTargetSize[0])
            cropY = random.randint(0, self.originSize[1] - self.cropTargetSize[1])
            cropArea = (cropX, cropY, cropX + self.cropTargetSize[0], cropY + self.cropTargetSize[1])


            randomNum=(random.randint(0, 2))
            if randomNum==0:
                # 9 slice
                slice0 = random.randint(0, sliceCount-9)
                slice_i = random.randint(slice0 + 1, slice0 + 7)
                if (random.randint(0, 1)):
                    selectSliceList = [slice0, slice_i, slice0 + 8]
                    returnI = slice_i - slice0 - 1
                else:
                    selectSliceList = [slice0 + 8, slice_i, slice0]
                    returnI = slice0 - slice_i + 7
            elif randomNum==1:
                # 3 slice
                slice0 = random.randint(0, sliceCount-3)
                slice_i = slice0 + 1
                if (random.randint(0, 1)):
                    selectSliceList = [slice0, slice_i, slice0 + 2]
                    returnI = 3
                else:
                    selectSliceList = [slice0 + 2, slice_i, slice0]
                    returnI = 3
            else:
                # 5 slice
                slice0 = random.randint(0, sliceCount-5)
                slice_i = random.randint(slice0 + 1, slice0 + 3)
                if (random.randint(0, 1)):
                    selectSliceList = [slice0, slice_i, slice0 + 4]
                    returnI = (slice_i - slice0-1)*2 + 1
                else:
                    selectSliceList = [slice0 + 4, slice_i, slice0]
                    returnI = (slice0 - slice_i + 3)*2+1


            shouldFlip = random.randint(0, 1)==1

        else:
            # val, 3 slice
            cropArea = (0, 0, self.cropTargetSize[0], self.cropTargetSize[1])
            slice0 = index%(sliceCount-2)
            slice_i = slice0+1
            selectSliceList = [slice0, slice_i, slice0 + 2]
            returnI = 3

            shouldFlip = False
        
        # for every slice index(count=3)
        for index2 in selectSliceList:
            image = openImageWithAugmentation(self.imagePathList[index][index2], cropParam=cropArea, shouldFlip=shouldFlip)

            # # 3 channel to 1 channel
            # image=mask_pic_color_replace.replace_color_3to1(image)

            if self.transform is not None:
                image = self.transform(image)
            resultSliceList.append(image)
            
        return resultSliceList, returnI


    def __len__(self):
        return len(self.imagePathList)

    def __repr__(self):
        return self.datasetRootPath


def getAllImageList(datasetPath):
    """
    get all file path in dataset, resultList[i][j] means file No. j in dir No. i
    """
    resultList = []
    # for every dir
    for index, dir in enumerate(os.listdir(datasetPath)):
        dirPath = os.path.join(datasetPath, dir)
        resultList.append([])
        fileList = os.listdir(dirPath)
        fileList = sorted(fileList)
        # for every image file in one dir
        for fileName in fileList:
            filePath = os.path.join(dirPath, fileName)
            resultList[index].append(filePath)
    return resultList


def openImageWithAugmentation(path, resizeWh=None, cropParam=None, shouldFlip=False):
    """
    open image, with augmentation by params
    """

    with open(path, 'rb') as f:
        img = Image.open(f)

        if resizeWh != None:
            img = img.resize(resizeWh, Image.ANTIALIAS)

        if cropParam != None:
            img = img.crop(cropParam)

        if shouldFlip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img
