#model pred and 3d reconstruction

from io import StringIO

import torch
import torchvision.transforms as transforms
from PIL import Image

import dcmOperate
import deformationFieldOperate
import dice_loss_tool
import model
import numpy as np

import srWeights
from logger import log
import mask_pic_color_replace
import cv2
import os


import generateMesh
import torch.nn.functional as F

predImgSize=(544,544) 

torch.set_printoptions(sci_mode=False)

def create_dir(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)  
        log.info("makedirs path ={}".format(path))





if torch.cuda.is_available():
    device = torch.device("cuda:0")
    
    log.info("use cuda")
else:
    device = torch.device("cpu")
    
    log.info("won't use cuda")


unet1 = model.UNet1(6, 4)
unet2 = model.UNet2(20, 5)
unet1.to(device)
unet2.to(device)



# deformation field
valDfModule = deformationFieldOperate.DeformationFieldApplyModule(predImgSize[0], predImgSize[1], device)
valDfModule = valDfModule.to(device)



mean = [0.485, 0.456, 0.406]
std  = [1, 1, 1]
normalize = transforms.Normalize(mean=mean, std=std)
transform = transforms.Compose([transforms.ToTensor(), normalize])


def openImage(path, resizeDim=None):
    with open(path, 'rb') as f:
        img = Image.open(f)

        if resizeDim != None:
            img = img.resize(resizeDim, Image.NEAREST)
        return img

def readImage2Tensor(path):
    
    image = openImage(path, resizeDim=predImgSize)

    
    imageTensor = transform(image)
    return imageTensor


def getAlphaWeight(unet2out):
    alpha0 = F.sigmoid(unet2out[:, 4:5, :, :])
    alpha1 = 1 - alpha0
    return alpha0,alpha1

def modelPredOnce(slice0, slice1, insertSliceIndex, saveFilePath):
    slice0=slice0.unsqueeze(0)
    slice1=slice1.unsqueeze(0)

    slice0 = slice0.to(device)
    slice1 = slice1.to(device)

    unet1Out = unet1(torch.cat((slice0, slice1), dim=1))
    DF_10 = unet1Out[:, :2, :, :]
    DF_01 = unet1Out[:, 2:, :, :]

    insertSliceIndex=torch.tensor([insertSliceIndex])
    dfWeight = srWeights.calcDfWeight(insertSliceIndex, device)

    DF_0i_hat = dfWeight[0] * DF_10 + dfWeight[1] * DF_01
    DF_1i_hat = dfWeight[2] * DF_10 + dfWeight[3] * DF_01

    slice_i_0_hat = valDfModule(slice0, DF_0i_hat)
    slice_i_1_hat = valDfModule(slice1, DF_1i_hat)

    unet2out = unet2(
        torch.cat((slice0, slice1, DF_10, DF_01, DF_1i_hat, DF_0i_hat, slice_i_1_hat, slice_i_0_hat), dim=1))

    DF_0i = unet2out[:, :2, :, :] + DF_0i_hat
    DF_1i = unet2out[:, 2:4, :, :] + DF_1i_hat
    alpha0, alpha1 = getAlphaWeight(unet2out)

    slice_i_0 = valDfModule(slice0, DF_0i)
    slice_i_1 = valDfModule(slice1, DF_1i)

    fusionWeight = srWeights.calcFusionWeight(insertSliceIndex, device)

    slice_i_pred = (fusionWeight[0] * alpha0 * slice_i_0 + fusionWeight[1] * alpha1 * slice_i_1) / (
                fusionWeight[0] * alpha0 + fusionWeight[1] * alpha1)

    
    pred_gray = dice_loss_tool.rgb_to_grayscale_mapping(slice_i_pred, colors)

    pred_gray0=pred_gray[0]
    img2=mask_pic_color_replace.pic_replace_color(pred_gray0.cpu())
    cv2.imwrite(saveFilePath, img2)
    return



def pred(labelDir, saveDir):
    

    
    fileNames=os.listdir(labelDir)
    fileNames=[a for a in fileNames if a.lower().endswith(".png")]
    fileNames=sorted(fileNames)
    filePathList=[os.path.join(labelDir, a) for a in fileNames]
    
    imgTensorList=[]
    for path in filePathList:
        imgTensor=readImage2Tensor(path)
        imgTensorList.append(imgTensor)

    create_dir(saveDir)

    
    for index in range(len(imgTensorList)-1):
        slice0=imgTensorList[index]
        slice1=imgTensorList[index+1]

        slice0NewNo=index*8+1
        slice0NewPath = os.path.join(saveDir,"{:04d}.png".format(slice0NewNo) )
        slice0OldPath = filePathList[index]
        log.info("--input image resize save to {}, from {}".format(slice0NewPath,slice0OldPath ))
        
        openImage(slice0OldPath, resizeDim=predImgSize).save(slice0NewPath)

        for insertIndex in range(7):
            
            saveFilePath = os.path.join(saveDir,"{:04d}.png".format(slice0NewNo + insertIndex + 1) )
            log.info("start model pred/insert value, after {} , result={}".format(slice0NewNo, saveFilePath) )
            modelPredOnce(slice0, slice1, insertIndex, saveFilePath) 


    sliceLastNewNo = (len(imgTensorList)-1) * 8 + 1
    sliceLastNewPath = os.path.join(saveDir, "{:04d}.png".format(sliceLastNewNo))
    sliceLastOldPath = filePathList[-1]
    log.info("--input image resize save to {}, from {}".format(sliceLastNewPath, sliceLastOldPath))
    openImage(sliceLastOldPath, resizeDim=predImgSize).save(sliceLastNewPath)
    return





def loadWeightFile(ptPath):
    
    log.info("load weight file pt,path={}".format(ptPath))
    stateDict = torch.load(ptPath, map_location=device)
    
    unet1.load_state_dict(stateDict['unet1'])
    unet2.load_state_dict(stateDict['unet2'])

def png_2_3d(labelImgDir, needRgb, zMulti, result_path):
    file_list = sorted([file for file in os.listdir(labelImgDir) if file.lower().endswith('.png')])
    log.info("png_2_3d labelImgDir={}  file count={}".format(labelImgDir, len(file_list)))

    
    pcdata = StringIO()

    
    for z, filename in enumerate(file_list):
        image_path = os.path.join(labelImgDir, filename)
        image = Image.open(image_path)
        
        image_np = np.array(image)

        
        match_mask = np.all(image_np[:, :, :3] == needRgb, axis=-1)
        
        y_indices, x_indices = np.nonzero(match_mask)

        
        points = np.stack((x_indices, y_indices, np.full(x_indices.shape, z * zMulti)), axis=-1)
        if len(points)>0:
            log.info("file={},  size={}".format(filename, len(points)))
        
        for point in points:
            r, g, b = needRgb
            pcdata.write(f"{point[0]};{point[1]};{point[2]};{r};{g};{b}\n")

    
    pcstr = pcdata.getvalue()
    pcdata.close()

    generateMesh.writeTxtFile(pcstr, result_path)


def createMesh( label8xDirPath, pcDirPath, zMulti, meshDirPath):
    

    
    png_2_3d(label8xDirPath,needRgb=(255,0,0), zMulti=zMulti, result_path=os.path.join(pcDirPath,"1wall.txt"))
    png_2_3d(label8xDirPath,needRgb=(0,255,0), zMulti=zMulti, result_path=os.path.join(pcDirPath,"2cavity.txt"))
    png_2_3d(label8xDirPath,needRgb=(0,0,255), zMulti=zMulti, result_path=os.path.join(pcDirPath,"3leiomyosarcoma.txt"))
    png_2_3d(label8xDirPath,needRgb=(255,255,0), zMulti=zMulti, result_path=os.path.join(pcDirPath,"4naevus.txt"))


    
    layerInfoList=[]
    isEmpty=generateMesh.mesh_rebuild(os.path.join(pcDirPath,"1wall.txt"),os.path.join(meshDirPath,"1wall.ply"),[1,0,0],False)
    layerInfoList.append([isEmpty, "wall","1wall.ply"])
    isEmpty=generateMesh.mesh_rebuild(os.path.join(pcDirPath,"2cavity.txt"),os.path.join(meshDirPath,"2cavity.ply"),[0,1,0],False)
    layerInfoList.append([isEmpty, "cavity","2cavity.ply"])
    isEmpty=generateMesh.mesh_rebuild(os.path.join(pcDirPath,"3leiomyosarcoma.txt"),os.path.join(meshDirPath,"3leiomyosarcoma.ply"),[0,0,1],False)
    layerInfoList.append([isEmpty, "leiomyosarcoma","3leiomyosarcoma.ply"])
    isEmpty=generateMesh.mesh_rebuild(os.path.join(pcDirPath,"4naevus.txt"),os.path.join(meshDirPath,"4naevus.ply"),[1,1,0],False)
    layerInfoList.append([isEmpty, "naevus","4naevus.ply"])

    log.info("generate ply file finish finish, path={}".format(meshDirPath))





colors = torch.tensor([
    [0, 0, 0],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0]
], dtype=torch.float32) 




num_classes = len(colors)

colors = (colors / 255.0 - torch.tensor(mean)[None, :]) / torch.tensor(std)[None, :]
colors = colors.to(device)
log.info("colors ={}".format(colors))


if __name__ == '__main__':

    # 1. load model
    ptPath="/mnt/pythonWorkspace/CSSR/pt/CSSR_epoch124_dice0.8072.pt"
    loadWeightFile(ptPath)

    dirs = os.listdir("/mnt/pythonResources/dataset/shouyi/paperData/mask")
    dirs=[d for d in dirs if d.startswith("UMD")]
    log.info("dirs ={}".format(dirs))

    for dir in dirs:
        # 2. pred, sr
        maskDir= "/mnt/pythonResources/dataset/shouyi/paperData/mask/{}".format(dir)
        maskSr8xDir = maskDir.replace("mask", "mask8x")
        dcmDir= "/mnt/pythonResources/dataset/shouyi/paperData/dcm/{}".format(dir)

        log.info("start pred and rebuild mesh, maskDir={}".format(maskDir))
        pred(maskDir, maskSr8xDir)

        # 3. generate 3d mesh model
        dcmPath=os.path.join(dcmDir, os.listdir(dcmDir)[0])

        pixel_mm, slice_mm = dcmOperate.get_dcm_spacing_mm(dcmPath)
        zMulti8x=(slice_mm/pixel_mm)/8
        log.info("dcmPath ={} zMulti8x={}( ({}/{})/8 ), *8={}".format(dcmPath, zMulti8x, slice_mm, pixel_mm, slice_mm/pixel_mm))

        pcDir=maskDir.replace("mask", "pc")
        meshDir=maskDir.replace("mask", "mesh")
        createMesh(maskSr8xDir,pcDir, zMulti8x, meshDir)
