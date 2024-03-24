# CSSR model train
import argparse
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import deformationFieldOperate
import dice_loss_tool
import model
import dataloader
import datetime
from tensorboardX import SummaryWriter
import numpy as np
from logger import log
import mask_pic_color_replace
import cv2
import os

isDev = False
# isDev=True
exp_name = "exp{}{}".format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'), "_dev" if isDev else "")
curEpoch = -1

l1LossWeight = 1

diceLossWeight = 0.5


bestDice = -1
bestDiceEpoch = -1




datasetPngSizeDim = (560, 560)
trainCropSize = (384, 384)
valCropSize = (544, 544)

log.info("exp_name ={}".format(exp_name))

torch.set_printoptions(sci_mode=False)


def create_dir(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)
        log.info("makedirs path ={}".format(path))




parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1000, )
parser.add_argument("--train_batch_size", type=int, default=16)
parser.add_argument("--val_batch_size", type=int, default=1)

parser.add_argument("--origin_learning_rate", type=float, default=0.0001)
parser.add_argument("--lr_scheduler_milestones", type=list, default=[40, 80, 180])

parser.add_argument("--dataset_path", type=str, required=False, default="" )

parser.add_argument("--weight_save_dir_path", type=str, required=False, default=r"train/{}".format(exp_name))
parser.add_argument("--save_weight_epoch_interval", type=int, default=20)
args = parser.parse_args()


dataWriter = SummaryWriter('log/{}'.format(exp_name))


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    num_workers = 6
    log.info("use cuda, num_workers={}".format(num_workers))
else:
    device = torch.device("cpu")
    num_workers = 0
    log.info("won't use cuda, num_workers={}".format(num_workers))

unet1 = model.UNet1(6, 4)
unet2 = model.UNet2(20, 5)
unet1.to(device)
unet2.to(device)

# deformation field
trainDfModule = deformationFieldOperate.DeformationFieldApplyModule(trainCropSize[0], trainCropSize[1], device)
trainDfModule = trainDfModule.to(device)
valDfModule = deformationFieldOperate.DeformationFieldApplyModule(valCropSize[0], valCropSize[1], device)
valDfModule = valDfModule.to(device)



mean = [0.485, 0.456, 0.406]
std = [1, 1, 1]
normalize = transforms.Normalize(mean=mean, std=std)
transform = transforms.Compose([transforms.ToTensor(), normalize])

trainset = dataloader.MriSliceDataset(args.dataset_path + '/train', transform, datasetPngSizeDim, trainCropSize, True)
validationset = dataloader.MriSliceDataset(args.dataset_path + '/validation', transform, datasetPngSizeDim, valCropSize, False)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=num_workers) 
validationloader = torch.utils.data.DataLoader(validationset, batch_size=args.val_batch_size, shuffle=False)



L1_lossFn = nn.L1Loss()

params = list(unet2.parameters()) + list(unet1.parameters())
optimizer = optim.Adam(params, lr=args.origin_learning_rate)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler_milestones, gamma=0.3)


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

def imgShowGray(slice, saveFileName=None):
    img = slice

    if saveFileName:
        img2 = mask_pic_color_replace.pic_replace_color(img.cpu())
        saveDir = "saveImg/{}".format(exp_name)
        create_dir(saveDir)
        cv2.imwrite(os.path.join(saveDir, saveFileName), img2)
        return

    import matplotlib.pyplot as plt
    plt.imshow(img, cmap='gray', vmin=0, vmax=4)
    plt.colorbar() 

    plt.show()

def getAlphaWeight(unet2out):
    alpha0 = F.sigmoid(unet2out[:, 4:5, :, :])
    alpha1 = 1 - alpha0
    return alpha0,alpha1



def main():
    # train loop, for every epoch
    for epoch in range( args.epochs):
        log.info("exp_name={} Epoch: {}, bestDice(epoch) ={:.4f}\t{}".format(exp_name, epoch, bestDice, bestDiceEpoch ))
        curEpoch = epoch

        lossList = []
        l1LossList = []
        trainDiceLossList = []
        trainDiceList = []

        trainloader_length = len(trainloader)
        for indexInEpoch, (trainSliceList, trainSliceIndex) in enumerate(trainloader, 0):
            # 1. train
            slice0, slice_i, slice1 = trainSliceList

            slice0 = slice0.to(device)
            slice1 = slice1.to(device)
            # GT
            slice_i = slice_i.to(device)

            optimizer.zero_grad()

            unet1Out = unet1(torch.cat((slice0, slice1), dim=1))

            DF_10 = unet1Out[:, :2, :, :]
            DF_01 = unet1Out[:, 2:, :, :]

            dfWeight = model.calcDfWeight(trainSliceIndex, device)

            DF_0i_hat = dfWeight[0] * DF_10 + dfWeight[1] * DF_01
            DF_1i_hat = dfWeight[2] * DF_10 + dfWeight[3] * DF_01

            slice_i_0_hat = trainDfModule(slice0, DF_0i_hat)
            slice_i_1_hat = trainDfModule(slice1, DF_1i_hat)

            unet2out = unet2(torch.cat((slice0, slice1, DF_10, DF_01, DF_1i_hat, DF_0i_hat, slice_i_1_hat, slice_i_0_hat), dim=1))

            DF_0i = unet2out[:, :2, :, :] + DF_0i_hat
            DF_1i = unet2out[:, 2:4, :, :] + DF_1i_hat
            alpha0,alpha1=getAlphaWeight(unet2out)

            slice_i_0 = trainDfModule(slice0, DF_0i)
            slice_i_1 = trainDfModule(slice1, DF_1i)

            fusionWeight = model.calcFusionWeight(trainSliceIndex, device)

            slice_i_pred = (fusionWeight[0] * alpha0 * slice_i_0 + fusionWeight[1] * alpha1 * slice_i_1) / ( fusionWeight[0] * alpha0 + fusionWeight[1] * alpha1)

            # 2. Loss
            l1Loss = L1_lossFn(slice_i_pred, slice_i)

            pred_gray = dice_loss_tool.rgb_to_grayscale_mapping(slice_i_pred, colors)
            gt_gray = dice_loss_tool.rgb_to_grayscale_mapping(slice_i, colors)

            dice_per_img = dice_loss_tool.dice_coefficient(pred_gray.float(), gt_gray.float(), num_classes)
            trainDice = dice_per_img.mean()
            trainDiceList.append(trainDice.item())
            trainDiceLoss = 1 - trainDice

            l1Loss = l1LossWeight * l1Loss
            trainDiceLoss = diceLossWeight * trainDiceLoss
            loss = l1Loss + trainDiceLoss

            # 3. backward
            loss.backward()
            optimizer.step()

            lossList.append(loss.item())
            l1LossList.append(l1Loss.item())
            trainDiceLossList.append(trainDiceLoss.item())

            # 4. val
            if indexInEpoch == trainloader_length - 1:

                valLoss, valDice, valDicePerClass = validate(curEpoch)

                trainLossAvg = np.mean(lossList)
                trainDiceAvg = np.mean(trainDiceList)
                dataWriter.add_scalar('train/Loss', trainLossAvg, epoch)
                dataWriter.add_scalar('validation/Loss', valLoss, epoch)
                dataWriter.add_scalar('train/dice', trainDiceAvg, epoch)
                dataWriter.add_scalar('validation/dice', valDice, epoch)

                # record best dice
                if valDice > bestDice:
                    bestDice = valDice
                    bestDiceEpoch = epoch

                log.info(
                    "[%s]epoch | Loss: %0.6f(%0.6f+%0.6f) ValLoss:%0.6f | trainDice=%0.6f, valDice=%0.6f, valDiceClass=[%s]" % (
                        epoch, trainLossAvg, np.mean(l1LossList), np.mean(trainDiceLossList), valLoss,
                        trainDiceAvg, valDice.item(),
                        ",".join(["{:.6f}".format(item) for item in valDicePerClass.tolist()]),
                    ))


        scheduler.step()


        if ((epoch % args.save_weight_epoch_interval) == args.save_weight_epoch_interval - 1 or bestDice == valDice):
            modelWeights = {
                'unet1': unet1.state_dict(),
                'unet2': unet2.state_dict(),
            }

            create_dir(args.weight_save_dir_path)
            torch.save(modelWeights, args.weight_save_dir_path + "/CSSR_epoch" + str(epoch) + "_dice" + str(valDice.item()) + ".pt")

def validate(curEpoch):
    totalLoss = 0
    totalDice = 0

    totalDiceNp = torch.zeros(4)
    with torch.no_grad():
        for valIndex, (valSliceList, validationSliceIndex) in enumerate(validationloader, 0):
            slice0, slice_i, slice1 = valSliceList

            slice0 = slice0.to(device)
            slice1 = slice1.to(device)
            # GT
            slice_i = slice_i.to(device)

            unet1Out = unet1(torch.cat((slice0, slice1), dim=1))
            DF_10 = unet1Out[:, :2, :, :]
            DF_01 = unet1Out[:, 2:, :, :]

            dfWeight = model.calcDfWeight(validationSliceIndex, device)

            DF_0i_hat = dfWeight[0] * DF_10 + dfWeight[1] * DF_01
            DF_1i_hat = dfWeight[2] * DF_10 + dfWeight[3] * DF_01

            slice_i_0_hat = valDfModule(slice0, DF_0i_hat)
            slice_i_1_hat = valDfModule(slice1, DF_1i_hat)

            unet2out = unet2(torch.cat((slice0, slice1, DF_10, DF_01, DF_1i_hat, DF_0i_hat, slice_i_1_hat, slice_i_0_hat), dim=1))

            DF_0i = unet2out[:, :2, :, :] + DF_0i_hat
            DF_1i = unet2out[:, 2:4, :, :] + DF_1i_hat
            alpha0,alpha1=getAlphaWeight(unet2out)

            slice_i_0 = valDfModule(slice0, DF_0i)
            slice_i_1 = valDfModule(slice1, DF_1i)

            fusionWeight = model.calcFusionWeight(validationSliceIndex, device)

            slice_i_pred = (fusionWeight[0] * alpha0 * slice_i_0 + fusionWeight[1] * alpha1 * slice_i_1) / ( fusionWeight[0] * alpha0 + fusionWeight[1] * alpha1)


            pred_gray = dice_loss_tool.rgb_to_grayscale_mapping(slice_i_pred, colors)
            gt_gray = dice_loss_tool.rgb_to_grayscale_mapping(slice_i, colors)
            s0_gray = dice_loss_tool.rgb_to_grayscale_mapping(slice0, colors)
            s1_gray = dice_loss_tool.rgb_to_grayscale_mapping(slice1, colors)
            #  save some image
            if valIndex < 10 and curEpoch % 10 == 0:
                imgShowGray(gt_gray[0], "{}_{}_1gt_{}.png".format(curEpoch, valIndex, validationSliceIndex[0]))
                imgShowGray(pred_gray[0], "{}_{}_3pred_{}.png".format(curEpoch, valIndex, validationSliceIndex[0]))
                imgShowGray(s0_gray[0], "{}_{}_2s0.png".format(curEpoch, valIndex))
                imgShowGray(s1_gray[0], "{}_{}_4s1.png".format(curEpoch, valIndex))

            dice, diceList = dice_loss_tool.dice_coefficient2(pred_gray.float(), gt_gray.float(), num_classes)
            totalDiceNp += diceList.cpu()
            totalDice += dice

            dice_loss = 1 - dice

            # val loss
            l1Loss = L1_lossFn(slice_i_pred, slice_i)

            l1Loss = l1LossWeight * l1Loss
            dice_loss = diceLossWeight * dice_loss

            loss = l1Loss + dice_loss
            totalLoss += loss.item()


    val_len = len(validationloader)
    return (totalLoss / val_len), (totalDice / val_len), ( totalDiceNp / val_len)


if __name__ == '__main__':
    main()