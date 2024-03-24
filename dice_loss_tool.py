# compute dice,and so on
import torch



def rgb_to_grayscale_mapping(batch_rgb_images, colors):
    """ 3 channels image to mask(1 channel) image, by vector L1 distance"""
    B, C, H, W = batch_rgb_images.shape
    N = colors.shape[0]
    colors = colors.view(N, C, 1, 1)

    expanded_rgb_images = batch_rgb_images.unsqueeze(1)
    # L1 distance
    distances = torch.norm(expanded_rgb_images - colors, dim=2)

    # for background class, enlargement distance
    # distances[:, 0, :, :] *=2
    # distances[:, 0, :, :] *=10

    #  (B, H, W)
    grayscale_indices = torch.argmin(distances, dim=1)
    return grayscale_indices


def dice_coefficient(pred, target, num_classes, smooth=1e-5):
    dice_coeffs = torch.zeros((pred.shape[0], num_classes-1), device=pred.device)
    for class_index in range(1, num_classes):
        pred_class = (pred == class_index).float()
        target_class = (target == class_index).float()
        intersection = (pred_class * target_class).sum(dim=(1, 2))
        union = pred_class.sum(dim=(1, 2)) + target_class.sum(dim=(1, 2))
        dice_coeffs[:, class_index-1] = (2 * intersection + smooth) / (union + smooth)  # smooth, avoid dividing by 0
    result_dice = dice_coeffs.mean(dim=1)
    return result_dice


def dice_coefficient2(pred, target, num_classes, smooth=1e-5):
    dice_coeffs = torch.zeros((pred.shape[0], num_classes-1), device=pred.device)
    for class_index in range(1, num_classes):
        pred_class = (pred == class_index).float()
        target_class = (target == class_index).float()
        intersection = (pred_class * target_class).sum(dim=(1, 2))
        union = pred_class.sum(dim=(1, 2)) + target_class.sum(dim=(1, 2))
        dice_coeffs[:, class_index-1] = (2 * intersection + smooth) / (union + smooth)
    return dice_coeffs.mean(),dice_coeffs.mean(dim=0)
