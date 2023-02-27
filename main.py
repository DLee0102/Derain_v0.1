import Dehazed.defog_v2 as defog
import Derain.Derain_platform.PreNet_rtest as derain
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image

input_path = '/Users/dingli/Desktop/python-pro/defog/Derain_v0.1/testdata'
model_path = '/Users/dingli/Desktop/python-pro/defog/Derain_v0.1/Derain/Derain_platform/result_Version6/model_best.ckpt'
output_path = '/Users/dingli/Desktop/python-pro/defog/Derain_v0.1/results'
temp_path = '/Users/dingli/Desktop/python-pro/defog/Derain_v0.1/temp/temp_img.jpg'

def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img

def transform_invert(img_, transform_train):
    """
    将data 进行反transfrom操作
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])
 
    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    img_ = np.array(img_) * 255
 
    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]) )
 
    return img_


if __name__ == '__main__':
    dataloader, net = derain.prepareModel(input_path, model_path)
    test_tfm = transforms.Compose([
        # transforms.CenterCrop([128, 128]),    # 这行没有必要，用原始图片进行测试即可
        transforms.ToTensor(),
    ])
    # 用于打印日志
    cnt = 0
    total = 0

    # 获取测试用例总数
    for input, label in dataloader:
        total += 1

    for input, label in dataloader:
        cnt += 1
        # input = input.to('cuda')        # 用cuda加速测试，也可以不用，不用cuda加速测试速度会很慢

        print('finished:{:.2f}%'.format(cnt*100/total))

        with torch.no_grad():
            output_image, _ = net(input) # 输出的是张量
            # output_image = tensor_to_np(output_image)
            # output_image = transform_invert(output_image, test_tfm)
            save_image(output_image, temp_path)
            # print(output_image)

        output_image = cv2.imread(temp_path)
        # print(output_image)
        defog.deFogging(output_image, output_path, cnt)

