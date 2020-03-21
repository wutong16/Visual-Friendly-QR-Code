# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

def Transfer(image_file ='photo1.jpg', qart_file ='./work_dir/img5/qart.jpg', output_path='./work_dir/StyleTransfer',
             run_time=60):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    version=6
    qr_size=version*4+17
    module_size=9
    render_size=qr_size*module_size
    imsize = render_size

    loader = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor

    def image_loader(image_name):
        image = Image.open(image_name)
        # image = image.convert('L')
        # fake batch dimension required to fit network's input dimensions
        image = loader(image).unsqueeze(0)
        return image.to(device, torch.float)

    def prepare_accuracy_target(image_file, qart_file, target_file):
        Qart = Image.open(qart_file).resize([render_size, render_size])
        style_img = Image.open(image_file).resize([render_size, render_size])

        #pattern = get_mask(version, option='gaussian',gaussian_base=6)
        pattern = get_mask(version, option='circle',center_reduce=2/3)
        pattern = np.dstack([pattern, pattern, pattern])

        target_img = pattern * Qart + (1 - pattern) * style_img
        target_img = Image.fromarray(target_img.astype('uint8')).convert('RGB')
        target_img.save(target_file)

    target_file=qart_file[:-4]+'.target.jpg'
    prepare_accuracy_target(image_file, qart_file, target_file)
    # exit()
    target_img = image_loader(target_file)
    style_img = image_loader(image_file)
    content_img = image_loader(qart_file)

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"


    unloader = transforms.ToPILImage()  # reconvert into PIL image

    plt.ion()

    def imshow(tensor, title=None):
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)      # remove the fake batch dimension
        image = unloader(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(0.001) # pause a bit so that plots are updated

    def imsave(tensor, title=None):
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)      # remove the fake batch dimension
        image = unloader(image)
        if os.path.exists(title):
            os.remove(title)
        image.save(title)


    class AccuracyLoss(nn.Module):
        def __init__(self, qart_target):
            super(AccuracyLoss, self).__init__()
            self.target = qart_target

        def forward(self, input):
            self.loss = F.mse_loss(input,self.target)
            return input

    class ContentLoss(nn.Module):
        def __init__(self, target,):
            super(ContentLoss, self).__init__()
            self.target = target.detach()

        def forward(self, input):
            self.loss = F.mse_loss(input, self.target)
            return input

    def gram_matrix(input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

    class StyleLoss(nn.Module):
        def __init__(self, target_feature):
            super(StyleLoss, self).__init__()
            self.target = gram_matrix(target_feature).detach()

        def forward(self, input):
            G = gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)
            return input

    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            self.mean = mean.clone().detach().view(-1, 1, 1)
            self.std = std.clone().detach().view(-1, 1, 1)

        def forward(self, img):
            return (img - self.mean) / self.std

    # desired depth layers to compute style/content losses :
    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                   style_img, content_img,
                                   content_layers=content_layers_default,
                                   style_layers=style_layers_default):
        cnn = copy.deepcopy(cnn)

        normalization = Normalization(normalization_mean, normalization_std).to(device)

        content_losses = []
        style_losses = []
        accuracy_losses = []

        model = nn.Sequential(normalization)

        accuracy_loss = AccuracyLoss(target_img)
        model.add_module("accuracy_loss_{}".format(0),accuracy_loss)
        accuracy_losses.append(accuracy_loss)

        i = 0
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
        model = model[:(i + 1)]

        return model, style_losses, content_losses, accuracy_losses


    input_img = content_img.clone()

    def get_input_optimizer(input_img):
        # this line to show that input is a parameter that requires a gradient
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer

    def run_style_transfer(cnn, normalization_mean, normalization_std,
                           content_img, style_img, input_img, num_steps=30,
                           style_weight=1000000, content_weight=1, accuracy_weight=200):
        """Run the style transfer."""
        print('# Running the style transfer model ..')
        model, style_losses, content_losses, accuracy_losses = get_style_model_and_losses(cnn,
            normalization_mean, normalization_std, style_img, content_img)
        optimizer = get_input_optimizer(input_img)

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        run = [0]
        print('# Results will be saved at {}'.format(output_path + '/run_time<n>.jpg ..'))
        print("# runtime".format(run), end=' ')
        while run[0] <= num_steps:

            def closure():
                # correct the values of updated input image
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0
                accuracy_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss
                for al in accuracy_losses:
                    accuracy_score += al.loss

                style_score *= style_weight
                content_score *= content_weight
                accuracy_score *= accuracy_weight

                loss = style_score + content_score + accuracy_score
                loss.backward()

                run[0] += 1
                if run[0] % 10 == 0:
                    print("{}".format(run), end=' ')
                    # print('Style Loss : {:4f} Content Loss: {:4f} Accuracy Loss: {:4f}'.format(
                    #     style_score.item(), content_score.item(), accuracy_score.item()))
                    # imsave(input_img, title=output_path + '/run_time' + str(run[0]) + '.jpg')
                if run[0] ==  num_steps:
                    input_img.data.clamp_(0, 1)
                    imsave(input_img, title = output_path + '/run_time'+str(run[0])+'.jpg')
                return style_score + content_score + accuracy_score

            optimizer.step(closure)
        print()

        input_img.data.clamp_(0, 1)

        return input_img


    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img,run_time)
    # plt.ioff()
    # plt.show()

def get_mask(version=6, module_size=9, option='square', gaussian_base=5, center_base=0.8, center_reduce=2 / 3):
    center_start = int((1 - center_reduce) * module_size // 2)
    center_end = int((1 + center_reduce) * module_size // 2)
    qr_size = version * 4 + 17
    render_size = qr_size * module_size
    if option == 'square':
        pattern = np.zeros([module_size, module_size])
        for i in range(center_start, center_end):
            for j in range(center_start, center_end):
                pattern[i][j] = 1  # devided by centual square and egde part
        p = pattern
        for _ in range(qr_size - 1):
            p = np.concatenate((p, pattern), axis=0)
        P_square = p
        for _ in range(qr_size - 1):
            P_square = np.concatenate((P_square, p), axis=1)
        mask = P_square
    elif option == 'circle':
        pattern = np.zeros([module_size, module_size])
        for i in range(center_start, center_end):
            for j in range(center_start, center_end):
                if np.sqrt((i-module_size/2)**2 + (j-module_size/2)**2)<=(1-center_reduce)*module_size:
                    pattern[i][j] = 1  # devided by centual square and egde part
        p = pattern
        for _ in range(qr_size - 1):
            p = np.concatenate((p, pattern), axis=0)
        P_square = p
        for _ in range(qr_size - 1):
            P_square = np.concatenate((P_square, p), axis=1)
        mask = P_square
    elif option == 'gaussian':
        center = module_size // 2 + 1
        e = np.zeros([module_size, module_size])
        for i in range(module_size):
            for j in range(module_size):
                e[i][j] = (i + 1 - center) ** 2 + (j + 1 - center) ** 2  # devided by gaussian weight
        b = 10 * np.exp(-e)  # 控制点的大小
        gaussian = np.exp(-e / gaussian_base)
        p = gaussian
        for _ in range(qr_size - 1):
            p = np.concatenate((p, gaussian), axis=0)
        P_gaussian = p
        for _ in range(qr_size - 1):
            P_gaussian = np.concatenate((P_gaussian, p), axis=1)
        mask = P_gaussian
    else:
        raise NameError('Unexpected option for mask!')
    return mask

if __name__ == '__main__':
    begin = 1
    end = 100
    path = './input/good/'
    for i in range(begin, end + 1):
        image = 'img' + str(i) + '.jpg'
        if not os.path.exists(path + image):
            continue
        qart_path = './qart/qart_'+str(i)+'.jpg'
        Transfer(image_file=image, work_dir=path, n=i, qart_file=qart_path)