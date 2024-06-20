# import torch
# import torch.nn as nn

# from nets.vgg import VGG16


# def get_img_output_length(width, height):
#     def get_output_length(input_length):
#         # input_length += 6
#         filter_sizes = [2, 2, 2, 2, 2]
#         padding = [0, 0, 0, 0, 0]
#         stride = 2
#         for i in range(5):
#             input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
#         return input_length
#     return get_output_length(width) * get_output_length(height) 
    
# class Siamese(nn.Module):
#     def __init__(self, input_shape, pretrained=False):
#         super(Siamese, self).__init__()
#         self.vgg = VGG16(pretrained, 3)
#         del self.vgg.avgpool
#         del self.vgg.classifier
        
#         flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])
#         self.fully_connect1 = torch.nn.Linear(flat_shape, 512)
#         self.fully_connect2 = torch.nn.Linear(512, 1)

#     def forward(self, x):
#         x1, x2 = x
#         #------------------------------------------#
#         #   我们将两个输入传入到主干特征提取网络
#         #------------------------------------------#
#         x1 = self.vgg.features(x1)
#         x2 = self.vgg.features(x2)   
#         #-------------------------#
#         #   相减取绝对值，取l1距离
#         #-------------------------#     
#         x1 = torch.flatten(x1, 1)
#         x2 = torch.flatten(x2, 1)
#         x = torch.abs(x1 - x2)
#         #-------------------------#
#         #   进行两次全连接
#         #-------------------------#
#         x = self.fully_connect1(x)
#         x = self.fully_connect2(x)
#         return x
import torch
import torch.nn.functional as F
import torch.nn as nn
# Import the ResNext model from torchvision or your custom implementation
from torchvision.models import resnext50_32x4d
import torchvision.models as models
from holocron.models.classification import repvgg_a0 as repvgg
import sys
# sys.path.insert(0,'/home/hozon/bug/Siamese-pytorch/nets')
from nets.ridnet import Block, BasicBlock
from nets.convnext import ConvNeXt
def get_model(modelname, input_shape):
    model = None
    if modelname == "convnext":
        model = models.convnext_tiny(pretrained=True)
        # print(model)
        # del model.avgpool
        # del model.classifier
        model.avgpool = Identity()
        model.classifier = Identity()
        # model = ConvNeXt(3, 1)
        # del model.head  # Typically, you remove the last fully connected layer
        # del model.avgpool
    elif modelname == "repvgg":
        model = repvgg(pretrained=True)
        del model.head  # Typically, you remove the last fully connected layer
        del model.pool
    elif modelname == "resnext":
        model = resnext50_32x4d(pretrained=True)
        model.avgpool = Identity()
        model.fc = Identity()
    flat = get_flat_shape(model, input_shape)
    return model, flat
def get_flat_shape(model, input_shape):
    dummy = torch.ones(1, 3, *input_shape)
    with torch.no_grad():
        dummy = model(dummy)
    return dummy.numel()
# 定义一个 Identity 类，它仅仅返回输入数据
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # Calculate the euclidean distance and calculate the contrastive loss
      euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

      loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


      return loss_contrastive
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
class Siamese(nn.Module):
    def __init__(self, input_shape, pretrained=False):
        super(Siamese, self).__init__()
        self.model, flat_shape = get_model("resnext", input_shape)
        # self.head = BasicBlock(3, 64, 1, 1, 0)
        # self.head2 = BasicBlock(64, 3, 1, 1, 0)
        # print(self.model)
        # print("flat_shape",flat_shape)
        # self.ridBlock = Block(64, 64)
        self.fully_connect1 = torch.nn.Linear(flat_shape, 512)
        self.fully_connect2 = torch.nn.Linear(512, 1)

    def forward(self, x):
        x1, x2 = x
        # x1 = self.head(x1)
        # x2 = self.head(x2)
        # x1 = self.ridBlock(x1)
        # x2 = self.ridBlock(x2)
        # x1 = self.head2(x1)
        # x2 = self.head2(x2)
        x1 = self.model(x1)
        x2 = self.model(x2)
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        # losses = self.loss(x1, x2)
        x = torch.abs(x1 - x2)
        x = self.fully_connect1(x)
        x = self.fully_connect2(x)
        return x