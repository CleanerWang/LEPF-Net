import sys
import torch
import argparse
import dataloader
import torch.optim
import torch.nn as nn
from Networks import LEPFNet as net

class L_TV(nn.Module):

    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train(config):

    # 初始化DehazeNet网络
    dehaze_net = net.dehaze_net()

    # 初始化DehazeNet权重
    dehaze_net.apply(weights_init)

    # 加载训练集
    train_dataset = dataloader.loader(config.orig_images_path, config.hazy_images_path,0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size,
                                               shuffle=False, num_workers=config.num_workers, pin_memory=True)

    # 损失函数
    L_mse = nn.MSELoss()
    L1_G = torch.nn.L1Loss()
    L_tv = L_TV()

    # 判断cuda是否能用
    if torch.cuda.is_available():
        L_mse.cuda()
        L1_G = L1_G.cuda()
        L_tv = L_tv.cuda()
        dehaze_net = dehaze_net.cuda()

    # Adam优化器
    optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=config.lr, betas=(config.lr_b1, config.lr_b2))

    # 训练初始模型
    dehaze_net.train()

    for epoch in range(config.num_epochs):
        for iteration, (img_orig, img_haze) in enumerate(train_loader):

            # 图片放入CUDA
            img_orig = img_orig.cuda()
            img_haze = img_haze.cuda()

            optimizer.zero_grad()

            # 获得清晰图片
            clean_image, cpm = dehaze_net(img_haze)

            # 获取损失
            loss_one = 10*L1_G(clean_image, img_orig)
            Loss_TV = 10*L_tv(cpm)
            loss_G = Loss_TV + loss_one

            loss_G.sum().backward()
            torch.nn.utils.clip_grad_norm_(dehaze_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            # 打印Loss
            sys.stdout.write("\r[epoch:%2d/%d][%d/%d][Loss : %.4f]" %
                             (1 + epoch, config.num_epochs, iteration + 1, len(train_loader), loss_G.item())
                             )

        # 保存当前模型
        modle_path = config.snapshots_folder % (epoch + 1)
        torch.save(dehaze_net.state_dict(), modle_path)

def doConfig():
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_b1', type=float, default=0.5)
    parser.add_argument('--lr_b2', type=float, default=0.99)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    # UIEB结果文件夹
    parser.add_argument('--gen_UIEB_dir', type=str, default="/home/jqyan/YiJianWang/LEPF-Net/results/UIEB/")
    # NUYv2-RBG结果文件夹
    parser.add_argument('--gen_UWCNN_dir', type=str, default="/home/jqyan/YiJianWang/LEPF-Net/results/UWCNN/")
    # 模糊图像
    parser.add_argument('--hazy_images_path', type=str, default='/home/jqyan/YiJianWang/LEPF-Net/dataset/haze/')
    # 清晰图像
    parser.add_argument('--orig_images_path', type=str, default='/home/jqyan/YiJianWang/LEPF-Net/dataset/clear/')
    # UIEB测试集
    parser.add_argument('--test_UIEB_dir', type=str, default="/home/jqyan/YiJianWang/LEPF-Net/dataset/test/UIEB/*")
    # NUYv2-RGB测试集
    parser.add_argument('--test_UWCNN_dir', type=str, default="/home/jqyan/YiJianWang/LEPF-Net/dataset/test/UWCNN/*")
    # 模型存放文件夹
    parser.add_argument('--snapshots_folder', type=str, default="/home/jqyan/YiJianWang/LEPF-Net/snapshots/epoch_%s.pth")

    return parser.parse_args()

if __name__ == "__main__":

    config = doConfig()
    train(config)











