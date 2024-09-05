import os

from dataset import Fusion_dataset
from mTrain import RGB2YCrCb, YCrCb2RGB

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import argparse
from LightWeightNet.LightWeightNet import net

from torch.utils.data import DataLoader
from utils import *


def eval():
    torch.set_grad_enabled(False)
    model.eval()


    count_paraa_in_MB()
    totalTime=0
    for batch in testing_data_loader:
        with torch.no_grad():
            vis, ir, name = batch[0], batch[1], batch[2]

        image_vis_ycrcb = RGB2YCrCb(vis)
        visY = image_vis_ycrcb[:, :1]

        visY = visY.to('cuda:0')
        ir = ir.to('cuda:0')

        with torch.no_grad():
            start_time = time.time()
            VisL, VisR, VisInputY = model(visY)

            IrL, IrR, IrInput = model(ir)
            Lmax = torch.max(VisL, IrL)
            Rmax = torch.max(VisR, IrR)

            FuseY = Lmax * Rmax
            end_time = time.time()
            totalTime=totalTime+(end_time-start_time)
            saveFusionYWithCrCb(FuseY, image_vis_ycrcb, name)


        if not os.path.exists(opt.output_folder):
            os.mkdir(opt.output_folder)


    average_time = totalTime / len(testing_data_loader.dataset)
    average_time_formatted = "{:.3f}".format(average_time)
    print(f"平均运行时间：{average_time_formatted} 秒")
    torch.set_grad_enabled(True)


def count_paraa_in_MB():
    params_size_MB = count_parameters_in_MB(model)
    print(f"总训练参数大小：{params_size_MB} MB")


def count_parameters_in_MB(model):
    total_params = np.sum(
        np.fromiter((np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name), dtype=int))
    total_params_in_MB = total_params / 1e6
    return total_params_in_MB


def saveFusionYWithCrCb(I, image_vis_ycrcb, name):
    fusionImage_ycrcb = torch.cat(
        (I, image_vis_ycrcb[:, 1:2, :,
            :], image_vis_ycrcb[:, 2:, :, :]),
        dim=1,
    )
    fusionResult_RGB = YCrCb2RGB(fusionImage_ycrcb)
    ones = torch.ones_like(fusionResult_RGB)
    zeros = torch.zeros_like(fusionResult_RGB)
    fusionResult_RGB = torch.where(fusionResult_RGB > ones, ones, fusionResult_RGB)
    fusionResult_RGB = torch.where(
        fusionResult_RGB < zeros, zeros, fusionResult_RGB)
    fused_image = fusionResult_RGB.cpu().numpy()
    fused_image = fused_image.transpose((0, 2, 3, 1))
    fused_image = (fused_image - np.min(fused_image)) / (
            np.max(fused_image) - np.min(fused_image)
    )
    fused_image = np.uint8(255.0 * fused_image)
    for k in range(len(name)):
        image = fused_image[k, :, :, :]
        image = image.squeeze()
        image = Image.fromarray(image)
        save_path = os.path.join(opt.output_folder + '/Fuse/', name[k])
        image.save(save_path)
        print('Fusion {0} Sucessfully!'.format(save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECFusion')
    parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
    parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
    parser.add_argument('--model', default='weights/epoch_400.pth', help='Pretrained base model')
    parser.add_argument('--output_folder', type=str, default='./results/DEC/')
    opt = parser.parse_args()
    test_set = Fusion_dataset('val')
    testing_data_loader = DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net().to(device)

    model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
    eval()
