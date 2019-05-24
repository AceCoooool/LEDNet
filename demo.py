import os
import sys
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
from torchvision import transforms

cur_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_path, '..'))
from model.lednet import LEDNet
import utils as ptutil


def parse_args():
    parser = argparse.ArgumentParser(description='Demo for LEDNet from a given image')

    parser.add_argument('--input-pic', type=str, default=os.path.join(cur_path, 'png/demo.png'),
                        help='path to the input picture')
    parser.add_argument('--pretrained', type=str,
                        default=os.path.expanduser('~/cbb/own/pretrained/seg/lednet/LEDNet_final.pth'),
                        help='Default Pre-trained model root.')
    parser.add_argument('--cuda', type=ptutil.str2bool, default='true',
                        help='demo with GPU')

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cpu')
    if args.cuda:
        device = torch.device('cuda')
    # Load Model
    model = LEDNet(19).to(device)
    model.load_state_dict(torch.load(args.pretrained))
    model.eval()

    # Load Images
    img = Image.open(args.input_pic)

    # Transform
    transform_fn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = transform_fn(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)

    predict = torch.argmax(output, 1).squeeze(0).cpu().numpy()
    mask = ptutil.get_color_pallete(predict, 'citys')
    mask.save(os.path.join(cur_path, 'png/output.png'))
    mmask = mpimg.imread(os.path.join(cur_path, 'png/output.png'))
    plt.imshow(mmask)
    plt.show()
