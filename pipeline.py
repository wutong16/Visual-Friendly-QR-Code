from __future__ import absolute_import

from pyqart.art.qart import QArtist
from pyqart.qr.printer import QrImagePrinter
from render.image_render import Render
from render.neural_style import *
from render.utils import *
import os
import argparse
import sys
import time
import cv2

'''
https://github.com/7sDream/pyqart
'''
class VisualFriendlyQR(object):
    def __init__(self, image_file='input/example1.jpg', level=0, url='https://cn.bing.com/',
                 mask_file=None, output_dir=None, render_only=True, image_enhance=False,
                 rotation = 0, module_size=9, center_shape='square', transfer_runtime=60):
        self.url = url
        self.work_dir = os.path.join('./work_dirs', image_file.split('/')[-1].split('.')[0])
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)
        self.output_dir = './output' if output_dir is None else output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.image_file = image_file
        self.mask_file = mask_file
        self.prefer_mode = True if mask_file is not None else False
        self.transfer_runtime = transfer_runtime
        self.transfer_path = os.path.join(self.work_dir, 'StyleTransfer')
        self.transfer_file = self.transfer_path + '/run_time{}.jpg'.format(transfer_runtime)
        self.qart_file = os.path.join(self.work_dir, 'qart_mask.jpg') if self.prefer_mode else os.path.join(self.work_dir, 'qart.jpg')
        self.save_file = os.path.join(self.output_dir, image_file.split('/')[-1].split('.')[0] + '.QVF.jpg')

        self.render_only = render_only
        self.rotation = rotation
        self.image_enhance = image_enhance

        # we only support version=6 at present, to be update soon
        self.version = 6
        self.qr_size =self.version * 4 + 17
        self.rotation = 0
        self.level = level
        self.module_size = module_size
        self.render_size = self.qr_size * module_size
        self.center_shape = center_shape

        if self.render_only:
            assert os.path.exists(self.qart_file), '{} not exist!'.format(self.qart_file)
            assert os.path.exists(self.transfer_file), '{} not exist!'.format(self.transfer_file)

        print(' *** Visually Friendly QR Model Built! *** ')

    def run(self):
        self.prepare_images()
        if not self.render_only:
            self.prepare_qart()
            self.prepare_NST()
        self.render_QVF()

    def prepare_images(self):
        try:
            cv2.imread(self.image_file)
        except:
            raise FileNotFoundError
        # rotation helps leave data area with less function codes
        # if self.mask_file is not None:
        #     try:
        #         mask = cv2.imread(self.mask_file)
        #     except:
        #         raise FileNotFoundError
        #     prefer_mask = cv2.resize(mask, (self.render_size, self.render_size))
        #     prefer_mask_gray = cv2.cvtColor(prefer_mask, cv2.COLOR_BGR2GRAY)
        #     center_y, center_x = find_mass_center(prefer_mask_gray, norm=True)
        #     if center_x > 0.5 and center_y > 0.5:
        #         self.rotation = 2
        #     elif center_x > 0.5 and center_y < 0.5:
        #         self.rotation = 3
        #     elif center_x < 0.5 and center_y > 0.5:
        #         self.rotation = 1
        #     else:
        #         self.rotation = 0
        # print('[Rotation]: ', self.rotation)

    def prepare_qart(self):
        print('[Start Qart Generation]')
        start_time = time.time()
        if os.path.exists(self.qart_file):
            print('# qart {} for it already exists, now overwriting it.'.format(self.qart_file))
        tmp_failed = True
        f_count = 0
        while(tmp_failed):
            artist = QArtist(url=self.url, level=self.level, rotation=self.rotation,
                             img=self.image_file, prefer_mask=self.mask_file, prefer_option=self.prefer_mode)
            QrImagePrinter.print(artist, self.qart_file, point_width=1)
            tmp_failed = artist.failure
            if tmp_failed:
                f_count += 1
                print('# Failed {} times, retrying...'.format(f_count))
        print('[Qart Generation Done] time used: {:.3} s, saved at {}'.format(time.time() - start_time, self.qart_file))

    def prepare_NST(self):
        print('[Start Neural Style Transfer]')
        start_time = time.time()
        Transfer(image_file=self.image_file, qart_file=self.qart_file, output_path=self.transfer_path,run_time=self.transfer_runtime)
        print('[Neural Style Transfer Done] time used: {:.3} s, saved at {}'.format(time.time() - start_time, self.transfer_path))

    def render_QVF(self):
        print('[Start QVF Rendering]')
        start_time = time.time()
        render = Render(version=self.version, module_size=self.module_size,
                        qart_file=self.qart_file, mask_file=self.mask_file, image_file=self.image_file,
                        transfer_file=self.transfer_file, save_file=self.save_file, center_shape=self.center_shape,
                        rotation=self.rotation, image_enhance=self.image_enhance)
        render.run_rendering()
        print('[QVF Rendering Done] time used: {:.3} s, saved at {}'.format(time.time() - start_time, self.save_file))

def main():
    parser = argparse.ArgumentParser(
        prog="Visual-Friendly QR Code",
        description="A program of generate Visual-friendly QR Code Codes.",
    )
    parser.add_argument(
        '-u', '--url', type=str,
        default='https://cn.bing.com/',
        help="url to be encoded",
    )
    parser.add_argument(
        '-i', '--image_file', type=str,
        default="./input/img74.jpg",
        help="path to background image",
    )
    parser.add_argument(
        '-m', '--mask_file', type=str, default=None,
        help="path to user-defined mask",
    )
    parser.add_argument(
        '-l', '--level', type=int, default=0,
        help="QrCode error correction level, 0 to 3, default is 0",
    )
    parser.add_argument(
        '-o', '--output_dir', type=str, default=None,
        help="results saved in the path",
    )
    parser.add_argument(
        '-r', '--render_only', action='store_true',
        help="when qart and transfer results are already generated"
    )
    parser.add_argument(
        '-e', '--image_enhance', action='store_true',
        help="image enhancement for weak examples, e.g. low contrast"
    )

    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    VFQR_model = VisualFriendlyQR(image_file=args.image_file, level=args.level, url=args.url,
                                  mask_file=args.mask_file, output_dir=args.output_dir, render_only=args.render_only,
                                  image_enhance=args.image_enhance)
    VFQR_model.run()


if __name__ == '__main__':
    main()