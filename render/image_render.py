import cv2
import numpy as np
from render.utils import *
import os.path
from PIL import Image
from PIL import ImageEnhance

class Render(object):
    def __init__(self, version=6, module_size=9,
                 qart_file='./work_dirs/img5/qart.jpg', image_file='./input/img5.jpg', mask_file=None,
                 transfer_file='./work_dirs/img5/StyleTransfer/run_time60.jpg', save_file='./output/img5.QVF.jpg',
                 center_shape='square', finetune=True, rotation=0, image_enhance=False,
                 thrd_low=100, thrd_high=170):
        self.version = version
        self.qr_size = version * 4 + 17
        self.module_size = module_size
        self.render_size = self.qr_size * module_size
        self.thrd_low = thrd_low
        self.thrd_high = thrd_high
        self.rotation = rotation

        self.center_shape = center_shape
        self.finetune = finetune
        self.save_file = save_file

        self.init_weight_maps(mask_file)
        self.init_source_images(image_file, qart_file, transfer_file, enhance=image_enhance)


    def init_source_images(self, image_file, qart_file, transfer_file, enhance=False):
        _, self.qart_img = cv2.threshold(cv2.cvtColor(cv2.imread(qart_file), cv2.COLOR_BGR2GRAY), 127, 255,
                                         cv2.THRESH_BINARY)

        self.img_input = cv2.resize(cv2.imread(image_file), (self.render_size, self.render_size))
        if enhance:
            self.img_input = image_enhance(self.img_input, mode='equalization')
        img_LAB = cv2.cvtColor(self.img_input, cv2.COLOR_BGR2LAB)
        self.L_i_init = img_LAB[:, :, 0]
        self.A_i_init = img_LAB[:, :, 1]
        self.B_i_init = img_LAB[:, :, 2]
        self.L_i = self.L_i_init.copy()

        self.bina_threshold = self._fake_threshold(self.L_i_init)
        _, self.L_p_init = cv2.threshold(self.L_i, self.bina_threshold, 255, cv2.THRESH_BINARY)
        self.L_p = self.L_p_init.copy()

        self.img_transfer = cv2.resize(cv2.imread(transfer_file), (self.render_size, self.render_size))
        _, self.L_t_init = cv2.threshold(cv2.cvtColor(self.img_transfer, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
        self.L_t = self.L_t_init.copy()

        self.L_b = full_resize(self.qart_img, self.version, self.module_size)

        self.L_r = np.zeros([self.render_size, self.render_size])
        pass

    def init_weight_maps(self, mask_file=None):
        if mask_file is not None:
            self.prefer_mode = True
            self.prefer_mask = cv2.cvtColor(cv2.resize(cv2.imread(mask_file), (self.render_size, self.render_size)),
                                            cv2.COLOR_BGR2GRAY)
            self.weight_r = self._get_mask(option='prefer', center_base=0.8, center_reduce=1 / 3)
        else:
            self.prefer_mode = False
            self.weight_r = self._get_mask(option='center', center_base=0.8, center_reduce=1 / 3)

    def run_rendering(self):

        self.process_binary(beta=1.3, gamma=0.7, center_reduce=1 / 2)
        self.process_grayscale()
        if self.finetune:
            self.run_finetune()
        self.output = self.build_output()
        cv2.imwrite(self.save_file, self.output)
        print('# QVF saved at {}'.format(self.save_file))
        return

    def process_grayscale(self):
        one = np.ones([self.render_size,self.render_size])
        # FIXME: whether self.belta or not
        module_centers = self._get_mask(option=self.center_shape)
        bina_thrd = self._fake_threshold(self.L_p)
        _,self.L_p = cv2.threshold(self.L_p.astype(np.uint8), bina_thrd, 255,cv2.THRESH_BINARY)
        L_s_correct = module_centers*self.L_b + (one-module_centers)*self.L_p #the bright bits should be brighter
        self.L_r = _Max(0, _Min(255 * one, (one - self.weight_r) * self.L_i + self.weight_r * L_s_correct))
        return

    def build_output(self):
        L_funct_added = add_function_code(self.qart_img, self.L_r, self.version, self.module_size, self.rotation)
        LAB_rebuild = np.dstack(
            [L_funct_added.astype(np.uint8), self.A_i_init.astype(np.uint8), self.B_i_init.astype(np.uint8)])
        return cv2.cvtColor(LAB_rebuild, cv2.COLOR_LAB2BGR)

    def eval_accuracy(self):
        qr_read = np.zeros([self.qr_size, self.qr_size])
        self.local_thrd = self._local_avg_brightness()

        for i in range(self.qr_size):
            for j in range(self.qr_size):
                sample = self.L_r[
                    i * self.module_size + self.module_size // 3:i * self.module_size + self.module_size * 2 // 3 ,
                    j * self.module_size + self.module_size // 3:j * self.module_size + self.module_size * 2 // 3]
                read = np.sum(np.sum(sample)) / (self.module_size//3)**2  # center
                if (read < self.thrd_low):
                    qr_read[i][j] = 0
                elif (read > self.thrd_high):
                    qr_read[i][j] = 1
                else:
                    qr_read[i][j] = 0.5

        differ_map = abs(qr_read*255 - self.qart_img)
        num_error = np.sum(np.sum(differ_map)) // 255
        accuracy_loss = np.sum(np.sum(differ_map)) / (self.qr_size ** 2)

        return num_error, accuracy_loss , differ_map

    def eval_similarity(self, mask_map):
        likeness_loss = np.sum(np.sum(abs(self.L_i/255 - self.L_r/255)) * mask_map) / self.render_size ** 2
        # Fixme: the scale
        return likeness_loss/1000

    def run_finetune(self, step = 0.03):
        print('# Begin iteratively finetuning ...')
        option = 'prefer' if self.prefer_mode else 'center'
        compare_map = self._get_mask(option=option, center_base = 1, center_reduce =1 / 3)
        compare_map = 1 - compare_map
        i = 0
        while True:
            self.process_grayscale()
            num_error, a_loss, differ_map = self.eval_accuracy()
            l_loss = self.eval_similarity(compare_map)
            if num_error<5:
                break
            differ_map = full_resize(differ_map, self.version, self.module_size)/255  # 1~inaccurate 0~accurate
            differ_map = cv2.blur(differ_map,(10,10))
            self.weight_r += step * compare_map * (0.5 * a_loss * differ_map - 0.01 * l_loss * (1 - differ_map)) #lower alpha, higher a_loss, lower l_loss
            i += 1
        print('# Finished finetuning after {} iterations '.format(i))
        return

    def process_binary(self, beta=1.3, gamma=0.6, center_reduce=1 / 2):
        option = 'prefer' if self.prefer_mode else 'center'
        mask_map = self._get_mask(option=option, center_base=2, center_reduce=center_reduce)

        noise_map = find_noise_map(self.L_p, self.L_b, self.version, self.module_size)
        noise_map = cv2.resize(noise_map,(self.render_size,self.render_size))

        style_weight = beta * (gamma * noise_map + (1 - gamma) * mask_map)
        style_weight = _Min(style_weight, np.ones([self.render_size, self.render_size]))
        L_t = cv2.blur(self.L_t, (5, 5))
        self.L_p = self.L_p * (1 - style_weight) + L_t * style_weight
        return

    def _get_mask(self, option ='gaussian', gaussian_base = 5, center_base = 0.8, center_reduce =1 / 3):
        # assert int((1- center_reduce)*self.module_size) % 2 == 0
        center_start = int((1 - center_reduce) * self.module_size // 2)
        center_end = int((1 + center_reduce) * self.module_size // 2)

        if option == 'square':
            module = np.zeros([self.module_size, self.module_size])
            for i in range(center_start, center_end):
                for j in range(center_start, center_end):
                    module[i][j] = 1  # devided by centual square and egde part
            row_module = module
            for _ in range(self.qr_size - 1):
                row_module = np.concatenate((row_module, module), axis=0)
            P_square = row_module
            for _ in range(self.qr_size - 1):
                P_square = np.concatenate((P_square, row_module), axis=1)
            mask = P_square
        elif option == 'gaussian':
            center = self.module_size // 2 + 1
            module = np.zeros([self.module_size, self.module_size])
            for i in range(self.module_size):
                for j in range(self.module_size):
                    module[i][j] = (abs((i + 1 - center))**2 + abs((j + 1 - center))**2)**1  # devided by gaussian weight
            gaussian = np.exp(-module / gaussian_base)
            row_module = gaussian
            for _ in range(self.qr_size - 1):
                row_module = np.concatenate((row_module, gaussian), axis=0)
            P_gaussian = row_module
            for _ in range(self.qr_size - 1):
                P_gaussian = np.concatenate((P_gaussian, row_module), axis=1)
            mask = P_gaussian
        elif option == 'center':
            P_center = center_base * np.ones((self.render_size, self.render_size))
            for i in range(self.render_size):
                for j in range(self.render_size):
                    d = ((i - self.render_size / 2) ** 2 + (j - self.render_size / 2) ** 2) / self.render_size ** 2
                    P_center[i][j] *= d ** (center_reduce)
            mask = P_center
        elif option == 'prefer':
            mask = cv2.blur(self.prefer_mask, (120, 120))
            mask = 0.8 * linear_adjust(mask, 0.5, 80) / 255
        else:
            raise NameError('Unexpected option for mask!')
        # cv2.imshow('mask', mask)
        # cv2.waitKey(0)
        return mask

    def _fake_threshold(self, gray_image):
        # self.bina_threshold = 127
        res = np.mean(gray_image)
        res += 20
        # print('Setting initial threshold as {}'.format(res))
        res = 127
        return res

    def _local_avg_brightness(self):
        ave_bright = np.zeros([self.qr_size,self.qr_size])
        for i in range(self.qr_size):
            for j in range(self.qr_size):
                start_i = max(0,(i-2)*self.module_size)
                end_i = min(self.render_size,(i+2)*self.module_size)
                start_j = max(0,(j-2)*self.module_size)
                end_j = min(self.render_size, (j + 2) * self.module_size)
                S = np.sum(np.sum(self.L_r[start_i:end_i,start_j:end_j]))
                ave_bright[i][j] = S/((end_i-start_i)*(end_j-start_j))
        return ave_bright

def _Max(m1, m2):
    return  (m1+m2)/2 + abs(m1-m2)/2

def _Min(m1, m2):
    return (m1+m2)/2 - abs(m1-m2)/2

if __name__ == '__main__':
    hard_num = [11, 62, 94, 3, 97]
    number = [ 1,30,5,33,48,74]
    for n in number:
        qart_file = './work_dirs/img{}/qart.jpg'.format(n)
        image_file = './input/img{}.jpg'.format(n)
        transfer_file = './work_dirs/img{}/StyleTransfer/run_time60.jpg'.format(n)
        save_file = './output/img{}.QVF.jpg'.format(n)
        R = Render(qart_file=qart_file, image_file=image_file,transfer_file=transfer_file, save_file=save_file,image_enhance=False)
        R.run_rendering()

