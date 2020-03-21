import cv2
import numpy as np
from PIL import Image
from PIL import ImageEnhance


def img_preprocess(img):
    enh_col = ImageEnhance.Color(img)
    color = 1.5
    image_colored = enh_col.enhance(color)
    image_colored.show()
    return img

def find_noise_map(L_p , L_b , version , module_size):
    qr_size = version * 4 + 17
    differ_map = abs(L_p - L_b)
    noise_map = np.zeros([qr_size,qr_size])
    for i in range(qr_size):
        for j in range(qr_size):
            sample = differ_map[i*module_size:(i+1)*module_size-1, j*module_size:(j+1)*module_size-1 ]
            sum = np.sum(np.sum(sample))
            noise_map[i][j]=sum
    # find the max element
    n_max = []
    for i in range(len(noise_map)):
        n_max.append(max(noise_map[i]))
    n_max = max(n_max)
    noise_map = noise_map/n_max
    return noise_map

def add_funct(img , version=5 , module_size=9):
    qr_size = version * 4 + 17
    render_size = qr_size * module_size
    qr_funct = full_resize(cv2.imread('./qr_funct.jpg'), version, module_size)
    img_funct_added=img
    for i in range(3):
        img_funct_added[:, :, i] = img[:, :, i]*qr_funct/255
    return img_funct_added

def add_surrounding_bina(qart, input, version=6, module_size=9, rotation=0):
    qr_size = version * 4 + 17
    ori_mask = np.zeros((qr_size,qr_size))
    ori_mask[0:9,0:9] = 1
    ori_mask[0:9, -9:] = 1
    ori_mask[-9:, 0:9] = 1
    ori_mask[-9:, -9:] = 1
    ori_mask[0:3, :] = 1
    ori_mask[-3:, :] = 1
    ori_mask[:, 0:3] = 1
    ori_mask[:, -3:] = 1
    ori_mask[-8:-5, -8:-5] = 1
    ori_mask[6, :] = 1
    ori_mask[:, 6] = 1

    white_mask = full_resize(ori_mask*255, version, module_size)
    white_mask = np.dstack([white_mask, white_mask, white_mask])
    full_qart = full_resize(qart, version, module_size)
    full_qart = np.dstack([ full_qart, full_qart, full_qart])

    out = (1.-white_mask/255)*input + white_mask/255.0*full_qart

    return out / 255

def add_function_code(qart, L_r, version=5, module_size=9, rotation = 0):
    qr_size = version * 4 + 17
    p_white = 0.3
    funct_mask = np.zeros((qr_size,qr_size))
    funct_mask[0:9, 0:9] = 1
    funct_mask[0:9, -8:] = 1
    funct_mask[-8:, 0:9] = 1
    funct_mask[-9:-4, -9:-4] = 1
    funct_mask[6, :] = 1
    funct_mask[:, 6] = 1
    funct_mask = np.rot90(funct_mask*255, k=4-rotation)
    funct_mask = full_resize(funct_mask, version, module_size) / 255
    big_qart = full_resize(qart, version, module_size) / 255
    L_funct_added = L_r * (1 - funct_mask) + (p_white * 255 + (1 - p_white) * L_r)* big_qart * funct_mask
    return L_funct_added

def full_resize(img , version=6 , module_size=9 , binarize = True):
    """
    return a binary image with one channel, resized with sharp edge
    """
    qr_size = version * 4 + 17
    render_size = qr_size * module_size
    m = module_size
    img_full = np.zeros((render_size,render_size))
    black = np.zeros((module_size,module_size))
    white = 255*np.ones((module_size,module_size))
    if len(img.shape) == 3:
        img = img[:, :, 0]
    for x in range(qr_size):
        for y in range(qr_size):
            if binarize:
                if img[x][y] < 127:
                    img_full[x * m:x * m + m, y * m:y * m + m] = black
                else:
                    img_full[x * m:x * m + m, y * m:y * m + m] = white
            else:
                img_full[x * m:x * m + m, y * m:y * m + m] = img[x][y]*np.ones([module_size , module_size ])
    return img_full

def linear_adjust(img, a, b, para_option ='free'):
    size = img.shape[1]
    if para_option == 'fix_average':
        img = np.array(img)
        m = img.mean()
        b = (1-a)*m
    elif para_option == 'free':
        b = b
    else:
        raise NameError
    img = a*img+b
    img = _Min(img,254*np.ones([size,size]))
    img = _Max(img, np.zeros([size,size]))
    return img

def _Max(m1, m2):
    return  (m1+m2)/2 + abs(m1-m2)/2

def _Min(m1, m2):
    return (m1+m2)/2 - abs(m1-m2)/2

def find_mass_center(gray_image, norm=True, invert=True):
    # image should be gray scale
    ret, thresh = cv2.threshold(gray_image, 127, 255, 0)
    if invert:
        thresh = 255 - thresh
    M = cv2.moments(thresh)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    if norm:
        cX /= gray_image.shape[0]
        cY /= gray_image.shape[0]
    return cX, cY

def image_enhance(input, mode='log', output='./output.jpg'):
    # cv2.imshow('input', input)
    if 'log' in mode:
        def log(c, img):
            output_img = c * np.log(1.0 + img)
            output_img = np.uint8(output_img + 0.5)
            return output_img
        out = log(20, input)
        # cv2.imshow('log_output', out)

    elif 'gamma' in mode:
        def gamma(img, c, v):
            lut = np.zeros(256, dtype=np.float32)
            for i in range(256):
                lut[i] = c * i ** v
            output_img = cv2.LUT(img, lut)
            output_img = np.uint8(output_img + 0.5)
            return output_img
        out = gamma(input, 0.00000005, 4.0)
        # cv2.imshow('gamma_output', out)

    elif 'lapras' in mode:
        kernel = np.array([[0, -1, 0], [0, 5, 0], [0, -1, 0]])
        out = cv2.filter2D(input, -1, kernel)
        # cv2.imshow('lapras_output', out)

    elif 'equalization' in mode:
        out = input.copy()
        for j in range(3):
            out[:, :, j] = cv2.equalizeHist(input[:, :, j])
        # cv2.imshow('equalization_output', out)

    else:
        raise NameError
    out = (0.5*input + 0.5*out).astype(np.uint8)
    # cv2.waitKey(0)
    return out


if __name__ == '__main__':
    img = cv2.imread('./input/img92.jpg')
    image_enhance(input=img, mode='equalization', output='./output.jpg')
