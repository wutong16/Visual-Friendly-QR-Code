# Visual-Friendly QR Code

Pytorch implementation of Visual-friendly Aesthetic QR Code Generation using Image Style Transfer, submitted to IEEE Transactions on Multimedia, under review.

## Installation
Please, install the following packages
* numpy
* PIL
* Pytorch >= 0.4
* torchvision
* OpenCV-Python (3.4.2 is recommended)


## Get Started 

A quick start to run the demos, and the results would be saved under `./output` 
```
python pipeline.py --render_only 
```
Change the image idx among `<1,5,30,33,48,74>` to re-produce the examples in the paper
```
python pipeline.py -i "./input/img74.jpg" -u 'https://cn.bing.com/' --render_only 
```
Try to use a mask to indicate the important region
```
python pipeline.py -i "./input/img22.jpg" -m './input/prefer_mask22.jpg' --render_only 
```
Try your own background image or urls and generate a Visual-Friendly QR from scratch by
```
python pipeline.py -i <path to your image> -u <your url>
```  

## Reference
This project is based on
* https://github.com/7sDream/pyqart
* https://github.com/pytorch/tutorials/blob/master/advanced_source/neural_style_tutorial.py
## Contact

This repo is currently maintained by Tong WU (wutong16.thu@gmail.com)

* The code is not fully tested and may contain unknown bugs, feel free to contact with us if you encounter with any issues!
* We have tested the readability of the generated QR code using Alipay.
