# coding: utf-8 -*-
# This file uses E2E-MLT and recognised character
# Author: Pinaki Nath Chowdhury <http://www.pinakinathc.me>
# Creative Commons (cc)
# CVPR Unit, Indian Statistical Institute, Kolkata
# Reference:
#  @article{buvsta2018e2e,
#   title={E2E-MLT-an unconstrained end-to-end method for multi-language scene text},
#   author={Bu{\v{s}}ta, Michal and Patel, Yash and Matas, Jiri},
#   journal={arXiv preprint arXiv:1801.09919},
#   year={2018}
# }

import cv2
import numpy as np
from models import ModelResNetSep2
import net_utils
from ocr_utils import ocr_image
import torch
import argparse
import os
from glob import glob

f = open('codec.txt', 'r', encoding='utf-8')
codec = f.readlines()[0]
f.close()

def create_boxes(box):
  boxes = []
  for xmin, xmax, ymin, ymax in box:
    boxes.append([ymin, xmax, ymin, xmin, ymax, xmin, ymax, xmax, 3])
  return np.array(boxes)

def get_character(im, box, args, net):
  """ This function actually uses the MLT OCR used by E2E for recognition.
  Args:
    im: input image in RGB format
    box: the top-left and bottom-right points of the image
    args: some parameters passed during input
    net: model is loaded in this variable

  Returns:
    text: Returns a string
  """
  text = ""
  with torch.no_grad():
    im_resized = im*1
    images = np.asarray([im_resized], dtype=np.float)
    images /= 128
    images -= 1
    im_data = net_utils.np_to_variable(images, is_cuda=args.cuda).permute(0, 3, 1, 2)

    boxes = create_boxes(box)

    for box in boxes:
      pts  = box[0:8]
      pts = pts.reshape(4, -1)
      try:
        det_text, conf, dec_s = ocr_image(net, codec, im_data, box)
      except:
        det_text = ""
        traceback.print_exc()
      text += det_text
    
    return text

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-cuda', type=int, default=1)
  parser.add_argument('-model', default='e2e-mlt.h5')
  parser.add_argument('-segm_thresh', default=0.5)
  parser.add_argument('-input_path', default=os.path.join("input_data",
    "*.png"))
  parser.add_argument('-output_path', default=os.path.join("output_data"))
  args = parser.parse_args()

  net = ModelResNetSep2(attention=True)
  net_utils.load_net(args.model, net)
  net = net.eval()

  if args.cuda:
    net = net.cuda()

  images = glob(args.input_path)
  for image_name in images:
  	img = cv2.imread(image_name)
  	M, N, _ = img.shape
  	box = [[0, M, 0, N]]
  	text = get_character(img, box, args, net)
  	print (text)
  	image_name = image_name.split("/")[-1][:-4]
  	cv2.imwrite(os.path.join(args.output_path, image_name+"_"+text+".png"),
      img)
