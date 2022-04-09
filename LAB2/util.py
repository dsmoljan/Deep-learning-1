import os
import math

import numpy as np
from numpy.lib.shape_base import column_stack
import skimage as ski
import skimage.io
import matplotlib.pyplot as plt
from torch import nn
import pdb
import torch

import itertools
import collections

CLASS_NAMES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def draw_conv_filters_pytorch(weights, weight_decay,save_dir):
  w = weights.copy()
  num_filters = w.shape[0]
  num_channels = w.shape[1]
  k = w.shape[2]
  assert w.shape[3] == w.shape[2]
  w = w.transpose(2, 3, 1, 0)
  w -= w.min()
  w /= w.max()
  border = 1
  cols = 8
  rows = math.ceil(num_filters / cols)
  width = cols * k + (cols-1) * border
  height = rows * k + (rows-1) * border
  img = np.zeros([height, width, num_channels])
  for i in range(num_filters):
    r = int(i / cols) * (k + border)
    c = int(i % cols) * (k + border)
    img[r:r+k,c:c+k,:] = w[:,:,:,i]
  filename = 'conv1_lambda_%.3f.png' % (weight_decay)
  ski.io.imsave(os.path.join(save_dir, filename), img)

def plot_training_progress(save_dir, data):
  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))

  linewidth = 2
  legend_size = 10
  train_color = 'm'
  val_color = 'c'

  num_points = len(data['train_loss'])
  x_data = np.linspace(1, num_points, num_points)
  ax1.set_title('Cross-entropy loss')
  ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax1.legend(loc='upper right', fontsize=legend_size)
  ax2.set_title('Average class accuracy')
  ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax2.legend(loc='upper left', fontsize=legend_size)
  ax3.set_title('Learning rate')
  ax3.plot(x_data, data['lr'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='learning_rate')
  ax3.legend(loc='upper left', fontsize=legend_size)

  save_path = os.path.join(save_dir, 'training_plot.png')
  print('Plotting in: ', save_path)
  plt.savefig(save_path)


# za predani istrenirani model, x_test i y_test, pronalazi 20 slika sa najvećim gubitkom, 
# te ih vraća, zajedno sa 3 razreda za koje im je model dao najveću vjerojatnost
def find_max_loss_images(model, x_test, y_test):
  images = collections.OrderedDict()
  n = 20
  CE_loss = nn.CrossEntropyLoss()
  activation = nn.Softmax(dim = 1)

  for i in range(len(x_test)):
    with torch.no_grad():
      y_out = model.forward(x_test[i, :].unsqueeze(dim = 0))
      #pdb.set_trace()
      loss = float(CE_loss(y_out, y_test[i].unsqueeze(dim=0)))
      images.update({i:loss})
  
  images = dict(sorted(images.items(), key=lambda item: item[1], reverse = True))
  max_loss_images = itertools.islice(images.items(), 0, n)

  final_dict = dict()
  for key, value in max_loss_images:
    with torch.no_grad():
      y_out = model.forward(x_test[key, :].unsqueeze(dim = 0))
      y_pred = torch.argmax(activation(y_out), dim = 1)
      predictions_string = ""
      predictions_string +=  "Actual:" + CLASS_NAMES[int(y_test[key])] + ", Predicted: " + CLASS_NAMES[y_pred]
      final_dict.update({predictions_string:x_test[key]})
  
  return final_dict

def draw_image(img, mean, std, title):
  img = img.transpose(1, 2, 0)
  img *= std
  img += mean
  img = img.astype(np.uint8)
  ski.io.imshow(img)
  ski.io.show()

