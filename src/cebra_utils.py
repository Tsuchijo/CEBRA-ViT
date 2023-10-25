import sys
import numpy as np
import matplotlib.pyplot as plt
import cebra
from PIL import Image
import cv2
import os
import torch
import torch.nn.functional as F
import itertools
import random
from torch import nn
import cebra.models
import cebra.data
from cebra.models.model import _OffsetModel, ConvolutionalModelMixin
from vit_pytorch import ViT
from vit_pytorch import SimpleViT     
from x_transformers import Encoder


class ChangeOrderLayer(nn.Module):
    def __init__(self, first_dim = -2, second_dim = 1):
        super().__init__()
        self.first_dim = first_dim
        self.second_dim = second_dim
    def forward(self, x):
        return x.movedim(self.first_dim, self.second_dim).squeeze() # Permute dimensions 1 and 2

@cebra.models.register("convolutional-model-offset11")
class ConvulotionalModel1(_OffsetModel, ConvolutionalModelMixin):

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        super().__init__(
            ## create a model which goes from a 128 x 128 image to a 1d vector
            ## of length num_output
            ChangeOrderLayer(),
            nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Linear(1024, num_output),

            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    # ... and you can also redefine the forward method,
    # as you would for a typical pytorch model

    def get_offset(self):
        return cebra.data.Offset(2, 3)
    

@cebra.models.register("convolutional-model-30frame")
class ConvulotionalModel30Frame(_OffsetModel, ConvolutionalModelMixin):

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        super().__init__(
            ## create a model which goes from a 128 x 128 image to a 1d vector
            ## of length num_output
            ChangeOrderLayer(1,1),
            nn.Conv2d(30, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.GELU(),
            nn.Linear(1024, num_units),
            nn.GELU(),
            nn.Linear(num_units, num_output),

            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )
        def get_offset(self):
            return cebra.data.Offset(2, 3)

    # ... and you can also redefine the forward method,
    # as you would for a typical pytorch model

@cebra.models.register("ViT-16-v1")
## create a vision transformer model which goes from a 258 x 190 image to a 1d vector
## using 
class ViT16v1(_OffsetModel, ConvolutionalModelMixin):
    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        super().__init__(
            ## create a model which goes from a 128 x 128 image to a 1d vector
            ## of length num_output
            ViT(
                image_size = num_neurons,
                channels=1,
                patch_size = 8,
                num_classes = num_output,
                dim = 128,
                depth = num_units,
                heads = 4,
                mlp_dim = 128,
                dropout = 0.1,
                emb_dropout = 0.1
            ),

            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )
    def get_offset(self):
        return cebra.data.Offset(0, 1)
    def forward(self, x):
        x = x.movedim(1,2)
        return self.net(x)
    
@cebra.models.register("ViT-16-v2")
## Smaller patches and smaller hidden dimension
class ViT16v2(_OffsetModel, ConvolutionalModelMixin):
    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        super().__init__(
            ## create a model which goes from a 128 x 128 image to a 1d vector
            ## of length num_output
            ViT(
                image_size = num_neurons,
                channels=1,
                patch_size = 4,
                num_classes = num_output,
                dim = 64,
                depth = num_units,
                heads = 3,
                mlp_dim = 128,
                dropout = 0.1,
                emb_dropout = 0.1
            ),

            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )
    def get_offset(self):
        return cebra.data.Offset(0, 1)
    def forward(self, x):
        x = x.movedim(1,2)
        return self.net(x)
    
@cebra.models.register("SimpleViT-v1")
## Using SimpleVit Architecture
class SimpleViTv1(_OffsetModel, ConvolutionalModelMixin):
    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        super().__init__(
            ## create a model which goes from a 128 x 128 image to a 1d vector
            ## of length num_output
            ViT(
                image_size = num_neurons,
                channels=1,
                patch_size = 4,
                num_classes = num_output,
                dim = 128,
                depth = num_units,
                heads = 4,
                mlp_dim = 196,
            ),

            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )
    def get_offset(self):
        return cebra.data.Offset(0, 1)
    def forward(self, x):
        x = x.movedim(1,2)
        return self.net(x)
    

@cebra.models.register("SimpleViT-v2")
## Using SimpleVit Architecture
class SimpleViTv2(_OffsetModel, ConvolutionalModelMixin):
    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        super().__init__(
            ## create a model which goes from a 128 x 128 image to a 1d vector
            ## of length num_output
            SimpleViT(
                image_size = num_neurons,
                channels=1,
                patch_size = 8,
                num_classes = num_output,
                dim = 128,
                transformer = Encoder(
                    dim = 128,                  # set to be the same as the wrapper
                    depth = num_units,
                    heads = 4,
                    ff_glu = True,              # ex. feed forward GLU variant https://arxiv.org/abs/2002.05202
                    residual_attn = True        # ex. residual attention https://arxiv.org/abs/2012.11747
                )
            ),

            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )
    def get_offset(self):
        return cebra.data.Offset(0, 1)
    def forward(self, x):
        x = x.movedim(1,2)
        return self.net(x)
        
        

    
@cebra.models.register("CNN-offset1")
class CNNOffset1(_OffsetModel, ConvolutionalModelMixin):

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        super().__init__(
            ## create a model which goes from a 128 x 128 image to a 1d vector
            ## of length num_output
            nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(4096, num_units),
            nn.GELU(),
            nn.Linear(num_units, num_units),
            nn.GELU(),
            nn.Linear(num_units, num_output),

            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )
    def forward(self, x):
        x = x.squeeze()
        x = x.unsqueeze(1)
        return self.net(x)

    # ... and you can also redefine the forward method,
    # as you would for a typical pytorch model

    def get_offset(self):
        return cebra.data.Offset(0, 1)
    

@cebra.models.register("CNN-offset3")
class CNNOffset3(_OffsetModel, ConvolutionalModelMixin):

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        super().__init__(
            ## create a model which goes from a 128 x 128 image to a 1d vector
            ## of length num_output
            ChangeOrderLayer(),
            nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(1024, num_output),

            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    # ... and you can also redefine the forward method,
    # as you would for a typical pytorch model

    def get_offset(self):
        return cebra.data.Offset(1, 2)


class VideoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(VideoLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x has shape (batch_size, sequence_length, input_size)
        # Initialize hidden state and cell state
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

def process_brain(brain_seq):
    brain_seq = np.array(brain_seq)
    # if brain seq > 128 x 128, downsample
    if brain_seq.shape[1] > 128:
        # downsample to 128 x 128
        brain_seq = np.array([cv2.resize(brain_frame, (128, 128)) for brain_frame in brain_seq])

    flat_seq = np.array([(brain_frame.flatten()) for brain_frame in brain_seq])
    return flat_seq.astype(float)


## Takes a sliding window of data and then returns a list of windows
# data: list of data
# window_size: size of window
# returns: list of windows
def bin_data(data, window_size):
    output = []
    for i in range(len(data) - window_size + 1):
        output.append(data[i:i+window_size])
    return output


## Loads data from a folder of TIF files
# filepath: path to folder
# processor: function to process each image
# max: max images to load as a proportion of array size
# min: min images to load as a proportion of array size
# returns: list of processed images, list of filenames
def import_data(filepath, processor, min = 0, max = 1):
    output_data = []
    output_name = []
    path_list = os.listdir(filepath)
    path_list.sort()
    random.Random(4).shuffle(path_list)
    min_index = int(min * len(path_list))
    max_index = int(max * len(path_list))
    for file in itertools.islice(path_list, min_index, max_index):
     filename = os.fsdecode(file)
     if filename.endswith(".tif"):
         out = cv2.imreadmulti(filepath + '/' + filename)[1]
         output_data.append(processor(out))
         output_name.append(filename.split('.')[0])
     elif filename.endswith(".npy"):
         output_data.append(processor(np.load(filepath + '/' + filename)))
         output_name.append(filename.split('.')[0])
     else:
         continue
    return output_data, output_name

def flatten_data(data):
    return np.concatenate(data, axis=0)

def pad_data(data, pre, post):
    print (np.array(data).shape)
    data = np.array(data)
    t, x, y = data.shape
    padded = np.zeros((t + pre + post, x, y))
    padded[pre:pre + t, :, :] = data
    padded[:pre, :, :] = data[0:1, :, :]
    padded[pre + t:, :, :] = data[-1:, :, :]
    return padded

def generate_CEBRA_embeddings(model, data, session_id, offset = (2,3)):
    data_torch = torch.empty(0,offset[0] + offset[1],128,128).to('cuda')
    padded = pad_data(np.squeeze(data), offset[0], offset[1])
    print (padded.shape)
    for i, frame in enumerate(data):
        frame = torch.from_numpy(np.array(padded[i: i + offset[0] + offset[1]])).float().unsqueeze(0).to('cuda')
        data_torch = torch.cat((data_torch, frame), dim = 0)
    data_torch = data_torch.swapdims(-2, 1)
    # batch process data to save memory
    output = None
    model = model[session_id].eval().to('cuda')
    for i in range(0, len(data_torch), 100):
        embedding = model(data_torch[i:i+100]).detach().cpu().numpy().squeeze()
        if i == 0:
            output = embedding
        else:
            output = np.concatenate((output, embedding), axis = 0)
    return output

def load_model(model_path):
    #find available device
    saved_solver = torch.load(model_path)
    model = saved_solver.model
    return model

def reshape_frames(frames, shape_ref):
    shape_list = [np.shape(x)[0] for x in shape_ref]
    gen_video_list = []
    index = 0
    for shape in shape_list:
        gen_video_list.append((frames[index : index + shape]))
        index += shape
    return gen_video_list

#choose a random window of set size from the data deterministically based on seed
def choose_random_window( window_size, seed, data):
    random.seed(seed)
    start = random.randint(0, len(data) - window_size)
    return data[start:start+window_size]

def choose_first_second( window_size, data):
    return data[0:0+window_size]

def normalize_array(in_array):
    return np.array([x / np.linalg.norm(x) for x in in_array])