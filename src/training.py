import sys
import numpy as np
import matplotlib.pyplot as plt
import cebra
from PIL import Image
import cv2
import os
import torch
import itertools
import random
import gc
import pandas as pd

## Add an import for cebra_utils which is in the same directory as this file
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import cebra_utils

## Given the path to a tif file, return that as a 3d numpy array
# @param path: path to tif file
# @return: 3d numpy array, first array is time dimension
def load_tif(path):
    img = cv2.imreadmulti(path, flags=(cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH))[1]
    img = np.array(img)
    return img

## Loads the brain data from a given trial
def load_brain_data(parent_directory, trial_num, type='gcamp'):
    # Load the data
    data_path = os.path.join(parent_directory, 'trial_' + str(trial_num) + '/brain/' + type + '.tif')
    data = load_tif(data_path)
    return data

def load_pose_data(parent_directory, trial_num):
    # Load the data
    data_path = os.path.join(parent_directory, 'trial_' + str(trial_num) + '/anipose/videos/pose-3d/vid.csv')
    data = pd.read_csv(data_path)
    data = data.to_numpy()
    return data

## Load image embeddings from .npy files
def load_embedding_data(parent_directory, trial_num):
    # Load the data
    data_path = os.path.join(parent_directory, 'trial_' + str(trial_num) + '_embeddings.npy')
    data = np.load(data_path)
    return data

## Go through all trials and load the brain data for each trial
def load_all_brain_data_trials(parent_directory, type='gcamp'):
    # Get the number of trials
    num_trials = len([x for x in os.listdir(parent_directory) if 'trial_' in x])
    # Load the data
    return np.array([load_brain_data(parent_directory, trial_num, type) for trial_num in range(num_trials)])
    
## Takes a numpy array in and returns a memory mapped numpy array
# @param arr: numpy array to be memory mapped
# @param path: path to save the memory mapped array to
# @return: memory mapped numpy array
def memmap(arr, path):
    # Save the array
    np.save(path, arr)
    # Load the array
    return np.load(path, mmap_mode='r')

## Creates a CEBRA multisession Data Loader with the given data, feature data and brain data must share first 2 dimensions
def init_dataloader(brain_data, feature_data, num_steps, time_offset, conditional, batch_size=1, cebra_offset=None ):
    datasets = []
    print('loading data')
    for session in zip(brain_data, feature_data):
        brain_data_tensor  = torch.FloatTensor(session[0]).unsqueeze(1)
        feature_data_tensor = torch.FloatTensor(session[1])
        datasets.append(cebra.data.datasets.TensorDataset(brain_data_tensor, continuous=feature_data_tensor, offset=cebra_offset))
    dataset_collection = cebra.data.datasets.DatasetCollection(*datasets)
    return cebra.data.multi_session.ContinuousMultiSessionDataLoader(
        dataset=dataset_collection,
        batch_size=batch_size,
        num_steps=num_steps,
        time_offset=time_offset,
        conditional=conditional,
    ).to('cuda')

## initialize a single session dataloader
def init_single_session_dataloader(brain_data, feature_data, discrete_data, num_steps, time_offset, conditional, batch_size=1, cebra_offset=None ):
    brain_data_tensor  = torch.FloatTensor(brain_data)
    feature_data_tensor = torch.FloatTensor(feature_data)
    discrete_data_tensor = torch.LongTensor(discrete_data)
    dataset = cebra.data.datasets.TensorDataset(brain_data_tensor, continuous=feature_data_tensor, discrete=discrete_data_tensor, offset=cebra_offset)
    return cebra.data.single_session.MixedDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_steps=num_steps,
        time_offset=time_offset,
        conditional=conditional,
    )

## Creat and train the model in partial batches of data
def train_model(dataloader, loader_type ,input_size, hidden_units, output_dimension, model_name, device, output_model_path, saved_model = None):
    print('Creating model')
    ## create list of models
    if loader_type == 'multisession':
        model = torch.nn.ModuleList([
        cebra.models.init(model_name, input_size,
                            hidden_units, output_dimension, True)
        for _ in range(len(list(dataloader.dataset.iter_sessions())))
        ]).to(device)
        if saved_model is not None:
            model.__setstate__(saved_model)
    elif loader_type == 'single':
        model = cebra.models.init(model_name, input_size,
                            hidden_units, output_dimension, True).to(device)
        if saved_model is not None:
            model.__setstate__(saved_model)
    else: 
        raise Exception('Invalid loader type')

    ## Load criterion
    criterion = cebra.models.criterions.LearnableCosineInfoNCE(temperature=2, min_temperature=0.2).to(device)
    start_state = criterion.state_dict()
    ## Load optimizer
    optimizer = torch.optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=0.001)

    print('Loading solver')
    ## Load solver and train on first slice of data
    if loader_type == 'multisession':
        solver = cebra.solver.MultiSessionSolver(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            tqdm_on=True,
        ).to(device)

    elif loader_type == 'single':
        solver = cebra.solver.SingleSessionSolver(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            tqdm_on=True,
        ).to(device)

    print('Training on slice 1')
    solver.fit(dataloader.to(device),
                save_frequency=500,
                logdir='runs',)
 
    print('Training complete, saving model')
    torch.save(solver, output_model_path)
    return solver

## From a given path load and then flatten the trial and brain data and do a test / train split
# @param data_path: path to the data
# @param split: the percentage of data to be used for training
# @return: flattened brain data, flattened feature data, discrete training data, testing brain data, testing feature data
def load_test_train(data_path, split, image_size=64, use_pose=False, embedding_path=None):
    num_trials = len([x for x in os.listdir(data_path) if 'trial_' in x])
    brain_data = [load_brain_data(data_path, x) for x in range(num_trials)]
    if use_pose == True:
        feature_data = [load_pose_data(data_path, x) for x in range(len(brain_data))]
    else:
        feature_data = [load_embedding_data(embedding_path, x) for x in range(len(brain_data))]
    # flatten the first dimension of brain data
    # n x 288 x 256 x 256 -> n * 288 x 256 x 256
    # before flattening take train test split
    training_brain_data = brain_data[:int(len(brain_data) * split)]
    testing_brain_data = brain_data[int(len(brain_data) * split):]
    flattened_brain_data = np.concatenate(training_brain_data, axis=0)
    training_feature_data = feature_data[:int(len(feature_data) * split)]
    testing_feature_data = feature_data[int(len(feature_data) * split):]
    flattened_feature_data = np.concatenate(training_feature_data, axis=0)
    flattened_brain_data = np.array([cv2.resize(img, (64, 64)) for img in flattened_brain_data])
    # create a discrete tensor for the brain data of 0-288 repeating
    discrete_training_data = np.concatenate(np.array([np.arange(288) for _ in range(len(training_brain_data))]), axis=0)
    return flattened_brain_data, flattened_feature_data, discrete_training_data, testing_brain_data, testing_feature_data

## Main training loop
if __name__ == "__main__":
    # Set this to true if the model takes 2d input
    Model_2D = True
    #Model_Name = 'offset1-model-v5'
    Model_Name = 'ViT-16-v1'
    Image_Size = 64

    # Load the data
    data_path = '/mnt/teams/TM_Lab/Tony/water_reaching/Data/rig1_data/processed/FRM1_2023-07-07_1'
    embedding_path = '/home/murph_4090ws/Documents/Water_Reaching_Classifier/FRM1_7-07'
    print('Loading data')
    flattened_brain_data, flattened_feature_data, discrete_data, _, _ = load_test_train(data_path, 1, Image_Size, use_pose=False, embedding_path=embedding_path)
    # concatenate the data

    ## If the trained model does not take 2d input reshape the data by flattening last two dimensions into one vector
    if Model_2D == False:
        flattened_brain_data = np.array([img.flatten() for img in flattened_brain_data])
        input_size = Image_Size * Image_Size
    else:
        input_size = Image_Size
    

    loader = init_single_session_dataloader(
        brain_data=flattened_brain_data,
        feature_data=flattened_feature_data,
        discrete_data=discrete_data,
        num_steps=5000,
        time_offset=30,
        conditional='time_delta',
        batch_size=1024,
        cebra_offset=cebra.data.datatypes.Offset(0,1),
    )

    # For ViT model we need to reshape the data to be 256 x 256 x 3 as the model expects 3 channels, so we use a 1,2 offset
    model = train_model(
        loader,
        loader_type='single',
        input_size=input_size,
        hidden_units=2,
        output_dimension=8,
        model_name=Model_Name,
        device='cuda',
        output_model_path='ViTModel_offset1_embedding2.pth',
    )

