'''
use this file to analyse results
load pre-trained model (trained in cluster)
'''

# importing python utility libraries
import os, sys, random, io, urllib
from datetime import datetime

# importing pytorch libraries
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

# importing data science libraries
import pandas as pd
import random as rd
import numpy as np

# importing python plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from IPython.display import Image, display

#AAE parts
from Encoder import Encoder
from Decoder import Decoder
from Discriminator import Discriminator

USE_CUDA = False

latentVecDim = 10

'''
The raw.githubusercontent.com domain is used to serve unprocessed versions of files stored in GitHub repositories.
 If you browse to a file on GitHub and then click the Raw link, that's where you'll go.
'''

# print current Python version
now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
print('[LOG {}] The Python version: {}'.format(now, sys.version))

# print current PyTorch version
now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
print('[LOG {}] The PyTorch version: {}'.format(now, torch.__version__))

# init deterministic seed
seed_value = 1234
rd.seed(seed_value) # set random seed
np.random.seed(seed_value) # set numpy seed
torch.manual_seed(seed_value) # set pytorch seed CPU
if (torch.backends.cudnn.version() != None and USE_CUDA == True):
    torch.cuda.manual_seed(seed_value) # set pytorch seed GPU

if not os.path.exists('./results'): os.makedirs('./results')  # create results directory

# load the synthetic ERP dataset
#ori_dataset = pd.read_csv('./data/fraud_dataset_v2.csv')

# load the synthetic ERP dataset
# load the synthetic ERP dataset
# Modifica per usare il file locale scaricato da reproduce_baseline.py se esiste
local_csv = 'data/fraud_dataset_v2.csv'
if os.path.exists(local_csv):
    print(f"[GenerateDatasets] Loading local dataset: {local_csv}")
    ori_dataset = pd.read_csv(local_csv)
else:
    url = 'https://raw.githubusercontent.com/GitiHubi/deepAI/master/data/fraud_dataset_v2.csv'
    print(f"[GenerateDatasets] Downloading from URL: {url}")
    ori_dataset = pd.read_csv(url)

# remove the "ground-truth" label information for the following steps of the lab
label = ori_dataset.pop('label')

#### REMOVE THIS BLOCK = 1 ################################################################33
'''
# prepare to plot posting key and general ledger account side by side
fig, ax = plt.subplots(1,2)
fig.set_figwidth(12)

# plot the distribution of the posting key attribute
g = sns.countplot(x=ori_dataset.loc[label=='regular', 'BSCHL'], ax=ax[0])
g.set_xticklabels(g.get_xticklabels(), rotation=0)
g.set_title('Distribution of BSCHL attribute values')

# plot the distribution of the general ledger account attribute
g = sns.countplot(x=ori_dataset.loc[label=='regular', 'HKONT'], ax=ax[1])
g.set_xticklabels(g.get_xticklabels(), rotation=0)
g.set_title('Distribution of HKONT attribute values');
#plt.show() # Plots created using seaborn need to be displayed like ordinary matplotlib plots. This can be done using the plt.show() command
fig.savefig("output.png")
'''
#### REMOVE THIS BLOCK = 1 ################################################################33

## PRE-PROCESSING of CATEGORICAL TRANSACTION ATTRIBUTES:
# select categorical attributes to be "one-hot" encoded
categorical_attr_names = ['KTOSL', 'PRCTR', 'BSCHL', 'HKONT', 'BUKRS', 'WAERS']
# encode categorical attributes into a binary one-hot encoded representation 
# Use sparse=True to save memory (approx 2.5GB for dense vs <100MB for sparse)
ori_dataset_categ_transformed = pd.get_dummies(ori_dataset[categorical_attr_names], sparse=True)

## PRE-PROCESSING of NUMERICAL TRANSACTION ATTRIBUTES:  In order to faster approach a potential global minimum it is good practice to scale and normalize numerical input values prior to network training
# select "DMBTR" vs. "WRBTR" attribute
numeric_attr_names = ['DMBTR', 'WRBTR']
# add a small epsilon to eliminate zero values from data for log scaling
numeric_attr = ori_dataset[numeric_attr_names] + 1e-4
numeric_attr = numeric_attr.apply(np.log)
# normalize all numeric attributes to the range [0,1]
ori_dataset_numeric_attr = (numeric_attr - numeric_attr.min()) / (numeric_attr.max() - numeric_attr.min())

## merge cat and num atts
# merge categorical and numeric subsets
# sparse=True ensures we don't blow up memory here
ori_subset_transformed = pd.concat([ori_dataset_categ_transformed, ori_dataset_numeric_attr], axis = 1)

## PRETRAINED MODEL RESTORATION
mini_batch_size = 128

# creation of the imposed latent prior distribution
# define the number of gaussians
tau = 20
# define radius of each gaussian
radius = 0.8
# define the sigma of each gaussian
sigma = 0.01
# define the dimensionality of each gaussian
dim = latentVecDim

# REMOVED DEAD CODE: Synthetic gaussian sampling was not used in output and consumed memory.

# restore pretrained model checkpoint
encoder_model_name = 'https://github.com/jcssilva4/deep_learning_proj1/blob/master/models/10_20_20191231-22_36_34_ep_5000_encoder_model.pth?raw=true'
decoder_model_name = 'https://github.com/jcssilva4/deep_learning_proj1/blob/master/models/10_20_20191231-22_36_34_ep_5000_decoder_model.pth?raw=true'

# Read stored model from the remote location
encoder_bytes = urllib.request.urlopen(encoder_model_name)
decoder_bytes = urllib.request.urlopen(decoder_model_name)

# Load tensor from io.BytesIO object
encoder_buffer = io.BytesIO(encoder_bytes.read())
decoder_buffer = io.BytesIO(decoder_bytes.read())

# init training network classes / architectures
encoder_eval = Encoder(input_size=ori_subset_transformed.shape[1], hidden_size=[256, 64, 16, 4, latentVecDim])
decoder_eval = Decoder(output_size=ori_subset_transformed.shape[1], hidden_size=[latentVecDim, 4, 16, 64, 256])

# push to cuda if cudnn is available
if (torch.backends.cudnn.version() != None) and (USE_CUDA == True):
    encoder_eval = encoder_eval.cuda()
    decoder_eval = decoder_eval.cuda()
    
# load trained models
encoder_eval.load_state_dict(torch.load(encoder_buffer, map_location = 'cpu')) 
decoder_eval.load_state_dict(torch.load(decoder_buffer, map_location = 'cpu')) 

# VISUALIZE LATENT SPACE REPRESETATION
# set networks in evaluation mode (don't apply dropout)
encoder_eval.eval()
decoder_eval.eval()

# init batch count
batch_count = 0
z_enc_transactions_list = []

# iterate manually in batches to avoid dense conversion of the whole dataset
# This fixes the OOM crash on low-memory environments
total_rows = len(ori_subset_transformed)
for i in range(0, total_rows, mini_batch_size):
    batch_df = ori_subset_transformed.iloc[i : i + mini_batch_size]
    
    # Convert sparse/mixed batch to dense numpy then tensor
    # .values on sparse df block converts to dense automatically for the slice
    batch_np = batch_df.values.astype(float) 
    enc_transactions_batch = torch.from_numpy(batch_np).float()
    
    if (torch.backends.cudnn.version() != None) and (USE_CUDA == True):
        enc_transactions_batch = enc_transactions_batch.cuda()

    # determine latent space representation
    # wrap in no_grad to save memory for gradients
    with torch.no_grad():
        z_enc_transactions_batch = encoder_eval(enc_transactions_batch)
    
    z_enc_transactions_list.append(z_enc_transactions_batch.cpu())
    
    batch_count += 1
    if batch_count % 100 == 0:
        print(f"Processed batch {batch_count} / {total_rows // mini_batch_size}")

# convert to numpy array efficiently
z_enc_transactions_all = torch.cat(z_enc_transactions_list, dim=0).numpy()



# obtain regular transactions as well as global and local anomalies
regular_data = z_enc_transactions_all[label == 'regular']
global_outliers = z_enc_transactions_all[label == 'global']
local_outliers = z_enc_transactions_all[label == 'local']
print("|R| = " + str(len(regular_data)) + " ## |GA| = " + str(len(global_outliers)) + " ## |LA| = " + str(len(local_outliers)))

# save base article latent space for t = tau
'''
labels
0 stands for regular 
1 stands for global A
2 stands for local A
'''
base_dataSet = open("latent_data_sets/ldim" + str(latentVecDim) + "_tau" + str(tau) +"_basisLS.txt","w")
for L in ["\t".join(item) for item in regular_data.astype(str)]:
    base_dataSet.writelines(L + "\t0\n")
for L in ["\t".join(item) for item in global_outliers.astype(str)]:
    base_dataSet.writelines(L + "\t1\n")
for L in ["\t".join(item) for item in local_outliers.astype(str)]:
    base_dataSet.writelines(L + "\t2\n")
