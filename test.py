from comet_ml import Experiment

from data.datasets import TextDataset, ParallelDataset
from stargan_tst.models.StarGANModel import StarGANModel
from stargan_tst.models.GeneratorModel import GeneratorModel
from stargan_tst.models.DiscriminatorModel import DiscriminatorModel
from eval import *
from utils.utils import *

import argparse
import logging
import os
import numpy as np, pandas as pd
import random

import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

''' 
    ----- ----- ----- ----- ----- ----- ----- -----
                    PARSING PARAMs       
    ----- ----- ----- ----- ----- ----- ----- -----
'''

parser = argparse.ArgumentParser()

# basic parameters
parser.add_argument('--n_styles',  type=int, dest="n_styles",  default=2, help='Number of different styles in the dataset -1.')
parser.add_argument('--lang', type=str, dest="lang", default='en', help='Dataset language.')
parser.add_argument('--max_samples_test',  type=int, dest="max_samples_test",  default=None, help='Max number of examples to retain from the test set. None for all available examples.')

parser.add_argument('--path_db_test', type=str, dest="path_db_test", help='Path to dataset for testing.')
parser.add_argument('--path_to_references', type=str, nargs='+', dest="path_to_references", help='List of paths to reference files, one per style.')
parser.add_argument('--n_references',  type=int, dest="n_references",  default=None, help='Number of human references for test.')
parser.add_argument('--lowercase_ref', action='store_true', dest="lowercase_ref", default=False, help='Whether to lowercase references.')
parser.add_argument('--bertscore', action='store_true', dest="bertscore", default=True, help='Whether to compute BERTScore metric.')

parser.add_argument('--max_sequence_length', type=int,  dest="max_sequence_length", default=64, help='Max sequence length')

# Training arguments
parser.add_argument('--batch_size', type=int,  dest="batch_size",  default=64,     help='Batch size used during training.')
parser.add_argument('--num_workers',type=int,  dest="num_workers", default=4,     help='Number of workers used for dataloaders.')
parser.add_argument('--pin_memory', action='store_true', dest="pin_memory",  default=False, help='Whether to pin memory for data on GPU during data loading.')

parser.add_argument('--use_cuda_if_available', action='store_true', dest="use_cuda_if_available", default=False, help='Whether to use GPU if available.')

parser.add_argument('--generator_model_tag', type=str, dest="generator_model_tag", help='The tag of the model for the generator (e.g., "facebook/bart-base").')
parser.add_argument('--discriminator_model_tag', type=str, dest="discriminator_model_tag", help='The tag of the model discriminator (e.g., "distilbert-base-cased").')

# arguments for saving the model and running test
parser.add_argument('--save_base_folder', type=str, dest="save_base_folder", help='The folder to use as base path to store model checkpoints')
parser.add_argument('--from_pretrained', type=str, dest="from_pretrained", default=None, help='The folder to use as base path to load model checkpoints')
parser.add_argument('--test_id', type=str, dest="test_id", default=None, help='Test ID')

# arguments for comet
parser.add_argument('--comet_logging', action='store_true', dest="comet_logging",   default=False, help='Set flag to enable comet logging')
parser.add_argument('--comet_key',       type=str,  dest="comet_key",       default=None,  help='Comet API key to log some metrics')
parser.add_argument('--comet_workspace', type=str,  dest="comet_workspace", default=None,  help='Comet workspace name (usually username in Comet, used only if comet_key is not None')
parser.add_argument('--comet_project_name',  type=str,  dest="comet_project_name",  default=None,  help='Comet experiment name (used only if comet_key is not None')
parser.add_argument('--exp_group', type=str, dest="exp_group", default=None, help='To group experiments on Comet')

args = parser.parse_args()

max_samples_test = args.max_samples_test

hyper_params = {}
print ("Arguments summary: \n ")
for key,value in vars(args).items():
    hyper_params[key] = value
    print (f"\t{key}:\t\t{value}")

# target styles randomly chosen
def assign_target_style(source_style, n_styles):
    target_style = random.randint(0, n_styles-1)
    while source_style==target_style:
        target_style = random.randint(0, n_styles-1)
    return target_style

if args.path_to_references is not None:    
    ds_test = ParallelDataset(validation_file=args.path_db_test, style_files=args.path_to_references, max_dataset_samples=args.max_samples_test)
else:
    ds_test = TextDataset(file_path=args.path_db_test, max_samples=args.max_samples_test, target_label_fn=assign_target_style, n_styles=args.n_styles)

print(f"Testing data  : {len(ds_test)}")

if args.path_to_references is not None:
    dl_test = DataLoader(ds_test,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                collate_fn = ParallelDataset.customCollate)
    del ds_test
else:
    dl_test = DataLoader(ds_test,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=args.pin_memory)
    del ds_test

print (f"Testing lenght (batches): {len(dl_test)}")

if args.comet_logging :
    experiment = Experiment(api_key=args.comet_key,
                            project_name=args.comet_project_name,
                            workspace=args.comet_workspace)
    experiment.log_parameters(hyper_params)
else:
    experiment = None

if args.use_cuda_if_available:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
else:
    device = torch.device("cpu")

if args.from_pretrained is not None:
    if 'epoch' in args.from_pretrained:
        checkpoints_paths = [args.from_pretrained]
    else:
        checkpoints_paths = sorted([dir for dir in os.listdir(args.from_pretrained) if dir.startswith('epoch')], key=lambda dir: int(dir.split('_')[1]))
        checkpoints_paths = [args.from_pretrained+path+'/' for path in checkpoints_paths]
    epochs = [int(path.split('_')[-1][:-1]) for path in checkpoints_paths]
else:
    checkpoints_paths, epochs = [''], [0]

''' 
    ----- ----- ----- ----- ----- ----- ----- -----
                        TEST       
    ----- ----- ----- ----- ----- ----- ----- -----
'''
if args.from_pretrained is not None:
    G = GeneratorModel(args.generator_model_tag, f'{args.from_pretrained}Generator/', num_domains=args.n_styles, max_seq_length=args.max_sequence_length)
    D = DiscriminatorModel.DiscriminatorModel(args.discriminator_model_tag, f'{args.from_pretrained}Discriminator/', num_domains=args.n_styles, max_seq_length=args.max_sequence_length)
    print('Generator e Discriminator pre-addestrati caricati correttamente')
else:
    G = GeneratorModel(args.generator_model_tag, num_domains=args.n_styles, max_seq_length=args.max_sequence_length)
    D = DiscriminatorModel.DiscriminatorModel(args.discriminator_model_tag, num_domains=args.n_styles, max_seq_length=args.max_sequence_length)
    print('Generator e Discriminator inizializzati con pesi casuali')

for checkpoint, epoch in zip(checkpoints_paths, epochs):
    if args.from_pretrained is not None:
        G = GeneratorModel(args.generator_model_tag, f'{checkpoint}Generator/', num_domains=args.n_styles, max_seq_length=args.max_sequence_length)
        print('Generator pretrained model loaded correctly')
        D = DiscriminatorModel.DiscriminatorModel(args.discriminator_model_tag, f'{checkpoint}Discriminator/', num_domains=args.n_styles, max_seq_length=args.max_sequence_length)
        print('Discriminator pretrained model loaded correctly')
    else:
        G = GeneratorModel(args.generator_model_tag, num_domains=args.n_styles, max_seq_length=args.max_sequence_length)
        print('Generator pretrained model not loaded - Initial weights will be used')
        D = DiscriminatorModel.DiscriminatorModel(args.discriminator_model_tag, num_domains=args.n_styles, max_seq_length=args.max_sequence_length)
        print('Discriminator pretrained models not loaded - Initial weights will be used')
    
    stargan = StarGANModel(G, D, device=device)
    evaluator = Evaluator(stargan, args, experiment)

    if args.path_to_references is not None:
        evaluator.run_eval_ref(epoch, epoch, 'test', dl_test)
    else:
        evaluator.run_eval_no_ref(epoch, epoch, 'test', dl_test)

print('End checkpoint(s) test...')
