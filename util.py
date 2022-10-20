# Copyright 2022 Google LLC

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

""" The file containing utility functions required for running the code. """

import numpy as np
import torch
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import pandas as pd
import torch.nn.functional as F
import math
import os



def load_dataset(args, runid):
		with open('datasets/{0}/split.pickle'.format(args.ds_name), 'rb') as handle:
				dataset = pickle.load(handle)
		return dataset

def fetch_model(args, num_groups, dataset, class_counts):
		from model import WassAndEntropy
		return WassAndEntropy(args, num_groups, dataset, class_counts)

def generate_batched_loader(args, dataset):
		dataloader = {}
		dataloader["train"] = loader(args, dataset["train"], args.b_size)
		dataloader["tta"] = loader(args, dataset["train"], args.tta_bs)					  # tta is the test adaptation phase, increasing batch size for training data to reduce variance in the estimation of the constraints 
		dataloader["val"] = loader(args, dataset["val"], dataset["val"][0].shape[0])      # utilizing entire val set in a single batch
		# add_random_noise(dataset)
		dataloader["test"] = loader(args, dataset["test"], dataset["test"][0].shape[0])   # utilizing entire test set in a single batch
		return dataloader

def loader(args, a, batch_size):
		a[0] = remove_nans(a[0])
		_b = torch.utils.data.TensorDataset(torch.FloatTensor(a[0]), torch.LongTensor(a[1]), torch.LongTensor(a[2]))
		_loader = torch.utils.data.DataLoader(_b, batch_size=batch_size, shuffle=True)
		return _loader

def remove_nans(matrix):
		matrix[np.isnan(matrix)] = 0.0
		return matrix

def compute_accuracy(y_pred, y_true):
		return accuracy_score(y_true, y_pred)

def compute_equ_odds(actual_labels, y_pred, protected_labels, non_protected_labels):
		def true_positive_parity(actual_labels, y_pred, protected_labels, non_protected_labels):
				protected_ops = y_pred[np.bitwise_and(protected_labels == 1, actual_labels == 1)]
				if len(protected_ops) > 0:
						protected_prob = sum(protected_ops)/len(protected_ops)
				else:
						protected_prob = 0.0

				non_protected_ops = y_pred[np.bitwise_and(non_protected_labels == 1, actual_labels == 1)]
				if len(non_protected_ops) > 0:
						non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)
				else:
						non_protected_prob = 0.0
				return abs(protected_prob - non_protected_prob)

		def false_positive_parity(actual_labels, y_pred, protected_labels, non_protected_labels):
				protected_ops = y_pred[np.bitwise_and(protected_labels == 1, actual_labels == 0)]
				if len(protected_ops) > 0:
						protected_prob = sum(protected_ops)/len(protected_ops)
				else:
						protected_prob = 0.0
		
				non_protected_ops = y_pred[np.bitwise_and(non_protected_labels == 1, actual_labels == 0)]
				if len(non_protected_ops) > 0:
						non_protected_prob = sum(non_protected_ops)/len(non_protected_ops)
				else:
						non_protected_prob = 0.0
				return abs(protected_prob - non_protected_prob)

		def equalized_odds(actual_labels, y_pred, protected_labels, non_protected_labels):
				return true_positive_parity(actual_labels, y_pred, protected_labels, non_protected_labels) + \
								false_positive_parity(actual_labels, y_pred, protected_labels, non_protected_labels)

		equ_odds = equalized_odds(actual_labels, y_pred, protected_labels, non_protected_labels)		
		return equ_odds


def compute_metrics(y_pred, y_true, s, y_pred_probs):
		metrics = {}
		metrics["accuracy"] = round(100 * compute_accuracy(y_pred, y_true), 4)
		equ_odds = compute_equ_odds(y_true, y_pred, s, 1 - s)
		metrics["EOdds"] = round(equ_odds, 4) * 0.5 			# note the use of the factor of 0.5 in the equalized odds
		return metrics


def save_results_as_csv(run_type, metrics, runid, my_file, file_path):
		# run-type shows whether the run is validation or test
		results = {'run_type': run_type, 'runid': runid, "accuracy": metrics["accuracy"], "EOdds": metrics["EOdds"]}
		columns = ['run_type', 'runid', 'accuracy', 'EOdds']

		if my_file.is_file():
			df = pd.read_csv(file_path, sep='\t')
			df = df.append(results, ignore_index=True)
			df.to_csv(file_path, sep='\t', encoding='utf-8', index=False)

		else:
			df = pd.DataFrame(columns=columns)
			df = df.append(results, ignore_index=True)
			df.to_csv(file_path, sep='\t', encoding='utf-8', index=False)


def average_results(df):
		train_df = df.loc[df['run_type'] == "train"]
		average_train_df_mean = train_df.mean(axis=0, numeric_only=True)
		average_train_df_std = train_df.std(axis=0, ddof=0, numeric_only=True)

		val_df = df.loc[df['run_type'] == "validation"]
		average_val_df_mean = val_df.mean(axis=0, numeric_only=True)
		average_val_df_std = val_df.std(axis=0, ddof=0, numeric_only=True)

		test_df = df.loc[df['run_type'] == "test"]
		average_test_df_mean = test_df.mean(axis=0, numeric_only=True)
		average_test_df_std = test_df.std(axis=0, ddof=0, numeric_only=True)

		print("Training -> accuracy = ", round(average_train_df_mean["accuracy"], 3), " +- ", round(average_train_df_std["accuracy"], 3),
											" ; EOdds = ", round(average_train_df_mean["EOdds"], 3), " +- ", round(average_train_df_std["EOdds"], 3))

		print("Validation -> accuracy = ", round(average_val_df_mean["accuracy"], 3), " +- ", round(average_val_df_std["accuracy"], 3),
											" ; EOdds = ", round(average_val_df_mean["EOdds"], 3), " +- ", round(average_val_df_std["EOdds"], 3))

		print("Test -> accuracy = ", round(average_test_df_mean["accuracy"], 3), " +- ", round(average_test_df_std["accuracy"], 3),
											" ; EOdds = ", round(average_test_df_mean["EOdds"], 3), " +- ", round(average_test_df_std["EOdds"], 3))


'''
This function updates the argparse arguments to the corresponding optimal values.
Can be commented to run the model on different values
'''
def update_optimal_hyperparams(args):
		from hyperparams import HYP as desired_hp
		hp_ds = getattr(desired_hp(), args.ds_name) # hyperparameter class corresponding to the dataset inside each model class
		for attribute, value in hp_ds.__dict__.items():
				setattr(args, attribute, value)
		return args