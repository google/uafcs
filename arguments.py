# Copyright 2022 Google LLC

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

""" The file containing list of arguments and their details like data-type and default values. """

import argparse


def parse_arguments():
		parser = argparse.ArgumentParser()
		parser.add_argument("--model_type", type=str, required=True)
		parser.add_argument("--device", type=str, default='cuda:0')

		parser.add_argument("--ds_name", type=str, required=True, help='Dataset name : ["compas","adult","lawschool","communities"]')
		parser.add_argument("--runs", type=int, default=50, help="number of runs with random train-val-test splits")
		parser.add_argument('--do_train', action='store_true', help='whether to train the model or directly evaluate it')

		# model training
		parser.add_argument("--epochs", type=int, default=50)
		parser.add_argument("--optimizer", type=str, default='adam')
		parser.add_argument("--lr", type=float, default=1e-3)
		parser.add_argument("--w_decay", type=float, default=1e-5)
		parser.add_argument("--b_size", type=int, default=32)
		parser.add_argument("--grad_clip", type=float, default=5.0)

		# MLP dimensions
		parser.add_argument("--backbone_dim1", type=int, default=100)
		parser.add_argument("--backbone_dim2", type=int, default=64)
		parser.add_argument("--classifier_dim", type=int, default=32)
		parser.add_argument("--adversary_dim", type=int, default=32)

		# Parameters for test data adaptation
		parser.add_argument("--k_shot", type=int, default=0, help="Number of unlabelled test samples available to learn from")
		parser.add_argument("--num_pretrain_epochs", type=int, default=15, help="number of epochs for pretraining")
		
		# Auxiliary Loss hyperparameters
		parser.add_argument("--lambda_1", type=float, default=0.005, help='lambda factor for unlabelled test samples entropy loss')
		parser.add_argument("--lambda_2", type=float, default=0.005, help='lambda factor for wass fairness loss')
	
		parser.add_argument("--c1", type = float, default = 10.0, help = "Regularizer for Weight predictor constraints on unlabelled test samples")
		parser.add_argument("--c2", type = float, default = 10.0, help = "Regularizer for Weight predictor constraints on training samples")
		parser.add_argument("--tta_bs", type=int, default=128, help="Batch size for training samples during test adaptation")

		args = parser.parse_args()

		return args
