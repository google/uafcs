# Copyright 2022 Google LLC

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

""" The file with the main functions corresponding to running experiments under various settings. """

import numpy as np
import torch
import util
import math
import torch.nn.functional as F
import os
import pandas as pd


def perform_update(args, optimizer, scheduler, model, x, s, y):
		optimizer.zero_grad()
		outs = model(x)
		loss = model.compute_loss(args, outs, s, y)
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
		optimizer.step()
		scheduler.step()


def inference_(args, model, _loader):
		model.eval()
		all_y_pred, all_y, all_s, all_pred_probs = [], [], [], []

		with torch.no_grad():
				for _iter, _data in enumerate(_loader):
						x, s, y = _data
						x = x.to(args.device)
						outs = model(x)

						y_pred = torch.argmax(outs[0], dim=1)
						all_y_pred.append(y_pred)
						all_y.append(y)
						all_s.append(s)
						all_pred_probs.append( F.softmax(outs[0], dim=1)[:, 1] )
						
		all_y_pred = torch.cat(all_y_pred, dim=0).view(-1)
		all_y = torch.cat(all_y, dim=0).view(-1)
		all_s = torch.cat(all_s, dim=0).view(-1)
		all_pred_probs = torch.cat(all_pred_probs, dim=0).view(-1)

		metrics = util.compute_metrics(all_y_pred.to("cpu").numpy(), all_y.numpy(), all_s.numpy(), all_pred_probs.to("cpu").numpy())
		return metrics


def perform_run(args, runid):
		print("Current run ", runid)

		dataset = util.load_dataset(args, runid)
		dataloader = util.generate_batched_loader(args, dataset)  # generating pytorch format dataloader

		args.in_dim = dataset["train"][0].shape[1] # input dimensions
		args.num_labels = dataset["num_labels"]
		args.num_groups = dataset["num_groups"]

		class_counts = [np.sum(dataset["train"][2] == 0 ), np.sum(dataset["train"][2] == 1) ]
		model = util.fetch_model(args, args.num_groups, dataset, class_counts)
		model.to(args.device)
		print("model initialized")

		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.w_decay, amsgrad=False)
		T_max = args.epochs * math.ceil( dataset["train"][0].shape[0] / args.b_size ) # for LR scheduler
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=- 1, verbose=False) # no restarts here
		print("optimizer and scheduler initialized")

		args.path_model_save = "saved_models/{0}/{1}/".format(args.model_type, args.ds_name)
		if not os.path.exists(args.path_model_save):
				os.makedirs(args.path_model_save)

		if args.do_train:
				print("\nTraining model ...")
				min_val_acc = 0.0 	# used for saving best validation model
				for epoch in range(1, args.num_pretrain_epochs+1):
						model.train()
						for _iter, _data in enumerate(dataloader['train']):
								x, s, y = _data
								x, s, y = x.to(args.device), s.to(args.device), y.to(args.device) # x -> input ; s -> sensitive attribute ; y -> classification label
								perform_update(args, optimizer, scheduler, model, x, s, y)
						print("Finished epoch ", epoch)

						current_val_metrics = inference_(args, model, dataloader["val"])
						if current_val_metrics["accuracy"] > min_val_acc:
								torch.save(model.state_dict(), args.path_model_save + str(runid) + ".pth")
								print("Saved model at epoch ", epoch, " with validation accuracy = ", current_val_metrics["accuracy"])
								min_val_acc = current_val_metrics["accuracy"]

				model.initialize_entropy_weights(args, optimizer.param_groups[0]['lr'])	# initialize network F_w to predict the weights
				for epoch in range(args.num_pretrain_epochs+1, args.epochs+1):
						model.train()
						for _iter, _data in enumerate(dataloader['tta']):
								x, s, y = _data
								x, s, y = x.to(args.device), s.to(args.device), y.to(args.device) # x -> input ; s -> sensitive attribute ; y -> classification label

								# return the loss values for the corresponding constraints
								constraint_1, constraint_2 = model.step_entropy_weights(args, x)  # the maximization w.r.t the weights

								optimizer.zero_grad()
								outs = model(x)

								loss_1 = model.compute_loss(args, outs, s, y)
								loss_2 = model.auxiliary_losses(args)
								loss = loss_1 + loss_2
								loss.backward()

								torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
								optimizer.step()
								scheduler.step()
						print("Finished epoch ", epoch)

				torch.save(model.state_dict(), args.path_model_save + str(runid) + ".pth")
				
		model.load_state_dict(torch.load( args.path_model_save + str(runid) + ".pth" ))
		print("Model loaded for inference")

		metrics_train = inference_(args, model, dataloader["train"])
		print("\nTraining set Performance at current runid ", runid, " is = ", metrics_train)

		print("\nValidation Run ...")
		metrics_val = inference_(args, model, dataloader["val"])
		print("Validation Performance at current runid ", runid, " is = ", metrics_val)

		metrics_test = inference_(args, model, dataloader["test"])
		print("Final Test Performance at current runid ", runid, " is = ", metrics_test)
		print("\n\n\n")

		return metrics_train, metrics_val, metrics_test


if __name__ == "__main__":
		from arguments import parse_arguments
		args = parse_arguments()

		args = util.update_optimal_hyperparams(args)	# update the hyperparameters to optimal values

		all_train_metrics, all_val_metrics, all_test_metrics = [], [], []

		for run in range(1, args.runs+1):
				train_metrics, valid_metrics, test_metrics = perform_run(args, run)
				all_train_metrics.append(train_metrics)
				all_val_metrics.append(valid_metrics)
				all_test_metrics.append(test_metrics)
	
		# Details for corresponding log file for this entire run
		dir_path = "./result_logs/{0}/{1}/".format(args.model_type, args.ds_name)
		if not os.path.exists(dir_path):
				os.makedirs(dir_path)

		from time import strftime
		from datetime import datetime
		from pathlib import Path
		file_path = dir_path + "_log_{0}.csv".format( datetime.now().strftime('%Y-%m-%d %H:%M:%S') )	# appending the log_filename with local date and time to store all runs
		my_file = Path(file_path)
		print("Log file for the experiment - ", my_file)

		for run in range(1, args.runs+1):
				util.save_results_as_csv("train", all_train_metrics[run-1], run, my_file, file_path)

		for run in range(1, args.runs+1):
				util.save_results_as_csv("validation", all_val_metrics[run-1], run, my_file, file_path)

		for run in range(1, args.runs+1):
				util.save_results_as_csv("test", all_test_metrics[run-1], run, my_file, file_path)

		results_df = pd.read_csv(file_path, sep='\t')
		print("Arguments -> ", args, "\n")
		# print(results_df, "\n\n")

		util.average_results(results_df)
