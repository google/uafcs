# Copyright 2022 Google LLC

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

""" The file containing the code to create asymmetric covariate shift for various datasets. """

import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from sklearn.decomposition import PCA
from util import remove_nans
import math



def save_split(args):
		dataset = load_other_datasets(args)

		dataset["num_labels"] = np.unique( dataset["train"][2] ).shape[0]
		dataset["num_groups"] = np.unique( dataset["train"][1] ).shape[0]

		print("Train set size = ", dataset["train"][0].shape[0], "; Val set size = ", dataset["val"][0].shape[0],
												"; Test set size = ", dataset["test"][0].shape[0])

		with open('datasets/{0}/split.pickle'.format(args.ds_name), 'wb') as handle:
				pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_asymm_covar_shift(data, sens_data_X, args):

		all_train_idx, all_val_idx, all_test_idx = [], [], []
		for num_ in range(2):   # iterating over the groups
				_idx = np.where( sens_data_X == num_ )[0]
				_a = data[_idx]
				pca = PCA(n_components=2)
				pc2 = pca.fit_transform(_a)
				pc = pc2[:, 0]
				pca_dir = pca.components_[0]

				direction = pca_dir / np.linalg.norm(pca_dir)
				a = _a @ direction.reshape(data.shape[1], 1)
				b = np.percentile(a, 60)

				# gamma determing the magnitude of shift in a given attribute. Sample values provided below, can be adjusted
				if num_ == 0:
					gamma = 10.0
				else:
					gamma = 1.0

				a = a.reshape(-1)
				probs = np.exp( gamma * (a - b) )
				probs = probs / np.sum(probs)

				_test_indices = np.random.choice( range(a.shape[0]), size=math.ceil(a.shape[0] * 0.4), replace=False, p=probs )

				_train_indices = np.setdiff1d( np.arange(a.shape[0]), _test_indices)
				_val_indices = np.random.choice( _train_indices, size=int(0.16 * _train_indices.shape[0]), replace=False )

				all_train_idx.extend( list(_idx[_train_indices.astype(int)]) )
				all_val_idx.extend(list(_idx[_val_indices.astype(int)] ))
				all_test_idx.extend(list(_idx[_test_indices.astype(int)]))

		return np.array(all_train_idx), np.array(all_val_idx), np.array(all_test_idx)



def normalize(X, input_type, df_X_train=None, mu=None, s=None):
		"""
		input_type: whether the input is train, val or test
		mu, s: None when input_type is train, use the ones computed from X_train otherwise
		"""
		raw_X = X.copy(deep=True)
		if input_type == "train":
				train_mu, train_sigma, mu_array, sigma_array = {}, {}, [], []
				for c in list(X.columns):
						if X[c].min() < 0 or X[c].max() > 1:
								mu = X[c].mean()
								s = X[c].std(ddof=0)

								if s < 0.1:	# adjust if sigma is too small
										s = 1.0

								train_mu[c] = mu
								mu_array.append(mu)

								train_sigma[c] = s
								sigma_array.append(s)

								X.loc[:, c] = (X[c] - mu) / s
						else:
								mu_array.append(0)
								sigma_array.append(1)

				return X, train_mu, train_sigma, raw_X, mu_array, sigma_array

		elif input_type == "all_X":
				for c in list(X.columns):
						if X[c].min() < 0 or X[c].max() > 1:
								all_mu = X[c].mean()
								all_s = X[c].std(ddof=0)
								X.loc[:, c] = (X[c] - all_mu) / all_s
				return X, raw_X

		else:
				for c in list(X.columns):
						if df_X_train[c].min() < 0 or df_X_train[c].max() > 1:
								X.loc[:, c] = (X[c] - mu[c]) / s[c]
				return X, raw_X



def load_other_datasets(args):
		data_X = pd.read_csv('./datasets/{0}/{0}_X.csv'.format(args.ds_name))
		data_y = pd.read_csv('./datasets/{0}/{0}_y.csv'.format(args.ds_name))

		normalized_data_X, data_X = normalize(data_X, "all_X")
		normalized_data_X = remove_nans(normalized_data_X)

		# Sensitive attribute array for asymmetric splitting
		if args.ds_name == "adult":
			sens_data_X = normalized_data_X['sex']
			sens_data_X[sens_data_X < 0] = 0
			sens_data_X[sens_data_X > 0] = 1
		elif args.ds_name == "drug":
			sens_data_X = normalized_data_X['Race']
		elif args.ds_name == "communities":
			sens_data_X = normalized_data_X['majority_white']
		elif args.ds_name == "arrhythmia":
			sens_data_X = normalized_data_X['Gender']

		train_indices, val_indices, test_indices = create_asymm_covar_shift(normalized_data_X.values, sens_data_X, args)

		X_train, y_train = data_X.iloc[train_indices, :], data_y.iloc[train_indices]
		X_val, y_val = data_X.iloc[val_indices, :], data_y.iloc[val_indices]
		X_test, y_test = data_X.iloc[test_indices, :], data_y.iloc[test_indices]

		if args.ds_name == "adult":
				y_test = y_test.reset_index(drop=True)
				y_test = y_test['income']

				sensitive_features_train = X_train.pop('sex') # X_train['sex']
				sensitive_features_val = X_val.pop('sex') # X_val['sex']
				sensitive_features_test = X_test.pop('sex') # X_test['sex']

		elif args.ds_name == "communities":
				y_test = y_test.reset_index(drop=True)
				y_test = y_test['ViolentCrimesPerPop']

				sensitive_features_train = X_train.pop('majority_white') # X_train['majority_white']
				sensitive_features_val = X_val.pop('majority_white') # X_val['majority_white']
				sensitive_features_test = X_test.pop('majority_white') # X_test['majority_white']

		elif args.ds_name == "drug":
				y_test = y_test.reset_index(drop=True)
				y_test = y_test['Label']

				sensitive_features_train = X_train.pop('Race') # X_train['Race']
				sensitive_features_val = X_val.pop('Race') # X_val['Race']
				sensitive_features_test = X_test.pop('Race') # X_test['Race']

		elif args.ds_name == "arrhythmia":
				y_test = y_test.reset_index(drop=True)
				y_test = y_test['Label']

				sensitive_features_train = X_train.pop('Gender') # X_train['Race']
				sensitive_features_val = X_val.pop('Gender') # X_val['Race']
				sensitive_features_test = X_test.pop('Gender') # X_test['Race']

		# Z-score normalization
		X_train, train_mu, train_sigma, raw_X_train, mu_array, sigma_array = normalize(X_train.copy(deep=True), "train")
		X_val, raw_X_val = normalize(X_val.copy(deep=True), "val", df_X_train=raw_X_train, mu=train_mu, s=train_sigma)
		X_test, raw_X_test = normalize(X_test.copy(deep=True), "test", df_X_train=raw_X_train, mu=train_mu, s=train_sigma)

		X_train, X_val, X_test, y_train, y_val, y_test, sensitive_features_train, sensitive_features_val, sensitive_features_test = X_train.to_numpy(), \
														X_val.to_numpy(), X_test.to_numpy(), y_train.to_numpy().reshape(-1), y_val.to_numpy().reshape(-1), y_test.to_numpy().reshape(-1), \
														sensitive_features_train.to_numpy(), sensitive_features_val.to_numpy(), sensitive_features_test.to_numpy()

		raw_X_train, raw_X_val, raw_X_test = raw_X_train.to_numpy(), raw_X_val.to_numpy(), raw_X_test.to_numpy()

		mu_array, sigma_array = np.array(mu_array), np.array(sigma_array)

		sensitive_features_train[sensitive_features_train < 0] = 0
		sensitive_features_train[sensitive_features_train > 0] = 1
		
		sensitive_features_val[sensitive_features_val < 0] = 0
		sensitive_features_val[sensitive_features_val > 0] = 1
		
		sensitive_features_test[sensitive_features_test < 0] = 0
		sensitive_features_test[sensitive_features_test > 0] = 1
		
		perform_size_asserts(X_train, sensitive_features_train, y_train)
		perform_size_asserts(X_val, sensitive_features_val, y_val)
		perform_size_asserts(X_test, sensitive_features_test, y_test)

		return {"train": [X_train, sensitive_features_train, y_train, raw_X_train], "val": [X_val, sensitive_features_val, y_val, raw_X_val],
						"test": [X_test, sensitive_features_test, y_test, raw_X_test], "train_mu": mu_array, "train_sigma": sigma_array}


def perform_size_asserts(a, b, c):
		assert a.shape[0] == b.shape[0] == c.shape[0], "error in numpy shapes"


if __name__ == "__main__":
		parser = argparse.ArgumentParser()
		parser.add_argument(
				"--ds_name",
				type=str,
				required=True,
		)

		args = parser.parse_args()
		save_split(args)