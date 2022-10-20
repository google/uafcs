# Copyright 2022 Google LLC

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

""" The file containing the model architecture construction utilized in the experiments. """

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ot
from torch.distributions import Categorical


# Base classes
class Backbone(nn.Module):
		def __init__(self, args):
				super(Backbone, self).__init__()

				self.a1 = nn.Linear(args.in_dim, args.backbone_dim1)
				self.drop1 = nn.Dropout(p=0.25)
				self.a2 = nn.Linear(args.backbone_dim1, args.backbone_dim2)

				nn.init.xavier_normal_(self.a1.weight)
				nn.init.xavier_normal_(self.a2.weight)

		def forward(self, inputs):
				x = F.relu( self.a1( inputs ) )
				x = self.drop1(x)
				x = F.relu( self.a2( x ) )
				return x


class Classifier(nn.Module):
		def __init__(self, args):
				super(Classifier, self).__init__()

				self.a1 = nn.Linear(args.backbone_dim2, args.classifier_dim)
				self.drop1 = nn.Dropout(p=0.25)
				self.a2 = nn.Linear(args.classifier_dim, args.num_labels)

				nn.init.xavier_normal_(self.a1.weight)
				nn.init.xavier_normal_(self.a2.weight)

		def forward(self, inputs):
				x = F.relu(self.a1(inputs))
				x = self.drop1(x)
				x = self.a2(x)
				return x


# Our Model
class WassAndEntropy(nn.Module):
		def __init__(self, args, num_groups, dataset, class_counts):
				super(WassAndEntropy, self).__init__()

				self.backbone = Backbone(args)
				self.classifier = Classifier(args)

				self.clf_loss_criterion = nn.CrossEntropyLoss()
				self.unlbl_test_samples_idxs = None
				self.unlbl_test_samples = None					 # unlabelled test samples
				self.unlbl_test_samples_group_count = None
				self.set_unlbl_test_samples(args, num_groups, dataset)

		def set_unlbl_test_samples(self, args, num_groups, dataset):
				_data = dataset["test"]
				_sens_attrs = _data[1]

				if args.k_shot > 1:
						subset_attributes = []
						group_count = {}
						idxs = [] 				# sampling an equal number of instances randomly from all groups
						for num_ in range(num_groups):
								_idx = np.where( _sens_attrs == num_ )[0]
								_count = min(_idx.shape[0], args.k_shot//num_groups)
								group_count[num_] = _count
								subset_attributes.extend( [num_] * _count )
								_idx = np.random.choice(_idx, size=_count, replace=False, p=None)
								idxs.append(_idx)
						
						idxs = np.concatenate( tuple(idxs) )
						subset = _data[0][idxs]
						self.unlbl_test_samples_idxs = nn.Parameter( torch.LongTensor(idxs).to(args.device), requires_grad=False )
						self.unlbl_test_samples_group_count = group_count
						self.unlbl_test_samples = nn.Parameter( torch.FloatTensor(subset).to(args.device), requires_grad=False )

		def forward(self, inputs):
				x = self.backbone(inputs)
				clf_outs = self.classifier(x)
				return (clf_outs, x)

		def compute_loss(self, args, outs, s, y):
				clf_loss = self.clf_loss_criterion(outs[0], y)
				return clf_loss

		def auxiliary_losses(self, args):
				repers = self.forward(self.unlbl_test_samples)
				wass_loss = self.obtain_wass_loss(args, repers[1])
				entropy_loss = self.obtain_entrpy_loss(args, repers[0])
				return (args.lambda_1 * entropy_loss) + (args.lambda_2 * wass_loss)

		def obtain_wass_loss(self, args, repers):
				g_0 = repers[ : self.unlbl_test_samples_group_count[0] ]
				g_1 = repers[ self.unlbl_test_samples_group_count[0] : ]

				dist_a = ( torch.ones(g_0.shape[0]) / g_0.shape[0] ).view(-1,)
				dist_b = ( torch.ones(g_1.shape[0]) / g_1.shape[0] ).view(-1,)

				M = ot.dist(g_0, g_1)
				loss = ot.emd2(dist_a.to(args.device), dist_b.to(args.device), M)
				return loss

		def obtain_entrpy_loss(self, args, test_clfs):
				a = F.softmax(test_clfs, dim=1)
				a = a + 1e-5
				a = torch.div(a, torch.sum(a, dim=1).unsqueeze(1))
				b = torch.log(a)
				_b = -(a * b)
				ent = torch.sum( _b, dim=1)
				
				weights = torch.exp(-self.current_iter_test_weights) / self.current_iter_test_weights.shape[0]
				weighted_ent = weights * ent
				return torch.sum(weighted_ent)

		def initialize_entropy_weights(self, args, current_lr):
				self.weight_net = WeightPredictor(args)
				self.weight_net.to(args.device)
				self.weightnet_opt = torch.optim.Adam(self.weight_net.parameters(), lr=current_lr, betas=(0.9, 0.999),
																	eps=1e-08, weight_decay=0.0, amsgrad=False)

		def step_entropy_weights(self, args, train_inputs):
				self.generate_current_iter_ent_weights(train_inputs=train_inputs)
				with torch.no_grad():
						repers = self.forward(self.unlbl_test_samples)

				ent = self.obtain_entrpy_loss(args, repers[0])
				constraint_1 = torch.pow( torch.sum(self.current_iter_test_weights) / self.current_iter_test_weights.shape[0] - 1.0, 2)
				constraint_2 = torch.pow( torch.sum(1 / ( self.current_iter_train_weights + 1e-3) ) / self.current_iter_train_weights.shape[0] - 1.0, 2)

				self.weightnet_opt.zero_grad()
				(-ent + (args.c1 * constraint_1) + (args.c2 * constraint_2)).backward()
				torch.nn.utils.clip_grad_norm_(self.weight_net.parameters(), args.grad_clip)
				self.weightnet_opt.step()
				
				self.generate_current_iter_ent_weights()	# generating weights again, after the max step
				return constraint_1, constraint_2


		def generate_current_iter_ent_weights(self, train_inputs=None):
				if train_inputs is not None:
						_inp = torch.cat( [self.unlbl_test_samples, train_inputs], dim=0 )
				else:
						_inp = self.unlbl_test_samples

				with torch.no_grad():
						_inp = self.backbone(_inp)
				_ws = self.weight_net( _inp )
				self.current_iter_test_weights = _ws[ : self.unlbl_test_samples.shape[0] ]
				if train_inputs is not None:
						self.current_iter_train_weights = _ws[ self.unlbl_test_samples.shape[0] : ]
						

# Network F_w to predict weights, used in the minimax optimization
class WeightPredictor(nn.Module):
		def __init__(self, args):
				super(WeightPredictor, self).__init__()
				self.a1 = nn.Linear(args.backbone_dim2, 16)
				self.drop1 = nn.Dropout(p=0.25)
				self.a2 = nn.Linear(16, 1)
				nn.init.xavier_normal_(self.a1.weight)
				nn.init.xavier_normal_(self.a2.weight)

		def forward(self, inputs):
				x = F.relu( self.a1( inputs ) )
				x = self.drop1(x)
				x = F.relu( self.a2( x ) )
				return x.view(-1)
