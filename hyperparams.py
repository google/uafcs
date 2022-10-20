# Copyright 2022 Google LLC

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

""" The file containing the list of hyperparameters and their default values under various settings. """

class HYP:
		def __init__(self):
				class adult:
						def __init__(self):
								self.epochs = 50
								self.optimizer = 'adam'
								self.lr = 1e-3
								self.w_decay = 5e-4
								self.b_size = 32
								self.grad_clip = 5.0
								self.lambda_2 = 0.01
								self.lambda_1 = 1.0
								self.c1 = 1.0
								self.c2 = 50.0
								self.tta_bs = 128
								self.num_pretrain_epochs = 15

				self.adult = adult()


				class communities:
						def __init__(self):
								self.epochs = 50
								self.optimizer = 'adam'
								self.lr = 1e-3
								self.w_decay = 1e-5
								self.b_size = 32
								self.grad_clip = 5.0
								self.lambda_2 = 0.0001
								self.lambda_1 = 0.005
								self.c1 = 50.0
								self.c2 = 50.0
								self.tta_bs = 256
								self.num_pretrain_epochs = 15

				self.communities = communities()


				class drug:
						def __init__(self):
								self.epochs = 50
								self.optimizer = 'adam'
								self.lr = 1e-3
								self.w_decay = 1e-5
								self.b_size = 32
								self.grad_clip = 5.0
								self.lambda_2 = 0.1
								self.lambda_1 = 0.1
								self.c1 = 50.0
								self.c2 = 10.0
								self.tta_bs = 128
								self.num_pretrain_epochs = 15

				self.drug = drug()

				class arrhythmia:
						def __init__(self):
								self.epochs = 50
								self.optimizer = 'adam'
								self.lr = 1e-3
								self.w_decay = 1e-5
								self.b_size = 32
								self.grad_clip = 5.0
								self.lambda_2 = 0.005
								self.lambda_1 = 0.01
								self.c1 = 100.0
								self.c2 = 50.0
								self.tta_bs = 256
								self.num_pretrain_epochs = 15

				self.arrhythmia = arrhythmia()
