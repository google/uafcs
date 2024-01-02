# Unsupervised Adaptation for Fairness under Covariate Shift

This repository serves to open-source the code used in the paper: "[Improving Fairness-Accuracy tradeoff with few Test Samples under Covariate Shift](https://arxiv.org/abs/2310.07535)".

## Preliminaries
### Getting started

To avoid any conflict with your existing Python setup, it is suggested to work in a virtual environment with [`virtualenv`](https://docs.python-guide.org/dev/virtualenvs/). To install `virtualenv`:
```bash
pip install --upgrade virtualenv
```
Create a virtual environment, activate it and install the requirements in [`requirements.txt`](requirements.txt).
```bash
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```
## Running the code
### Generating Data Splits

```
python generate_data_splits.py --ds_name {0} --gamma {1}
```
Here<br />
{0}: is the dataset name - "arrhythmia", "adult", "communities", "drug" <br />
{2}: gamma - the shift factor for the test set <br />

Similarly, the following command can be used to generate the asymmetric splits
```
python generate_asymm_splits.py --ds_name {0}
```
"gamma" values for the groups can be adjusted in "create_asymm_covar_shift" function

<br />

### Running the main file
To run the model
```
python main.py --model_type {0} --ds_name {1} --do_train --k_shot {2}
```
Here<br />
{0}: is the model name - can be assigned to anything of interest, eg - "wass_and_entropy_model" <br />
{1}: dataset name <br />
{2}: the number of unlabelled test samples utilized <br />
The remaining arguments can be found in "arguments.py" file with explanations

## Disclaimer

**This is not an officially supported Google product.**
