# Generating Data Splits

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

# Running the main file
To run the model
```
python main.py --model_type {0} --ds_name {1} --do_train --k_shot {2}
```
Here<br />
{0}: is the model name - can be assigned to anything of interest, eg - "wass_and_entropy_model" <br />
{1}: dataset name <br />
{2}: the number of unlabelled test samples utilized <br />
The remaining arguments can be found in "arguments.py" file with explanations