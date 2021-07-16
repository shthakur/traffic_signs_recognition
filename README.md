## GTSRB Deep Learning Solution

#### Requirements
1. Install PyTorch from http://pytorch.org

2. Run the following command to install additional dependencies

```bash
pip install -r requirements.txt
```
#### Preprocessing

Run `preprocessing.py` to first process the data to remove imbalance and add extra data augmentations

#### Training and validating your model
Run the script `main.py` to train your model.

#### Evaluating your model on the test set

During training, model checkpoints are saved as `model_x.pth`
You can run on of the checkpoints:

```
python evaluate.py --data [data_dir] --model [model_file]
```

This generates a CSV files which can be submitted to Kaggle. The best model gets 99.4 on public leaderboard.

Following parameters can be used to achieve that result:

```
params = Namespace()
params.lr = 0.0001
params.batch_size = 64
params.seed = 7
params.cnn = '100, 150, 250, 350'
params.locnet = '200,300,200'
params.locnet2 = None
params.locnet3 = '150,150,150'
params.st = True
params.resume = False
params.epochs = 15
params.patience = 10
params.dropout = 0.5
```
