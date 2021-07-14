## GTSRB Deep Learning Solution

#### Requirements
1. Install PyTorch from http://pytorch.org

2. Run the following command to install additional dependencies

```bash
pip install -r requirements.txt
```

#### Training and validating your model
Run the script `main.py` to train your model.

#### Evaluating your model on the test set

During training, model checkpoints are saved as `model_x.pth`
You can run on of the checkpoints:

```
python evaluate.py --data [data_dir] --model [model_file]
```

