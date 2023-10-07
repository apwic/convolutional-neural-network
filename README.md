# Convolutional Neural Network

Convolutional Neural Network from scratch using Python. Implemented forward propagation, backward propagation, and 10-Fold Cross Validation.
Models used can be changed in `helper.py` or can be loaded with the format as in `/output`.

## Requirement

1. Numpy
2. Matplotlib

```zsh
pip install numpy matplotlib
```

## How to use

```zsh
python main.py run [-h] [-e EPOCHS] [-b BATCH_SIZE] [-n NUM_SAMPLES] [-lr LEARNING_RATE] [-s SAVE] [-l LOAD]
```

Details on the args:

```zsh
usage: main.py run [-h] [-e EPOCHS] [-b BATCH_SIZE] [-n NUM_SAMPLES] [-lr LEARNING_RATE] [-s SAVE] [-l LOAD]

options:
  -h, --help            show this help message and exit

Training Arguments:
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs (default: 3)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size (default: 5)
  -n NUM_SAMPLES, --num_samples NUM_SAMPLES
                        Length of split dataset (default: 10)
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Length of split dataset (default: 0.01)

Model Management:
  -s SAVE, --save SAVE  Save the model after training (default: None)
  -l LOAD, --load LOAD  Name of the file to load (including .json) (default: None)
```
