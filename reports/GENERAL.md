# AJ-RNN

## What I have done

- migarted AJ-RNN to tensorflow 2.0
- diveded the model in layers
- split train/validation/test to avoid autocorrelation
- some experiments on loss & learning rate
- tried LSTM and GRU cells
- function to save/restore the model

## TODO

- improve accuracy

# Random Forests

## What I have done

### Dataset

Ive adjusted the dataset as follows:

```python
{'time(0)::band(0)': array([0.02333209, 0.02921154, 0.05180097, ..., 0.01856665, 0.01677188,
        0.01881421], dtype=float32),
 'time(0)::band(1)': array([0.03176535, 0.03681616, 0.06535629, ..., 0.02866184, 0.02890525,
        0.03261729], dtype=float32),
 'time(0)::band(2)': array([0.0506984 , 0.05916968, 0.09253751, ..., 0.02877651, 0.02871185,
        0.02968184], dtype=float32),
...
```

In this way I have `time(t)::band(b)` as a feature.

## TODO

- batch the dataset
- remove `check_dataset=False`
- try to improve accuracy

# Temp CNN

## What I have done

- learned about the
- adapted the dataset to the new format
- tested some architectures

## TODO

- improve accuracy

# Reports

## TODO

- show the train/validation/test splits in a map
- show the classes in a map
- confusion matrix
