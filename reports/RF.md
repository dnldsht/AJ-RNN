# Random Forests
## Dataset
the dataset has been mapped with this function
```python
def map_data(d):
  return {
    f"time({t})::band({b})": d[:,t,b]
      for t in range(d.shape[-2])
        for b in range(d.shape[-1])
  }
```
The new data look like this
```
{'time(0)::band(0)': array([0.02333209, 0.02921154, 0.05180097, ..., 0.01856665, 0.01677188,
        0.01881421], dtype=float32),
 'time(0)::band(1)': array([0.03176535, 0.03681616, 0.06535629, ..., 0.02866184, 0.02890525,
        0.03261729], dtype=float32),
 'time(0)::band(2)': array([0.0506984 , 0.05916968, 0.09253751, ..., 0.02877651, 0.02871185,
        0.02968184], dtype=float32),
....
```

Training random forest on 83699 example(s) and 864 feature(s).
Final OOB metrics: accuracy:0.990538 logloss:0.0512743
300 root(s), 551972 node(s), and 744 input feature(s).

History training
'val_loss': [0.0],
'val_accuracy': [0.9291],

Test Set:
1/1 - 4s - loss: 0.0000e+00 - accuracy: 0.9171 - 4s/epoch - 4s/step
{'loss': 0.0, 'accuracy': 0.9171094298362732}


# Notes
- The dataset han not been batched
- check_dataset=False has been introduced


