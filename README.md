# Nested LSTM
Keras implementation of Nested LSTMs from the paper [Nested LSTMs](https://arxiv.org/abs/1801.10308)

From the paper:
> Nested LSTMs add depth to LSTMs via nesting as opposed to stacking. The value of a memory cell
in an NLSTM is computed by an LSTM cell, which has its own inner memory cell. Nested LSTMs outperform both stacked and single-layer
LSTMs with similar numbers of parameters in our experiments on various character-level language
modeling tasks, and the inner memories of an LSTM learn longer term dependencies compared with
the higher-level units of a stacked LSTM

# Usage
Via Cells
```python
from nested_lstm import NestedLSTMCell
from keras.layers import RNN

ip = Input(shape=(nb_timesteps, input_dim))
x = RNN(NestedLSTMCell(units=64, depth=2))(ip)
...
```

Via Layer
```python
from nested_lstm import NestedLSTM

ip = Input(shape=(nb_timesteps, input_dim))
x = NestedLSTM(units=64, depth=2)(ip)
...
```

# Difference between Stacked LSTMs and Nested LSTMs (from the paper)
<img src="https://github.com/titu1994/Nested-LSTM/blob/master/images/difference.PNG?raw=true" height=100% width=100%>

# Cell diagram (depth = 2, from the paper)
<img src="https://github.com/titu1994/Nested-LSTM/blob/master/images/nested_lstm_diagram.PNG?raw=true" height=100% width=100%>

# Requirements
- Keras 2.1.3+
- Tensorflow 1.2+ or Theano. CNTK untested.
