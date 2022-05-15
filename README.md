# Pytorch implementation of multiple instance learning model with and without attention

This repository implements different neural network based approches for Multiple instance learning problem presented in the following papers:
- [Revisiting Multiple Instance Neural Networks](https://arxiv.org/pdf/1610.02501v1.pdf)
- [Attention-based Deep Multiple Instance Learning](https://arxiv.org/pdf/1802.04712v4.pdf)
## Installation

In the project repository run the following:
```pip install -e .```

## Data 
This implementation uses the deafult data for Multiple instance learning. 

## Hyper-parameters
The following hyper-parameters from the source papers are adopted in the experiments

| dataset        | start learning rate   |  weight decay  |  momentum |
| :------:   | :----:  | :----:  |  :----:  |
| musk1      | 0.0005  |  0.005  |  0.9     |
| musk2      | 0.0005  |  0.03   |  0.9     |
| fox        | 0.0001  |  0.01   |  0.9     |
| tiger      | 0.0005  |  0.005  |  0.9     |
| elephat    | 0.0001  |  0.005  |  0.9     |

The number of max epoch is set to 100.


## To do:

Split data into train test when the test indices are not given!
