import torch.nn as nn


'''
References:
https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/
'''


class FullyConnected(nn.Module):

    def __init__(self, inputs, outputs, hidden_neurons):
        super(FullyConnected, self).__init__()

        input_layer = nn.Linear(inputs, hidden_neurons[0])
        output_layer = nn.Linear(hidden_neurons[-1], outputs)

        hidden_layers = []

        for inputs, outputs in zip(hidden_neurons, hidden_neurons[1:]):
            hidden_layers.append(nn.Linear(inputs, outputs))
            hidden_layers.append(nn.ReLU())

        self.model = nn.Sequential(
                input_layer,
                nn.ReLU(),
                *hidden_layers,
                output_layer
        )

    def forward(self, x):
        return self.model(x)
