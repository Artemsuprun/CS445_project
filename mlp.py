# needed libraries
import torch.nn as nn


# The MLP class
class MLP(nn.Module):
    def __init__(self, input, hidden, output, act = 'relu'):
        super(MLP, self).__init__()

        # default variables
        self.loss = nn.MSELoss()

        # Validate activations
        if act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'relu':
            self.act = nn.ReLU()
        else:
            raise Exception('Invalid activation function')
        
        # Validate the parameters and set them
        if input <= 0:
            raise Exception("Input cannot be 0 or below.")
        self.input = int(input + 0.5)

        if output <= 0:
            raise Exception("Output cannot be 0 or below.")
        self.output = int(output + 0.5)

        hidden = [hidden] if isinstance(hidden, int) else hidden
        if len(hidden) <= 0:
            raise Exception("A MLP needs to have at least one hidden layer.")
        self.hidden = []
        for h in hidden:
            if h <= 0:
                raise Exception("Hidden cannot be 0 or below")
            self.hidden.append( int(h + 0.5) )
        
        # Setting up the layers with the activations
        self.layers = []
        for i in range(len(self.hidden)):
            if i == 0:
                self.layers.append( nn.Linear(in_features=self.input, out_features=self.hidden[i]) )
            else:
                self.layers.append( nn.Linear(in_features=self.hidden[i-1], out_features=self.hidden[i]) )
            self.layers.append(self.act)
        self.layers.append( nn.Linear(in_features=self.hidden[-1], out_features=self.output) )
        self.layers.append(nn.Sigmoid())
        #self.layers.append(self.act)

        # Convert the layers into a sequential layers
        self.seq_model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.seq_model(x)
    
    def display_layers(self):
        print(self.seq_model)



def predict():
    pass