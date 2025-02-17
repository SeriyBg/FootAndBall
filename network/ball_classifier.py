import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvRNNCell(nn.Module):
    """ A simple recurrent convolutional unit without gates (simpler than LSTM/GRU). """
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super(ConvRNNCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.padding = kernel_size // 2  # Maintain spatial dimensions

        # Convolution to combine input and hidden state
        self.conv = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size=kernel_size, padding=self.padding)

    def forward(self, x, h_prev):
        if h_prev is None:
            h_prev = torch.zeros_like(x)  # Initialize hidden state as zero (same size as input)

        combined = torch.cat([x, h_prev], dim=1)  # Concatenate along channel axis
        h_next = torch.tanh(self.conv(combined))  # Apply convolution and activation

        return h_next  # No memory cell, just the updated hidden state


class BallClassifierRNN(nn.Module):
    """ Ball classifier with simple ConvRNN for temporal consistency. """
    def __init__(self, input_dim, hidden_dim, output_dim=2, kernel_size=3):
        super(BallClassifierRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv_rnn = ConvRNNCell(input_dim, hidden_dim, kernel_size)

        self.classifier = nn.Conv2d(hidden_dim, output_dim, kernel_size=3, padding=1)

    def forward(self, x, h_prev=None):
        h_next = self.conv_rnn(x, h_prev)  # Recurrent convolution operation
        out = self.classifier(h_next)  # Final classification

        return out, h_next  # Return updated hidden state for next frame
