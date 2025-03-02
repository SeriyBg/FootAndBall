import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvRNNCell(nn.Module):
    """ A simple recurrent convolutional unit without gates (simpler than LSTM/GRU). """

    def __init__(self, input_dim, hidden_dim, kernel_size=3, dropout_prob=0.3):
        super(ConvRNNCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.padding = kernel_size // 2  # Maintain spatial dimensions

        # Convolution to combine input and hidden state
        self.conv = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size=kernel_size, padding=self.padding)

        # # Normalizes across channels
        # self.norm = nn.GroupNorm(num_groups=hidden_dim // 4, num_channels=hidden_dim)  # 4 channels per group
        #
        # # Prevent overfitting with Dropout
        # self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x, h_prev):
        batch_size, _, H, W = x.shape  # Get current batch size

        if h_prev is None or h_prev.shape[0] != batch_size:
            h_prev = torch.zeros(batch_size, self.hidden_dim, H, W, device=x.device)  # Match batch size

        # Ensure spatial dimensions match
        if h_prev.shape[2] != H or h_prev.shape[3] != W:
            h_prev = F.interpolate(h_prev, size=(H, W), mode="bilinear", align_corners=False)

        # Concatenate along channels (dim=1), batch size is now guaranteed to match
        combined = torch.cat([x, h_prev], dim=1)
        h_next = F.tanh(self.conv(combined))  # Apply convolution and activation
        # h_next = self.norm(h_next)
        # h_next = self.dropout(h_next)

        return h_next


class ClassifierRNN(nn.Module):
    """ Ball classifier with frozen CNN and trainable ConvRNN """

    def __init__(self, ball_classifier, hidden_dim, output_dim=2, kernel_size=3):
        super(ClassifierRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.ball_classifier = ball_classifier  # Frozen CNN
        self.conv_rnn = ConvRNNCell(output_dim, hidden_dim, kernel_size)
        self.classifier = nn.Conv2d(hidden_dim, output_dim, kernel_size=3, padding=1)

    def forward(self, x, h_prev=None):
        """ Forward pass with frozen CNN and trainable RNN """
        with torch.no_grad():
            x = self.ball_classifier(x)  # Pass through frozen CNN

        batch_size, _, H, W = x.shape
        outputs = []

        for t in range(batch_size):
            x_t = x[t:t + 1, :, :, :]  # Process one frame at a time
            h_prev = self.conv_rnn(x_t, h_prev)  # Update hidden state
            outputs.append(h_prev)

        out = self.classifier(torch.cat(outputs, dim=0))  # Process all frames at once
        return out, h_prev


class CombinedClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2, kernel_size=3):
        super(CombinedClassifier, self).__init__()

        # CNN feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        )

        # RNN for temporal processing
        self.rnn = ClassifierRNN(hidden_dim, hidden_dim, output_dim, kernel_size)

    def forward(self, x, h_prev=None):
        x = self.feature_extractor(x)  # CNN extracts spatial features first
        out, h_next = self.rnn(x, h_prev)  # RNN processes refined features

        return out, h_next  # Return classification output and updated hidden state


