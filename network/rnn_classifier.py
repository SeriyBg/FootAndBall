import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3, dropout_prob=0.3):
        super(Conv2dGRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

        # Input-to-hidden and hidden-to-hidden convolution layers (3 gates: r, z, n)
        self.x2h = nn.Conv2d(input_size, hidden_size * 3, kernel_size=self.kernel_size, padding=self.padding)
        self.h2h = nn.Conv2d(hidden_size, hidden_size * 3, kernel_size=self.kernel_size, padding=self.padding)

        # Layer normalization for stability
        self.norm = nn.GroupNorm(num_groups=hidden_size // 4, num_channels=hidden_size * 3)

        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(p=dropout_prob)

        # Weight initialization
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.x2h.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.h2h.weight, nonlinearity='linear')
        if self.x2h.bias is not None:
            nn.init.zeros_(self.x2h.bias)
        if self.h2h.bias is not None:
            nn.init.zeros_(self.h2h.bias)

    def forward(self, x, hx=None):
        batch_size, _, H, W = x.shape  # Get spatial size

        if hx is None:
            hx = torch.zeros(batch_size, self.hidden_size, H, W, device=x.device)

        # Compute gate activations
        gates = self.x2h(x) + self.h2h(hx)
        gates = self.norm(gates)  # Apply normalization

        # Split into three gate tensors
        r_t, z_t, n_t = gates.chunk(3, dim=1)

        # Apply activation functions
        r_t = torch.sigmoid(r_t)  # Reset gate
        z_t = torch.sigmoid(z_t)  # Update gate
        n_t = torch.tanh(n_t + r_t * self.h2h(hx))  # Candidate activation

        # Compute new hidden state
        h_next = (1 - z_t) * n_t + z_t * hx
        h_next = self.dropout(h_next)  # Apply dropout

        return h_next


class Conv2dLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3, dropout_prob=0.3):
        super(Conv2dLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

        # Input-to-hidden and hidden-to-hidden convolution layers (4 gates: i, f, g, o)
        self.x2h = nn.Conv2d(input_size, hidden_size * 4, kernel_size=self.kernel_size, padding=self.padding)
        self.h2h = nn.Conv2d(hidden_size, hidden_size * 4, kernel_size=self.kernel_size, padding=self.padding)

        # Layer normalization for stability
        self.norm = nn.GroupNorm(num_groups=hidden_size // 4, num_channels=hidden_size * 4)

        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(p=dropout_prob)

        # Weight initialization
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.x2h.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.h2h.weight, nonlinearity='linear')
        if self.x2h.bias is not None:
            nn.init.zeros_(self.x2h.bias)
        if self.h2h.bias is not None:
            nn.init.zeros_(self.h2h.bias)

    def forward(self, x, hx=None):
        batch_size, _, H, W = x.shape  # Get spatial size

        if hx is None:
            hx = (torch.zeros(batch_size, self.hidden_size, H, W, device=x.device),
                  torch.zeros(batch_size, self.hidden_size, H, W, device=x.device))

        h_prev, c_prev = hx

        # Compute gates
        gates = self.x2h(x) + self.h2h(h_prev)
        gates = self.norm(gates)  # Apply normalization

        # Split into four gate tensors
        i_t, f_t, g_t, o_t = gates.chunk(4, dim=1)

        # Apply activation functions
        i_t = torch.sigmoid(i_t)  # Input gate
        f_t = torch.sigmoid(f_t)  # Forget gate
        g_t = torch.tanh(g_t)  # Cell candidate
        o_t = torch.sigmoid(o_t)  # Output gate

        # Compute new cell state
        c_next = f_t * c_prev + i_t * g_t
        c_next = self.dropout(c_next)  # Apply dropout

        # Compute new hidden state
        h_next = o_t * torch.tanh(c_next)

        return h_next, c_next


class Conv2dRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, dropout_prob=0.3):
        super(Conv2dRNNCell, self).__init__()

        self.hidden_size = hidden_size

        self.x2h = nn.Conv2d(input_size, hidden_size, kernel_size, padding=kernel_size // 2)
        self.h2h = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2)

        self.norm = nn.GroupNorm(num_groups=hidden_size // 4, num_channels=hidden_size)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x, h_prev=None):
        if h_prev is None:
            h_prev = torch.zeros_like(x[:, :self.hidden_size, :, :])  # Initialize hidden state

        print("X shape is {} and h_prev shape is {}".format(x.shape, h_prev.shape))
        h_next = self.x2h(x) + self.h2h(h_prev)
        h_next = self.norm(h_next)
        h_next = self.dropout(h_next)

        h_next = torch.tanh(h_next)

        return h_next


class ConvRNNCell(nn.Module):
    """ A simple recurrent convolutional unit without gates (simpler than LSTM/GRU). """

    def __init__(self, input_dim, hidden_dim, kernel_size=3, dropout_prob=0.3):
        super(ConvRNNCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.padding = kernel_size // 2  # Maintain spatial dimensions

        # Convolution to combine input and hidden state
        self.conv = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size=kernel_size, padding=self.padding)

        # Normalizes across channels
        self.norm = nn.GroupNorm(num_groups=hidden_dim // 4, num_channels=hidden_dim)  # 4 channels per group

        # Prevent overfitting with Dropout
        self.dropout = nn.Dropout(p=dropout_prob)

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
        h_next = self.norm(h_next)
        h_next = self.dropout(h_next)

        return h_next


class ClassifierRNN(nn.Module):
    """ Ball classifier with frozen CNN and trainable ConvRNN """

    def __init__(self, ball_classifier, hidden_dim, output_dim=2, kernel_size=3, type="rnn"):
        super(ClassifierRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.ball_classifier = ball_classifier  # Frozen CNN
        print("Output dim is {} and hidden dim is {}".format(output_dim, hidden_dim))
        if type == "rnn":
            self.conv_rnn = ConvRNNCell(output_dim, hidden_dim, kernel_size)
        elif type == "lstm":
            self.conv_rnn = Conv2dLSTMCell(output_dim, hidden_dim, kernel_size)
        elif type == "gru":
            self.conv_rnn = Conv2dGRUCell(output_dim, hidden_dim, kernel_size)
        else:
            raise ValueError("Invalid RNN type: {}".format(type))
        self.conv_rnn = Conv2dRNNCell(output_dim, hidden_dim, kernel_size)
        # self.classifier = nn.Conv2d(hidden_dim, output_dim, kernel_size=3, padding=1)
        self.classifier = nn.Sequential(
            nn.Conv2d(hidden_dim, output_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)
        )

    # def forward(self, x, h_prev=None):
    def forward(self, x):
        """ Forward pass with frozen CNN and trainable RNN """
        with torch.no_grad():
            x = self.ball_classifier(x)  # Pass through frozen CNN

        batch_size, _, H, W = x.shape
        outputs = []

        h_prev = torch.zeros(batch_size, self.hidden_dim, H, W, device=x.device)
        for t in range(batch_size):
            x_t = x[t:t + 1, :, :, :]  # Process one frame at a time
            h_prev = self.conv_rnn(x_t, h_prev)  # Update hidden state
            outputs.append(h_prev)

        out = self.classifier(torch.cat(outputs, dim=0))  # Process all frames at once
        # return out, h_prev
        return out


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


