# -*- coding: utf-8 -*-
import os
import sys

sys.path.append("/content/gdrive/MyDrive/ColabNotebooks")

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttention(torch.nn.Module):
    """
    Attention mechanism for the models
    """
    def __init__(self, dim=5):
        super().__init__()

        self.dim = dim

        # Using paper notation (W, V)
        self.W = nn.Linear(self.dim, self.dim)
        self.V = nn.Linear(self.dim, 1, bias=False)

    def forward(self, h):
        # Paper attenation mechanism
        #   et = V*tanh(W*ht + b)
        #   αt = softmax(et)
        #   c = sum(αt*ht)
        layer_1 = self.W(h)
        layer_1 = torch.tanh(layer_1)
        layer_2 = self.V(layer_1)
        alphas = F.softmax(layer_2, dim=1)
        c = h * alphas  # [batch, l, 2*h] x [batch, l, 1] = [batch, l, 2*h]
        output = torch.sum(
            c, 1
        )  # sum elements in dimension 1 (seq_length) [batch, 2*h]
        return output, alphas

class AdditiveAttentionBackwards(AdditiveAttention):
    """
    Attention mechanism for the models
    NOTE: Nearly same implementation as main additive attention
    but used in order to make it backwards compatible as some
    models have already been trained using this mode. Otherwise
    fails loading the model due non-matching architecture
    """

    def forward(self, h):
        output, alphas = super().forward(h)
        return output

class ModelPaper(nn.Module):
    """
    Implementation of the network architecture described
    in the paper
    Both regression and classification branches enabled
    """
    def __init__(self, l, filters, kernel, hunits):
        super().__init__()

        self.regression_enabled = True
        self.classification_enabled = True

        self.conv = nn.Sequential(
            nn.Conv1d(1, filters, kernel, padding=kernel // 2),
            nn.ReLU(),
            nn.Conv1d(filters, filters, kernel, padding=kernel // 2),
            nn.ReLU(),
            nn.Conv1d(filters, filters, kernel, padding=kernel // 2),
            nn.ReLU(),
            nn.Conv1d(filters, filters, kernel, padding=kernel // 2),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=filters,
            hidden_size=hunits,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        # input [batch, l-x(from convs), filters]
        # output [batch, l-x(de les convs), 2*hunits]

        self.attention = AdditiveAttention(dim=(2 * hunits))
        self.regression = nn.Sequential(
            nn.Linear(2 * hunits, hunits),
            nn.ReLU(),
            nn.Linear(hunits, l)
        )

        self.classification1 = nn.Sequential(
            nn.Conv1d(1, 10, 10, 1),
            nn.ReLU(),
            nn.Conv1d(10, 30, 8, 1),
            nn.ReLU(),
            nn.Conv1d(30, 40, 6, 1),
            nn.ReLU(),
            nn.Conv1d(40, 50, 5, 1),
            nn.ReLU(),
            nn.Conv1d(50, 50, 5, 1),
            nn.ReLU(),
            nn.Conv1d(50, 50, 5, 1),
            nn.ReLU(),
        )  # output --> [batch, 50, l-33]

        self.classification2 = nn.Sequential(
            nn.Flatten(start_dim=1)
        )  # flatten --> [batch, (l-33)*50]

        self.classification3 = nn.Sequential(
            nn.Linear((l - 33) * 50, 1024),
            nn.ReLU(),
            nn.Linear(1024, l),
            nn.Sigmoid()
        )

    def forward(self, x):
        reg = self.conv(x)
        reg = reg.permute(0, 2, 1)
        output_lstm, (h_n, c_n) = self.lstm(reg)
        context, alphas = self.attention(output_lstm)
        reg = self.regression(context)

        clas1 = self.classification1(x)
        clas2 = self.classification2(clas1)
        clas = self.classification3(clas2)
        y = reg * clas
        return y, reg, alphas, clas

class ModelPaperBackward(nn.Module):
    """
    Implementation of the network architecture described
    in the paper
    NOTE: Nearly same implementation as ModelPaper
    but used in order to make it backwards compatible as some
    models have already been trained using this mode. Otherwise
    fails loading the model due non-matching architecture

    Both regression and classification branches enabled
    """
    def __init__(self, l, filters, kernel, hunits):
        super().__init__()

        self.regression_enabled = True
        self.classification_enabled = True

        self.conv = nn.Sequential(
            nn.Conv1d(1, filters, kernel, padding=kernel // 2),
            nn.ReLU(),
            nn.Conv1d(filters, filters, kernel, padding=kernel // 2),
            nn.ReLU(),
            nn.Conv1d(filters, filters, kernel, padding=kernel // 2),
            nn.ReLU(),
            nn.Conv1d(filters, filters, kernel, padding=kernel // 2),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=filters,
            hidden_size=hunits,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        # input [batch, l-x(from convs), filters]
        # output [batch, l-x(from convs), 2*hunits]

        self.regression = nn.Sequential(
            AdditiveAttentionBackwards(
                dim=2 * hunits
            ),
            # input [batch, l (LSTM), 2*hunits]
            # output [batch, 2*hunits]
            nn.Linear(2 * hunits, hunits),
            nn.ReLU(),
            nn.Linear(hunits, l),
        )

        self.classification1 = nn.Sequential(
            nn.Conv1d(1, 10, 10, 1),
            nn.ReLU(),
            nn.Conv1d(10, 30, 8, 1),
            nn.ReLU(),
            nn.Conv1d(30, 40, 6, 1),
            nn.ReLU(),
            nn.Conv1d(40, 50, 5, 1),
            nn.ReLU(),
            nn.Conv1d(50, 50, 5, 1),
            nn.ReLU(),
            nn.Conv1d(50, 50, 5, 1),
            nn.ReLU(),
        )  # output [batch, 50, l-33]

        self.classification2 = nn.Sequential(
            nn.Flatten(start_dim=1)
        )  # flatten --> [batch, (l-33)*50]

        self.classification3 = nn.Sequential(
            nn.Linear((l - 33) * 50, 1024),
            nn.ReLU(),
            nn.Linear(1024, l),
            nn.Sigmoid()
        )

    def forward(self, x):
        reg = self.conv(x)
        reg = reg.permute(0, 2, 1)
        output_lstm, (h_n, c_n) = self.lstm(reg)
        reg = self.regression(output_lstm)

        clas1 = self.classification1(x)
        clas2 = self.classification2(clas1)
        clas = self.classification3(clas2)

        y = reg * clas
        alphas = torch.zeros(reg.shape)
        return y, reg, alphas, clas

class ModelOnlyRegression(nn.Module):
    """
    Implementation of the network architecture described
    in the paper but removing classification branch.
    Only regression branch is trained and used to predict
    appliance disaggregation

    Only regression branch enabled
    """
    def __init__(self, l, filters, kernel, hunits):
        super().__init__()

        self.regression_enabled = True
        self.classification_enabled = False

        self.conv = nn.Sequential(
            nn.Conv1d(1, filters, kernel, padding=kernel // 2),
            nn.ReLU(),
            nn.Conv1d(filters, filters, kernel, padding=kernel // 2),
            nn.ReLU(),
            nn.Conv1d(filters, filters, kernel, padding=kernel // 2),
            nn.ReLU(),
            nn.Conv1d(filters, filters, kernel, padding=kernel // 2),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=filters,
            hidden_size=hunits,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        # input [batch, l-x(from convs), filters]
        # output [batch, l-x(from convs), 2*hunits]

        self.attention = AdditiveAttention(dim=(2 * hunits))
        self.regression = nn.Sequential(
            nn.Linear(2 * hunits, hunits),
            nn.ReLU(),
            nn.Linear(hunits, l)
        )

    def forward(self, x):
        reg = self.conv(x)
        reg = reg.permute(0, 2, 1)
        output_lstm, (h_n, c_n) = self.lstm(reg)
        context, alphas = self.attention(output_lstm)
        reg = self.regression(context)

        y = reg
        clas = reg  # TEMPFIX to make it easy to integrate to de code (?)
        return y, reg, alphas, clas

class ModelClassAttention(nn.Module):
    """
    Implementation of the network architecture described
    in the paper but fitting classification with attention.
    Attention is used in both regression and classification

    Both regression and classification branches enabled
    """
    def __init__(self, l, filters, kernel, hunits):
        super().__init__()

        self.regression_enabled = True
        self.classification_enabled = True

        self.conv = nn.Sequential(
            nn.Conv1d(1, filters, kernel, padding=kernel // 2),
            nn.ReLU(),
            nn.Conv1d(filters, filters, kernel, padding=kernel // 2),
            nn.ReLU(),
            nn.Conv1d(filters, filters, kernel, padding=kernel // 2),
            nn.ReLU(),
            nn.Conv1d(filters, filters, kernel, padding=kernel // 2),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=filters,
            hidden_size=hunits,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        # input [batch, l-x(from convs), filters]
        # output [batch, l-x(from convs), 2*hunits]

        self.attention = AdditiveAttention(dim=(2 * hunits))
        self.regression = nn.Sequential(
            nn.Linear(2 * hunits, hunits),
            nn.ReLU(),
            nn.Linear(hunits, l)
        )

        self.classification1 = nn.Sequential(
            nn.Conv1d(1, 10, 10, 1),
            nn.ReLU(),
            nn.Conv1d(10, 30, 8, 1),
            nn.ReLU(),
            nn.Conv1d(30, 40, 6, 1),
            nn.ReLU(),
            nn.Conv1d(40, 50, 5, 1),
            nn.ReLU(),
            nn.Conv1d(50, 50, 5, 1),
            nn.ReLU(),
            nn.Conv1d(50, 50, 5, 1),
            nn.ReLU(),
        )  # output [batch, 50, l-33]

        self.classification2 = nn.Sequential(
            nn.Flatten(start_dim=1)
        )  # flatten [batch, (l-33)*50]

        self.classification3 = nn.Sequential(
            nn.Linear((l - 33) * 50 + 2 * hunits, 1024),
            nn.ReLU(),
            nn.Linear(1024, l),
            nn.Sigmoid(),
        )

    def forward(self, x):
        reg = self.conv(x)
        reg = reg.permute(0, 2, 1)
        output_lstm, (h_n, c_n) = self.lstm(reg)
        context, alphas = self.attention(output_lstm)
        reg = self.regression(context)

        clas1 = self.classification1(x)
        clas2 = self.classification2(clas1)
        clas3 = torch.cat((clas2, context), 1)
        clas = self.classification3(clas3)

        y = reg * clas
        return y, reg, alphas, clas
