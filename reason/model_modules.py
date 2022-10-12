import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2, drop_prob=0.2):

        super(GRUNet, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h=None):
        # x: B x T x nf
        # h: n_layers x B x nf
        B, T, nf = x.size()

        if h is None:
            h = self.init_hidden(B)

        out, h = self.gru(x, h)

        # out: B x T x nf
        # h: n_layers x B x nf
        out = self.fc(self.relu(out.contiguous().view(B * T, self.hidden_dim)))
        out = out.view(B, T, self.output_dim)

        # out: B x output_dim
        return out[:, -1]

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        return hidden

class MLP(nn.Module):
    """
    MLP Model.
    Args:
        input_dim (int) : Dimension of the network input.
        output_dim (int): Dimension of the network output.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means this MLP consists of two
            hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.
    Return:
        The output torch.Tensor of the MLP
    """

    def __init__(self, input_dim, output_dim, hidden_sizes, hidden_nonlinearity=F.relu, hidden_w_init=nn.init.xavier_normal_, hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None, output_w_init=nn.init.xavier_normal_, output_b_init=nn.init.zeros_, layer_normalization=False):
        super().__init__()

        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_nonlinearity = hidden_nonlinearity
        self._output_nonlinearity = output_nonlinearity
        self._layer_normalization = layer_normalization
        self._layers = nn.ModuleList()

        prev_size = input_dim

        for size in hidden_sizes:
            layer = nn.Linear(prev_size, size)
            hidden_w_init(layer.weight)
            hidden_b_init(layer.bias)
            self._layers.append(layer)
            prev_size = size

        layer = nn.Linear(prev_size, output_dim)
        output_w_init(layer.weight)
        output_b_init(layer.bias)
        self._layers.append(layer)

    def forward(self, input_val):
        """Forward method."""
        B = input_val.size(0)
        x = input_val.view(B, -1)
        for layer in self._layers[:-1]:
            x = layer(x)
            if self._hidden_nonlinearity is not None:
                x = self._hidden_nonlinearity(x)
            if self._layer_normalization:
                x = nn.LayerNorm(x.shape[1]).to(x.device)(x)

        x = self._layers[-1](x)
        if self._output_nonlinearity is not None:
            x = self._output_nonlinearity(x)

        return x

class CNNet(nn.Module):
    def __init__(self, ks, nf_in, nf_hidden, nf_out, do_prob=0.):
        super(CNNet, self).__init__()

        self.pool = nn.MaxPool1d(
            kernel_size=2, stride=None, padding=0,
            dilation=1, return_indices=False,
            ceil_mode=False)

        self.conv1 = nn.Conv1d(nf_in, nf_hidden, kernel_size=ks, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(nf_hidden)
        self.conv2 = nn.Conv1d(nf_hidden, nf_hidden, kernel_size=ks, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(nf_hidden)
        self.conv3 = nn.Conv1d(nf_hidden, nf_hidden, kernel_size=ks, stride=1, padding=0)
        self.bn3 = nn.BatchNorm1d(nf_hidden)
        self.conv_predict = nn.Conv1d(nf_hidden, nf_out, kernel_size=1)
        self.conv_attention = nn.Conv1d(nf_hidden, 1, kernel_size=1)
        self.dropout_prob = do_prob

    def forward(self, inputs):
        # inputs: B x T x nf_in
        inputs = inputs.transpose(1, 2)

        # inputs: B x nf_in x T
        x = F.relu(self.conv1(inputs))
        x = self.bn1(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        pred = self.conv_predict(x)

        # ret: B x nf_out
        ret = pred.max(dim=2)[0]
        return ret


class PropNet(nn.Module):

    def __init__(self, node_dim_in, edge_dim_in, nf_hidden, node_dim_out, edge_dim_out,
                 edge_type_num=1, pstep=2, batch_norm=1, use_gpu=True):

        super(PropNet, self).__init__()

        self.node_dim_in = node_dim_in
        self.edge_dim_in = edge_dim_in
        self.nf_hidden = nf_hidden

        self.node_dim_out = node_dim_out
        self.edge_dim_out = edge_dim_out

        self.edge_type_num = edge_type_num
        self.pstep = pstep

        # node encoder
        modules = [
            nn.Linear(node_dim_in, nf_hidden),
            nn.ReLU()]
        if batch_norm == 1:
            modules.append(nn.BatchNorm1d(nf_hidden))
        self.node_encoder = nn.Sequential(*modules)

        # edge encoder
        self.edge_encoders = nn.ModuleList()
        for i in range(edge_type_num):
            modules = [
                nn.Linear(node_dim_in * 2 + edge_dim_in, nf_hidden),
                nn.ReLU()]
            if batch_norm == 1:
                modules.append(nn.BatchNorm1d(nf_hidden))

            self.edge_encoders.append(nn.Sequential(*modules))

        # node propagator
        modules = [
            # input: node_enc, node_rep, edge_agg
            nn.Linear(nf_hidden * 3, nf_hidden),
            nn.ReLU(),
            nn.Linear(nf_hidden, nf_hidden),
            nn.ReLU()]
        if batch_norm == 1:
            modules.append(nn.BatchNorm1d(nf_hidden))
        self.node_propagator = nn.Sequential(*modules)

        # edge propagator
        self.edge_propagators = nn.ModuleList()
        for i in range(pstep):
            edge_propagator = nn.ModuleList()
            for j in range(edge_type_num):
                modules = [
                    # input: node_rep * 2, edge_enc, edge_rep
                    nn.Linear(nf_hidden * 3, nf_hidden),
                    nn.ReLU(),
                    nn.Linear(nf_hidden, nf_hidden),
                    nn.ReLU()]
                if batch_norm == 1:
                    modules.append(nn.BatchNorm1d(nf_hidden))
                edge_propagator.append(nn.Sequential(*modules))

            self.edge_propagators.append(edge_propagator)

        # node predictor
        modules = [
            nn.Linear(nf_hidden * 2, nf_hidden),
            nn.ReLU()]
        if batch_norm == 1:
            modules.append(nn.BatchNorm1d(nf_hidden))
        modules.append(nn.Linear(nf_hidden, node_dim_out))
        self.node_predictor = nn.Sequential(*modules)

        # edge predictor
        modules = [
            nn.Linear(nf_hidden * 2, nf_hidden),
            nn.ReLU()]
        if batch_norm == 1:
            modules.append(nn.BatchNorm1d(nf_hidden))
        modules.append(nn.Linear(nf_hidden, edge_dim_out))
        self.edge_predictor = nn.Sequential(*modules)

    def forward(self, node_rep, edge_rep=None, edge_type=None, start_idx=0,
                ignore_node=False, ignore_edge=False):
        # node_rep: B x N x node_dim_in
        # edge_rep: B x N x N x edge_dim_in
        # edge_type: B x N x N x edge_type_num
        # start_idx: whether to ignore the first edge type
        B, N, _ = node_rep.size()

        # node_enc
        node_rep = node_rep.contiguous()
        node_enc = self.node_encoder(node_rep.view(-1, self.node_dim_in)).view(B, N, self.nf_hidden)

        # edge_enc
        node_rep_r = node_rep[:, :, None, :].repeat(1, 1, N, 1)
        node_rep_s = node_rep[:, None, :, :].repeat(1, N, 1, 1)
        if edge_rep is not None:
            tmp = torch.cat([node_rep_r, node_rep_s, edge_rep], 3)
        else:
            tmp = torch.cat([node_rep_r, node_rep_s], 3)

        edge_encs = []
        for i in range(start_idx, self.edge_type_num):
            edge_enc = self.edge_encoders[i](tmp.view(B * N * N, -1)).view(B, N, N, 1, self.nf_hidden)
            edge_encs.append(edge_enc)
        # edge_enc: B x N x N x edge_type_num x nf
        edge_enc = torch.cat(edge_encs, 3)

        if edge_type is not None:
            edge_enc = edge_enc * edge_type.view(B, N, N, self.edge_type_num, 1)[:, :, :, start_idx:]

        # edge_enc: B x N x N x nf
        edge_enc = edge_enc.sum(3)

        for i in range(self.pstep):
            if i == 0:
                node_effect = node_enc
                edge_effect = edge_enc

            # calculate edge_effect
            node_effect_r = node_effect[:, :, None, :].repeat(1, 1, N, 1)
            node_effect_s = node_effect[:, None, :, :].repeat(1, N, 1, 1)
            tmp = torch.cat([node_effect_r, node_effect_s, edge_effect], 3)

            edge_effects = []
            for j in range(start_idx, self.edge_type_num):
                edge_effect = self.edge_propagators[i][j](tmp.view(B * N * N, -1))
                edge_effect = edge_effect.view(B, N, N, 1, self.nf_hidden)
                edge_effects.append(edge_effect)
            # edge_effect: B x N x N x edge_type_num x nf
            edge_effect = torch.cat(edge_effects, 3)

            if edge_type is not None:
                edge_effect = edge_effect * edge_type.view(B, N, N, self.edge_type_num, 1)[:, :, :, start_idx:]

            # edge_effect: B x N x N x nf
            edge_effect = edge_effect.sum(3)

            # calculate node_effect
            edge_effect_agg = edge_effect.sum(2)
            tmp = torch.cat([node_enc, node_effect, edge_effect_agg], 2)
            node_effect = self.node_propagator(tmp.view(B * N, -1)).view(B, N, self.nf_hidden)

        node_effect = torch.cat([node_effect, node_enc], 2).view(B * N, -1)
        edge_effect = torch.cat([edge_effect, edge_enc], 3).view(B * N * N, -1)

        # node_pred: B x N x node_dim_out
        # edge_pred: B x N x N x edge_dim_out
        if ignore_node:
            edge_pred = self.edge_predictor(edge_effect)
            return edge_pred.view(B, N, N, -1)
        if ignore_edge:
            node_pred = self.node_predictor(node_effect)
            return node_pred.view(B, N, -1)

        node_pred = self.node_predictor(node_effect).view(B, N, -1)
        edge_pred = self.edge_predictor(edge_effect).view(B, N, N, -1)
        return node_pred, edge_pred
