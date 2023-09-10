# torch.manual_seed(1234567)
from torch.autograd import Function
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import SAGEConv

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None



class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_features, num_classes, dann_lambda = 0.5):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.domain_classifier = GCNConv(hidden_channels, 2)
        self.dann_lambda = dann_lambda

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        y_class = self.conv2(x, edge_index)

        reverse_feature = ReverseLayerF.apply(x, self.dann_lambda)
        y_domain = self.domain_classifier(reverse_feature, edge_index)

        return y_class, y_domain


class GAT(torch.nn.Module):
    """Graph Attention Network"""
    def __init__(self, dim_in, dim_h, dim_out, heads=8, dann_lambda = 0.5):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1)
        self.domain_classifier = GATv2Conv(dim_h*heads, 2, heads=1)
        self.dann_lambda = dann_lambda


    def forward(self, x, edge_index):
        h = F.dropout(x, p=0.5, training=self.training)
        h = self.gat1(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        y_class = self.gat2(h, edge_index)

        reverse_feature = ReverseLayerF.apply(h, self.dann_lambda)
        y_domain = self.domain_classifier(reverse_feature, edge_index)

        return y_class, y_domain


class GSAGE(torch.nn.Module):
    """Graph Sage"""
    def __init__(self, in_channels = 2, hidden_channels = 8, out_channels = 2, dann_lambda = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels = in_channels, out_channels = hidden_channels)
        self.conv2 = SAGEConv(in_channels = hidden_channels, out_channels = out_channels)
        self.domain_classifier = SAGEConv(hidden_channels, 2)
        self.dann_lambda = dann_lambda


    def forward(self, x, edge_index):
        h = F.dropout(x, p=0.5, training=self.training)
        h = self.conv1(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        y_class = self.conv2(h, edge_index)

        reverse_feature = ReverseLayerF.apply(h, self.dann_lambda)
        y_domain = self.domain_classifier(reverse_feature, edge_index)

        return y_class, y_domain

    


def create_model(model_args):
  # model declaration

  match model_args['model_type']:
    case 'GCN':
      model = GCN(**model_args['model_hyperparams'])

    case 'GAT':
      model = GAT(**model_args['model_hyperparams'])

    case 'GSAGE':
        model = GSAGE(**model_args['model_hyperparams'])
        
    case other:
      print('model type is non-default!')
    
  return model
