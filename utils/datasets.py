from torch_geometric.datasets import StochasticBlockModelDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, NodeLoader
import numpy as np


class TransferDatasetExperiment():
  def __init__(self, description, data_args):

    possible_experiments = ['sbm']

    if description in possible_experiments:
      self.description = description

    else:
      print('non default dataset experiment')
      return

    self.src_data = None
    self.tgt_data = None
    self.data_args = None

    self.create_dataloaders(data_args)


  def create_dataloaders(self, data_args = dict()):
    match self.description:
      case 'sbm':
        src_data, tgt_data = create_transfer_symmetric_sbm_datasets(**data_args)

      case other:
        print('non default dataset experiment')
        return

    self.src_data = src_data
    self.tgt_data = tgt_data

  def get_dataloaders(self):
    return self.src_data, self.tgt_data




# creates diagonal probabilities for the sbm model. p/q are intra/inter class probability
def create_symmetric_block_probs(n_blocks, p, q):
  probs = []
  for i in range(n_blocks):
    block_probs = q*np.ones(n_blocks)
    block_probs[i] = p
    probs.append(block_probs.tolist())

  return probs





# creates an sbm symmetric dataset using pytorch geometric and returns nodes and edges
def create_sbm_symmetric_dataset(block_size, n_blocks, intra_block_prob, inter_block_prob, n_node_features, root='.'):
  block_sizes = [block_size]*n_blocks
  block_probs = create_symmetric_block_probs(n_blocks, intra_block_prob, inter_block_prob)
  dataset = StochasticBlockModelDataset(root=root , block_sizes = block_sizes, edge_probs = block_probs,
                                          num_channels = n_node_features)
  return dataset[0]




# creates dataloaders for target and source
def create_sbm_symmetric_transfer_dataloaders(dataset_s, dataset_t):
  x_src = dataset_s.x
  x_tgt = dataset_t.x

  edge_src = dataset_s.edge_index
  edge_tgt = dataset_t.edge_index

  #same features
  # x_src = torch.Tensor(np.ones_like(x_src))
  # x_tgt = torch.Tensor(np.ones_like(x_tgt))

  y_src = dataset_s.y
  y_tgt = dataset_t.y


  dataset_src = Data(x=x_src, edge_index = edge_src)
  dataset_tgt = Data(x=x_tgt, edge_index = edge_tgt)
  dataset_src.y = y_src
  dataset_tgt.y = y_tgt


  return dataset_src, dataset_tgt



# high-level transfer experiments with symmetric sbm datasets
def create_transfer_symmetric_sbm_datasets(root = '.', block_size = 300, n_blocks = 5, n_node_features = 100, source_probs = [0.8, 0.1], target_probs = [0.1, 0.8]):
  # source dataset
  intra_block_prob = source_probs[0]
  inter_block_prob = source_probs[1]
  dataset_s = create_sbm_symmetric_dataset(block_size, n_blocks, intra_block_prob, inter_block_prob, n_node_features, root = root + '/src')


  # target dataset
  intra_block_prob = target_probs[0]
  inter_block_prob = target_probs[1]
  dataset_t = create_sbm_symmetric_dataset(block_size, n_blocks, intra_block_prob, inter_block_prob, n_node_features, root = root + '/tgt')

  return create_sbm_symmetric_transfer_dataloaders(dataset_s, dataset_t)



def create_prob_sweep(p, q, step_size, n_steps):
    probs = []
    if p - step_size*n_steps < 0 or q + step_size*n_steps > 1:
        print('conditions (p - step_size*n_count < 0) and (q + step_size*n_count) are not met!')
        return

    for i in range(n_steps + 1):
        probs.append([p - step_size*i, q + step_size*i])

    return probs



def create_description_from_probs(source_probs=[0.1, 0.1], target_probs=[0.1, 0.1]):
  source_description = f'source : p,q = ({source_probs[0]:.3f}, {source_probs[1]:.3f})'
  target_description = f'target : p,q = ({target_probs[0]:.3f}, {target_probs[1]:.3f})'

  return source_description + '_' + target_description
