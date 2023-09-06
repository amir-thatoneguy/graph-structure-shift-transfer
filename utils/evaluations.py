import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from .datasets import create_description_from_probs
sns.set_theme(style = 'darkgrid')


def plot_results_dict(results, title = ".", legends = [], plot_args = dict()):
  import matplotlib.pyplot as plt
  import seaborn as sns
  sns.set_theme()

  plt.title(title)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')

  keys = results.keys()
  for key in keys:
    plt.plot(results[key], **plot_args)

  plt.legend(legends)


def plot_struct_sweep_histogram(results, experiments_iterator, key, subtitle = 'Target Accuracies'):
  x, y, z = create_mesh_from_results(results, experiments = experiments, key = key, aggregator = np.mean)

  plt.subplot(1,2,1)
  plt.scatter(x, y, c=z, cmap = 'magma', s=100)
  plt.title(subtitle)
  plt.colorbar()

  xlim = x.min() - 0.1, x.max() + 0.1
  ylim = y.min() - 0.1, y.max() + 0.1

  plt.xlim(xlim)
  plt.ylim(ylim)
  plt.xlabel('Source Hemophility')
  plt.ylabel('Target Hemophility')

  x, y, z_std = create_mesh_from_results(results, experiments = experiments, key = key, aggregator = np.std)


  plt.subplot(1,2,2)
  plt.scatter(x, y, c=z_std, cmap = 'magma', s = 100)
  plt.title(subtitle + ' (STD)')
  plt.colorbar()

  xlim = x.min() - 0.1, x.max() + 0.1
  ylim = y.min() - 0.1, y.max() + 0.1

  plt.xlim(xlim)
  plt.ylim(ylim)
  plt.xlabel('Source Hemophility')
  plt.ylabel('Target Hemophility')


def create_mesh_from_results(results, experiments, key = '', aggregator = np.mean):
  x, y, z = [], [], []

  for experiment in experiments:
    x.append(experiment[0][0])
    y.append(experiment[1][0])

    description = create_description_from_probs(experiment[0], experiment[1])
    z.append(aggregator(results[description][key]))

  return np.array(x), np.array(y), np.array(z)



# find accuracy from model outputs
def eval_cls_preds(cls_preds, y_true):
  preds = cls_preds.argmax(dim=1)

  num_samples = preds.shape[0]
  correct = int((preds == y_true).sum())
  acc = correct / num_samples

  return acc



# evaluate model accuracy
def eval_model_acc(model, data):
  model.eval()
  cls_preds, _ = model(data.x, data.edge_index)
  return eval_cls_preds(cls_preds, data.y)



def get_results(runs_losses, get_max_tgt_accs = True, get_max_src_accs = True):

  # find average losses
  results = dict()
  results['average_losses'] = average_dicts_of_number_lists(runs_losses)

  # find index of minimum target accuracy
  if get_max_tgt_accs:
    # get best accuracies at each run in an array
    results['max_target_accs'] = get_mapped_array_from_dicts(runs_losses, key = 'target_accs', map = np.max)

  if get_max_src_accs:
    results['max_source_accs'] = get_mapped_array_from_dicts(runs_losses, key = 'source_accs', map = np.max)

  return results


# find average dictionary of a set of dictionaries of scalar lists
def average_dicts_of_number_lists(list_of_dicts = []):

  n_dicts = len(list_of_dicts)
  sample_dict = list_of_dicts[0]
  keys = sample_dict.keys()

  average_dict = dict()

  for key in keys:
    average_dict[key] = np.zeros_like(np.array(sample_dict[key]))

  for dict_idx in range(n_dicts):
    curr_dict = list_of_dicts[dict_idx]

    for key in keys:
      average_dict[key] += np.array(curr_dict[key])/n_dicts

  return average_dict


# apply function to a certain key of a list of dicts
def get_mapped_array_from_dicts(list_of_dicts = [], key = 'source_accs', map = np.max):
  n_dicts = len(list_of_dicts)
  array = []

  for dict_idx in range(n_dicts):
    curr_dict = list_of_dicts[dict_idx]
    array.append(map(curr_dict[key]))

  return np.array(array)



def store_results(results = dict(), descriptions = [], root = './', experiments_name = ''):

  if root.endswith('/'):
    descriptions_path = root + experiments_name + '_' + 'experiments_descriptions.txt'

  else:
    descriptions_path = root + '/' + experiments_name + '_' + 'experiments_descriptions.txt'

  for description in descriptions:

    description_string = experiments_name + '_' + description

    if root.endswith('/'):
      path = root + description_string

    else:
      path = root + '/' + description_string

    # store average losses
    average_losses_root = path + '_' + 'average_losses'
    average_loss_dict = results[description]['average_losses']

    for key in average_loss_dict.keys():
      average_loss_path = average_losses_root + '_' + key
      np.save(average_loss_path, arr = average_loss_dict[key])

    # store max accuracy arrays
    if 'source_accs' in results[description].keys():
      source_accs_root = path + '_' + 'source_accs'
      np.save(source_accs_root, results[description]['source_accs'])

    if 'target_accs' in results[description].keys():
      target_accs_root = path + '_' + 'target_accs'
      np.save(target_accs_root, results[description]['target_accs'])


  with open(descriptions_path, 'w') as f:
    for description in descriptions:
        f.write(f"{description}\n")



# def load_results(root, experiments_name, average_results_keys = ['source', 'target', 'discriminator', 'total']):

#   average_results_dict = dict()
#   for key in average_results_keys:
#     average_results_dict[key] =





