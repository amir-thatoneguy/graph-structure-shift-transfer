from utils.datasets import *
from utils.evaluations import *
from utils.models import *
from utils.trainer import *
import torch


def multiple_runs(experiment_description, n_runs, epochs, model_args = dict(), data_args = dict(), optimizer_args = dict()):

  from tqdm import tqdm
  runs_losses = []

  # experiments loop
  for run in tqdm(range(n_runs)):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create data
    data_experiment = TransferDatasetExperiment(description = 'sbm', data_args = data_args)
    data_src, data_tgt = data_experiment.get_dataloaders()
    data_src = data_src.to(device)
    data_tgt = data_tgt.to(device)

    # create model
    model = create_model(model_args).to(device)

    # training stuff initializations
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_args)
    criterion = torch.nn.CrossEntropyLoss()

    # training loop
    trainer = Trainer(device, model, data_src, data_tgt, optimizer, criterion, epochs, log_epochs = False)
    trainer.train(epochs = n_epochs)
    runs_losses.append(trainer.losses)

  return get_results(runs_losses)