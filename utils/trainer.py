import numpy as np
import torch
from .evaluations import eval_cls_preds


#simple training loop. returns losses
class Trainer():
  def __init__(self, device, model, data_src, data_tgt, optimizer, criterion, transfer = True, transfer_type = 'dann', log_epochs = True):

    self.device = device
    self.model = model
    self.data_src = data_src
    self.data_tgt = data_tgt
    self.optimizer = optimizer
    self.criterion = criterion

    self.transfer = transfer
    self.transfer_type = transfer_type
    self.log_epochs = log_epochs

    losses = dict()
    losses['source'] = []
    losses['source_accs'] = []

    if transfer:
      losses['target'] = []
      losses['target_accs'] = []

      if transfer_type == 'dann':
        losses['discriminator'] = []

    losses['total'] = []

    self.losses = losses






  def train(self, epochs = 5, store_tgt_accs = True, store_src_accs = True):
    # data_src = NodeLoader(data_src, batch_size = 100, shuffle = False)
    # data_tgt = NodeLoader(data_tgt, batch_size = 100, shuffle = False)


    if self.log_epochs == True:

      print('Training!')
      from tqdm import tqdm
      epochs_iterable = tqdm(range(epochs))

    else: epochs_iterable = range(epochs)




    for epoch in epochs_iterable:

        self.model.train()
        self.optimizer.zero_grad()

        #forward
        cls_src_preds, domain_src_preds = self.model(self.data_src.x, self.data_src.edge_index)

        if self.transfer:
          cls_tgt_preds, domain_tgt_preds = self.model(self.data_tgt.x, self.data_tgt.edge_index)

        # source classification loss
        src_cls_loss = self.criterion(cls_src_preds, self.data_src.y)


        # transfer learning experiment
        if self.transfer:
          #adversarial discriminator loss
          if self.transfer_type == 'dann':
            domain_dsc_src_loss = self.criterion(domain_src_preds, torch.ones_like(domain_src_preds).to(self.device))
            domain_dsc_tgt_loss = self.criterion(domain_tgt_preds, torch.zeros_like(domain_tgt_preds).to(self.device))
            discriminator_loss = domain_dsc_src_loss/2

          total_loss = discriminator_loss + src_cls_loss


        #optimizer step
        total_loss.backward()
        self.optimizer.step()




        self.model.eval()

        #store losses
        self.losses['source'].append(src_cls_loss.detach().cpu().numpy())
        self.losses['total'].append(total_loss.detach().cpu().numpy())

        if store_tgt_accs:
          self.losses['target_accs'].append(eval_cls_preds(cls_tgt_preds, self.data_tgt.y))

        if store_src_accs:
          self.losses['source_accs'].append(eval_cls_preds(cls_src_preds, self.data_src.y))


        if self.transfer:
          eval_loss = self.criterion(cls_tgt_preds, self.data_tgt.y)
          self.losses['target'].append(eval_loss.detach().cpu().numpy())

          if self.transfer_type == 'dann':
            self.losses['discriminator'].append(discriminator_loss.detach().cpu().numpy())





    if self.log_epochs == True: print('\nDone')


