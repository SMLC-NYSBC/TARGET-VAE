import numpy as np
import torch
import os

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, delta=0.01, save_path='./', digits=3):
       
        self.patience = patience
        self.counter = 0
        self.min_loss = -np.inf
        self.early_stop = False
        self.delta = delta
        self.save_path = save_path
        self.digits = digits

        
        
    def __call__(self, loss, encoder, generator, epoch):
        
        if loss < self.min_loss + self.delta:
            self.counter += 1
            msg = '#EarlyStopping counter: {} out of {}'.format(self.counter, self.patience)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            msg = self.save_checkpoint(loss, encoder, generator, epoch)
            self.min_loss = loss
            self.counter = 0
            
        return msg

    
    
    def save_checkpoint(self, loss, encoder, generator, epoch):
        
        msg = '#Validation loss decreased {:.4f}: --> {:.4f}.  Saving model ...'.format(self.min_loss, loss)
        epoch_str = str(epoch+1).zfill(self.digits)

        generator.eval().cpu()
        torch.save(generator, self.save_path + 'generator.sav')

        encoder.eval().cpu()
        torch.save(encoder, self.save_path + 'inference.sav')

        return msg

    
