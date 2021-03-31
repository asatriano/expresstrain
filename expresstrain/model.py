# %%
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import numpy as np

from tqdm import tqdm

from pdb import set_trace

RANDOM_SEED=42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# %%

class ExpressTrain:
    '''Express Train Arguments:
        train_loader, valid_loader, test_loader,
        model, device, num_classes,  
        learning_rate, optimizer, metric_used, \
        loss_fn=None,
        class_weights=None, scheduler=None,
        bce_use=False, lr_adjuster_on_val=None, lr_div_factor=None, \
        survival=None, one_cycle_epochs=None, \
        metric_from_whole=True, \
        backward_every=1, fp16=False,
        save_every=None,
        path_performance=None,
        path_performance_and_model=None
        
        Example subclassing:

        import expresstrain as et

        class CustomExpressTrain(et.ExpressTrain):
            def __init__(self, **kwargs):
                super(CustomExpressTrain, self).__init__()
                self.initialize_all(kwargs)

            def on_train_epoch_start(self):
                print(f"Pre-train message: You're doing great: epoch {self.epoch}")

        trainer=CustomExpressTrain(train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
                                model=model, num_classes=num_classes,  
                                learning_rate=learning_rate, optimizer=optimizer, metric_used=metric_used, 
                                bce_use=bce_use, loss_fn=None,
                                class_weights=class_weights,
                                scheduler=scheduler, lr_adjuster_on_val=lr_adjuster, lr_div_factor=lr_div_factor, 
                                survival=survival, one_cycle_epochs=one_cycle_epochs, 
                                metric_from_whole=metric_from_whole, 
                                backward_every=backward_every, fp16=fp16, device=device, 
                                save_every=save_every,
                                path_performance=path_performance,
                                path_performance_and_model=path_performance_and_model)

        trainer.fit(epochs=epochs)

        Args:

        train_loader (torch.utils.data.dataloader.DataLoader): training dataloader
        valid_loader (torch.utils.data.dataloader.DataLoader): validation data loader
        test_loader (torch.utils.data.dataloader.DataLoader): test dataloader
        model (subclassed from torch.nn.Module): model, returning logits
        num_classes (int): number of classes (2 or more)
        learning_rate (float): learning_rate
        optimizer (torch.optim): optimizer
        metric_used (function): metric_used

        Optional Args:
        
        bce_use (bool): whether Binary Cross Entropy Loss should be used (default: False)
        loss_fn (torch.nn.modules.loss): non-default loss function (defaults are Binary Cross Entropy or CrossEntropy) (default: None)
        class_weights=class_weights (default: None),
        scheduler (torch.optim.lr_scheduler): scheduler (default: None)
        lr_adjuster_on_val (torch.optim.lr_scheduler): scheduler acting on validation metric (default: None)
        lr_div_factor (float): div_factor for torch.optim.lr_scheduler.OneCycleLR (default: None)
        survival (bool): indicates if this is a survival problem (default: False)
        one_cycle_epochs (int): new onecycle begins every one_cycle_epochs epochs (default: None) 
        metric_from_whole (bool): indicates whether metric is obtained from average over batches of a single epoch, or from predictions throughout the epoch (default: True) 
        backward_every (int): backward pass is performed every backward_every batches (gradient accumulation) (default: 1)
        fp16 (bool): indicates usage of Automatic Mixed PRecision (default: False)
        device (torch.device): device for analysis (default: torch.device('cpu'))
        save_every (int): progress is saved every save_every epochs (default: 5)
        path_performance (str): loss and metrics are saved in path_performance (default: None)
        path_performance_and_model (str): loss, metrics, and model parameters are saved inpath_performance_and_model (default: None)

        More:
        You can provide datasets as kwargs if you define methods to build dataloaders

    '''

    def __init__(self, **kwargs):
        self.initialize_all(kwargs)

    def initialize_all(self, kwargs):
        self.valid_loader=None # validation_loader to use for internal validation
        self.test_loader=None # test_loader to assess performance at inference on holdout
        self.bce_use=False # whether we should use BinaryCrossEntropyLoss
        self.device=torch.device('cpu') # pytorch device to use for analysis
        self.loss_fn=None   # if not None, specify desired loss function
        self.class_weights=None # class_weights according to pytorch convention
        self.scheduler=None # scheduler according to pytorch convention
        self.lr_adjuster_on_val=None # pytorch scheduler depending on validation results
        self.lr_div_factor=None # if not None, this activates one_cycle traing and divides lr
        self.survival=False #if not False, if triggers survival-specific loss functions
        self.one_cycle_epochs=None # how many epochs each one-cycle trainign cycle should last
        self.metric_from_whole=True # should metric be computed by single batch of whole epoch
        self.backward_every=1 # backward is performed every specified number of epochs
        self.fp16=False # half precision (nvidia amp) training: saves memory
        self.save_every=5 # saing loss, metric and model happens every specified epochs
        self.path_performance=None # path where loss and metrics are saved
        self.path_performance_and_model=None # path where loss, metrics, and model params are saved
        self.use_progbar=False # input True to use progress bar
        for key in kwargs: #all other parameyters are converted into attributes
            setattr(self, key, kwargs[key])

    def compute_output_and_loss(self, data, target):
        '''Inputs: data, target
            Outputs: output, loss'''
        if self.survival==True:
            assert(target.shape[1]==2)
        output=self.model(data)
        loss=self.loss_fn(output, target)
        if self.survival==True:
            print("Add L1 Loss: Not implemented here yet")
        return output, loss
    
    def forward_by_fp16(self, data, target):
        if self.fp16==True:
            with torch.cuda.amp.autocast():
                output, loss=self.compute_output_and_loss(data, target)
        else:
            output, loss=self.compute_output_and_loss(data, target)
        return output, loss

    def on_forward(self, data, target, train):
        '''Inputs: data, target, train (i.e. train mode)
        Outputs: output, loss'''
        if self.survival==False:
            if self.bce_use==True:
                self.loss_fn=torch.nn.BCEWithLogitsLoss()
            else:
                if self.loss_fn is None:
                    self.loss_fn=torch.nn.CrossEntropyLoss(weight=self.class_weights \
                                                    if self.class_weights is not None else None)
        else:
            print("Survival Loss Not implemented here yet")
        if self.survival==True:
            assert(target.shape[1]==2)
            assert(self.bce_use==False)
        if train==True:
            output, loss = self.forward_by_fp16(data, target)
        else:
            with torch.no_grad():
                output, loss = self.forward_by_fp16(data, target)
        return output, loss


    def on_backward(self, loss, batch, batches):
        '''Inputs: loss, batch, batches
        Outputs: None
        To do: use loss to do backward, update self.optimizer, zero gradient
        Use self.scaler if self.fp16==True'''
        if self.fp16==True:
            assert(self.scaler is not None)
        if self.fp16==False:
            loss.backward()
        else:
            self.scaler.scale(loss).backward()
        
        if ((batch+1) % self.backward_every ==0) or (batch==(batches-1)):
            if self.fp16==False:
                self.optimizer.step()
            else:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()
        return
    
    def prepare_data_label_for_forward(self, data, target):
        '''Input: data, target
        Output: data, target'''
        if self.survival==True:
            assert(target.shape[1]==2)
        if self.bce_use==True:
            target=target.to(torch.float)
        else:
            target=target.squeeze(-1)
        data, target = data.to(self.device), target.to(self.device)
        return data, target

    def compute_pred_from_output(self, output):
        '''Input: output
        Output: pred'''
        pred=torch.argmax(output, dim=1)
        return pred

    def prepare_pred_label_for_metric(self, output, target):
        '''Input: ouput, target
        Output: pred, target, to be use for the metric'''
        if self.survival==True:
            assert(target.shape[1]==2)
            print("Prediction for survival not implemented here yet")
        else:
            if self.bce_use==True:
                target_for_metric=target.to(torch.long)
                pred_for_metric=nn.Sigmoid()(output)
            else:
                target_for_metric=target
                pred_for_metric=self.compute_pred_from_output(output)
        return pred_for_metric, target_for_metric

    def set_enumerable(self, data_loader):
        if self.use_progbar == True:
            return tqdm(data_loader, total=len(data_loader))
        else:
            return data_loader
    
    def progbar_close(self, enumerable):
        if self.use_progbar == True:
            enumerable.close()
    
    def on_one_epoch(self, epoch, data_loader, train, inference_on_holdout):
        '''Input: epoch, data_loader, train status, inference_on_holdout status
        Output: loss_epoch, metric_epoch, pred_list[1:], target_list[1:]
        Hooks available:
        on_epoch_start, on_epoch_end,'''
        enumerable=set_enumerable(data_loader)
        loss_list=[]
        metric_list=[]
        shape_log_list=[1,1] if self.bce_use==True else [1]
        pred_list=torch.empty(shape_log_list)
        target_list=torch.empty(shape_log_list)
        # if bce_use==True:
        target_list=target_list.to(torch.long)
        if train==True:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()
        if (self.one_cycle_epochs is not None) and ((epoch+1) % self.one_cycle_epochs==0) and (train==True):
                self.scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer=self.optimizer,
                                                    max_lr=self.learning_rate,
                                                    steps_per_epoch=len(data_loader) if self.backward_every==1 else int((len(data_loader)-1)/self.backward_every_n_epochs),
                                                    div_factor=self.lr_div_factor,
                                                    epochs=self.one_cycle_epochs if self.backward_every==1 else (self.one_cycle_epochs+1)*self.backward_every_n_epochs*2,
                                                    verbose=False
                                                    )
        self.batches_total=len(data_loader)
        for batch_idx, (data, target) in enumerate(enumerable):
            self.batch=batch_idx
            self.on_batch_start()
            data, target=self.prepare_data_label_for_forward(data=data,
                                                        target=target
                                                        )
            output, loss=self.on_forward(data=data, 
                                    target=target,
                                    train=train,
                                    )
            if train==True:
                self.on_backward(loss=loss, 
                                batch=batch_idx, 
                                batches=len(data_loader)
                                )
            loss_list.append(loss.cpu().item())
            pred, target_metric=self.prepare_pred_label_for_metric(output=output,
                                                                target=target, 
                                                                )
            if self.num_classes==len(target_metric.unique()):
                metric_list.append(self.metric_used(pred, target_metric).cpu().item())
            if self.metric_from_whole==True:
                pred_list=torch.cat((pred_list, pred.cpu()), dim=0)
                target_list=torch.cat((target_list, target_metric.cpu()), dim=0)

            self.progbar_close(enumerable)
            self.on_batch_end()
        if self.metric_from_whole==True:
            metric_epoch=100*self.metric_used(pred_list[1:], target_list[1:])
        else:
            metric_epoch=100*sum(metric_list)/len(metric_list)
        loss_epoch=sum(loss_list)/len(loss_list)
        
        return loss_epoch, metric_epoch, pred_list[1:], target_list[1:]

    def save_progress(self, epoch,
            train_loss_list, train_metric_list, val_loss_list, val_metric_list):
        '''No outputs: saves progress'''
        if ((epoch+1) % self.save_every==0.) and ((self.path_performance is not None) or (self.path_performance_and_model is not None)):
            if self.path_performance is not None:
                torch.save(
                    {
                    'train_loss': train_loss_list,
                    'train_metric': train_metric_list,
                    'val_loss': val_loss_list,
                    'val_metric': val_metric_list
                    },
                    self.path_performance)
            if self.path_performance_and_model is not None:
                torch.save(
                    {
                    'train_loss': train_loss_list,
                    'train_metric': train_metric_list,
                    'val_loss': val_loss_list,
                    'val_metric': val_metric_list,
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    },
                    self.path_performance_and_model)

    def on_one_train_epoch(self, epoch, data_loader):
        '''Input: epoch, data_loader
        Output: loss, metric, predictions list, target list'''
        loss_epoch, metric_epoch, \
            pred_list, target_list = self.on_one_epoch(
                                                    epoch=epoch,
                                                    data_loader=data_loader,
                                                    train=True,
                                                    inference_on_holdout=False
                                                    )
        return loss_epoch, metric_epoch, pred_list, target_list

    def on_one_valid_epoch(self, epoch, data_loader):
        '''Input: epoch, data_loader
        Output: loss, metric, predictions list, target list'''
        loss_epoch, metric_epoch, \
            pred_list, target_list = self.on_one_epoch(
                                                    epoch=epoch,
                                                    data_loader=data_loader,
                                                    train=False,
                                                    inference_on_holdout=False
                                                    )
        return loss_epoch, metric_epoch, pred_list, target_list

    def on_one_test_epoch(self, epoch, data_loader):
        '''Input: epoch, data_loader
        Output: loss, metric, predictions list, target list'''
        loss_epoch, metric_epoch, \
            pred_list, target_list = self.on_one_epoch(
                                                    epoch=epoch,
                                                    data_loader=data_loader,
                                                    train=False,
                                                    inference_on_holdout=True
                                                    )
        return loss_epoch, metric_epoch, pred_list, target_list

    def lr_adjust_on_val(self):
        self.lr_adjuster_on_val.step(self.val_metric_list[-1])

    def fit(self, epochs):
        '''Runs training and validation loop for epochs, then tests, if test_loader
        is available.
        Hooks available:
        on_one_epoch,
        on_one_train_epoch,
        on_one_valid_epoch,
        one_one_test_epoch,
        on_train_epoch_start, on_train_epoch_end
        on_valid_epoch_start, on_valid_epoch_end
        on_test_epoch_start, on test_epoch_end'''
        if self.fp16==True:
            self.scaler=torch.cuda.amp.GradScaler()
        else:
            self.scaler=None

        self.model.to(self.device)

        self.train_loss_list=[]
        self.train_metric_list=[]
        self.val_loss_list=[]
        self.val_metric_list=[]

        for epoch in range(epochs):

            self.epoch=epoch

            self.on_epoch_start()
                    # epoch, data_loader, train, inference_on_holdout
            self.on_train_epoch_start()
            loss_epoch, metric_epoch, _, _ = self.on_one_train_epoch(
                                                    epoch=epoch,
                                                    data_loader=self.train_loader,
                                                    )
            self.train_loss_list.append(loss_epoch)
            self.train_metric_list.append(metric_epoch)
            self.on_train_epoch_end()

            if self.valid_loader is not None:
                self.on_valid_epoch_start()
                loss_epoch, metric_epoch, _, _ = self.on_one_valid_epoch(
                                                        epoch=epoch,
                                                        data_loader=self.valid_loader,
                                                        )
                self.val_loss_list.append(loss_epoch)
                self.val_metric_list.append(metric_epoch)
                self.on_valid_epoch_end()

                if (self.lr_adjuster_on_val is not None) and (self.valid_loader is not None):
                    self.lr_adjust_on_val()
            
            print(f"Epoch {epoch+1}/{epochs}, {self.metric_used.__name__}_train: {self.train_metric_list[-1]:.2f}")
            if self.valid_loader is not None:
                print(f"Epoch {epoch+1}/{epochs}, {self.metric_used.__name__}_valid: {self.val_metric_list[-1]:.2f}")
            
            if self.save_every is not None:

                # SAVING
                # epoch,
            # train_loss_list, train_metric_list, val_loss_list, val_metric_list
                self.save_progress(epoch,
                                    self.train_loss_list,
                                    self.train_metric_list,
                                    self.val_loss_list,
                                    self.val_metric_list)
            
            self.on_epoch_end()


        if self.test_loader is not None:
            self.test_loss_list=[]
            self.test_metric_list=[]

            self.on_test_start()
            loss_epoch, metric_epoch, self.pred_test_list, \
                self.target_test_list=self.on_one_test_epoch(
                                                    epoch=epoch,
                                                    data_loader=self.test_loader,
                                                    )
            self.test_loss_list.append(loss_epoch)
            self.test_metric_list.append(metric_epoch)
            self.on_test_end()

            print(f"{self.metric_used.__name__}_test: {self.test_metric_list[-1]:.2f}")

    def on_batch_start(self):
        pass

    def on_batch_end(self):
        pass

    def on_epoch_start(self):
        pass

    def on_epoch_end(self):
        pass

    def on_train_epoch_start(self):
        pass

    def on_train_epoch_end(self):
        pass

    def on_valid_epoch_start(self):
        pass

    def on_valid_epoch_end(self):
        pass

    def on_test_start(self):
        pass

    def on_test_end(self):
        pass