# Refactoring of training loop
# (support functions for training loop)

# %%
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import numpy as np

from pdb import set_trace

RANDOM_SEED=42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# %%
def compute_output_and_loss(model, data, target, loss_fn, survival=False):
    if survival==True:
        assert(target.shape[1]==2)
    output=model(data)
    # set_trace()
    loss=loss_fn(output, target)
    if survival==True:
        print("Add L1 Loss: Not implemented here yet")
    return output, loss

# %%   
def forward_by_fp16(model, data, target, loss_fn, survival, fp16):
    if fp16==True:
        with torch.cuda.amp.autocast():
            output, loss=compute_output_and_loss(model=model,
                                            data=data, 
                                            target=target, 
                                            loss_fn=loss_fn, 
                                            survival=survival)
    else:
        output, loss=compute_output_and_loss(model=model,
                                            data=data, 
                                            target=target, 
                                            loss_fn=loss_fn, 
                                            survival=survival)
    return output, loss

# %%
def on_forward(model, data, target, loss_fn=None, survival=False, bce_use=False, class_weights=None, train=True, fp16=False):
    if survival==False:
        if bce_use==True:
            loss_fn=torch.nn.BCEWithLogitsLoss()
        else:
            if loss_fn is None:
                loss_fn=torch.nn.CrossEntropyLoss(weight=class_weights \
                                                if class_weights is not None else None)
    else:
        print("Survival Loss Not implemented here yet")
    if survival==True:
        assert(target.shape[1]==2)
        assert(bce_use==False)
    if train==True:
        output, loss = forward_by_fp16(model=model,
                                    data=data,
                                    target=target,
                                    loss_fn=loss_fn,
                                    survival=survival,
                                    fp16=fp16)
        
    else:
        with torch.no_grad():
            output, loss = forward_by_fp16(model=model,
                                        data=data,
                                        target=target,
                                        loss_fn=loss_fn,
                                        survival=survival,
                                        fp16=fp16)
    return output, loss

# %%
def on_backward(loss, optimizer, batch, batches, scheduler, fp16, scaler, backward_every=1):
    if fp16==True:
        assert(scaler is not None)
    if fp16==False:
        loss.backward()
    else:
        scaler.scale(loss).backward()
    
    if ((batch+1) % backward_every ==0) or (batch==(batches-1)):
        if fp16==False:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
    return

# %%
def prepare_data_label_for_forward(data, target, device, bce_use, survival=False):
    if survival==True:
        assert(target.shape[1]==2)
    if bce_use==True:
        target=target.to(torch.float)
    else:
        target=target.squeeze(-1)
    data, target = data.to(device), target.to(device)
    return data, target

# %%
def compute_pred_from_output(output):
    pred=torch.argmax(output, dim=1)
    return pred

# %%
def prepare_pred_label_for_metric(output, target, bce_use, survival=False):
    if survival==True:
        assert(target.shape[1]==2)
        print("Prediction for survival not implemented here yet")
    else:
        if bce_use==True:
            target_for_metric=target.to(torch.long)
            pred_for_metric=nn.Sigmoid()(output)
        else:
            target_for_metric=target
            pred_for_metric=compute_pred_from_output(output)
    return pred_for_metric, target_for_metric
# %%
def on_one_epoch(epoch, data_loader, model, device, num_classes, bce_use, 
                learning_rate, optimizer, metric_used, \
                loss_fn=None,
                class_weights=None, scheduler=None, 
                lr_adjuster_on_val=None, lr_div_factor=None, \
                survival=False, one_cycle_epochs=None, \
                metric_from_whole=True, \
                train=False, inference_on_holdout=False,
                scaler=None, backward_every=1, fp16=False):
    loss_list=[]
    metric_list=[]
    shape_log_list=[1,1] if bce_use==True else [1]
    pred_list=torch.empty(shape_log_list)
    target_list=torch.empty(shape_log_list)
    # if bce_use==True:
    target_list=target_list.to(torch.long)
    if train==True:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()
    if (one_cycle_epochs is not None) and ((epoch+1) % one_cycle_epochs==0) and (train==True):
            scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                max_lr=learning_rate,
                                                steps_per_epoch=len(data_loader) if backward_every==1 else int((len(data_loader)-1)/backward_every_n_epochs),
                                                div_factor=lr_div_factor,
                                                epochs=one_cycle_epochs if backward_every==1 else (one_cycle_epochs+1)*backward_every_n_epochs*2,
                                                verbose=False
                                                )
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target=prepare_data_label_for_forward(data=data,
                                                    target=target,
                                                    device=device,
                                                    bce_use=bce_use,
                                                    survival=survival)
        output, loss=on_forward(model=model,
                                data=data, 
                                target=target,
                                loss_fn=loss_fn,
                                survival=survival,
                                bce_use=bce_use,
                                class_weights=class_weights,
                                train=train,
                                fp16=fp16)
        if train==True:
            on_backward(loss=loss,
                        optimizer=optimizer, 
                        batch=batch_idx, 
                        batches=len(data_loader),
                        scheduler=scheduler, 
                        fp16=fp16, 
                        scaler=scaler, 
                        backward_every=backward_every)
        loss_list.append(loss.cpu().item())
        pred, target_metric=prepare_pred_label_for_metric(output=output,
                                                        target=target, 
                                                        bce_use=bce_use,
                                                        survival=survival)
        if num_classes==len(target_metric.unique()):
            metric_list.append(metric_used(pred, target_metric).cpu().item())
        if metric_from_whole==True:
            pred_list=torch.cat((pred_list, pred.cpu()), dim=0)
            target_list=torch.cat((target_list, target_metric.cpu()), dim=0)
    if metric_from_whole==True:
        metric_epoch=100*metric_used(pred_list[1:], target_list[1:])
    else:
        metric_epoch=100*sum(metric_list)/len(metric_list)
    loss_epoch=sum(loss_list)/len(loss_list)
    if train==False and inference_on_holdout==False:
        if lr_adjuster_on_val is not None:
            lr_adjuster_on_val.step(metric_epoch)
    
    return model, optimizer, scheduler, lr_adjuster_on_val, \
        loss_epoch, metric_epoch, pred_list[1:], target_list[1:]

# %%
def save_progress(epoch, save_every, model, optimizer, path_performance, path_performance_and_model,
                train_loss_list, train_metric_list, val_loss_list, val_metric_list):
    if ((epoch+1) % save_every==0.) and ((path_performance is not None) or (path_performance_and_model is not None)):
        if path_performance is not None:
            torch.save(
                {
                'train_loss': train_loss_list,
                'train_metric': train_metric_list,
                'val_loss': val_loss_list,
                'val_metric': val_metric_list
                },
                path_performance)
        if path_performance_and_model is not None:
            torch.save(
                {
                'train_loss': train_loss_list,
                'train_metric': train_metric_list,
                'val_loss': val_loss_list,
                'val_metric': val_metric_list,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                },
                path_performance_and_model)

# %%
def fit(epochs,
        train_loader, valid_loader, test_loader,
        model, device, num_classes, bce_use, 
        learning_rate, optimizer, metric_used, \
        loss_fn=None,
        class_weights=None, scheduler=None,
        lr_adjuster_on_val=None, lr_div_factor=None, \
        survival=None, one_cycle_epochs=None, \
        metric_from_whole=True, \
        backward_every=1, fp16=False,
        save_every=None,
        path_performance=None,
        path_performance_and_model=None):

    if fp16==True:
        scaler=torch.cuda.amp.GradScaler()
    else:
        scaler=None

    model.to(device)

    train_loss_list=[]
    train_metric_list=[]
    val_loss_list=[]
    val_metric_list=[]

    for epoch in range(epochs):

        model, optimizer, scheduler, lr_adjuster_on_val, \
        loss_epoch, metric_epoch, _, _ =on_one_epoch(
                epoch, train_loader, model, device, num_classes, bce_use, 
                learning_rate, optimizer, metric_used, \
                class_weights=class_weights,
                loss_fn=loss_fn,
                scheduler=scheduler, lr_adjuster_on_val=lr_adjuster_on_val,
                lr_div_factor=lr_div_factor, \
                survival=survival, one_cycle_epochs=one_cycle_epochs, \
                metric_from_whole=metric_from_whole, \
                train=True, inference_on_holdout=False,
                scaler=scaler, backward_every=backward_every, fp16=fp16)
        train_loss_list.append(loss_epoch)
        train_metric_list.append(metric_epoch)

        model, optimizer, scheduler, lr_adjuster_on_val, \
        loss_epoch, metric_epoch, _, _=on_one_epoch(
                    epoch, valid_loader, model, device, num_classes, bce_use, 
                    learning_rate, optimizer, metric_used, \
                    class_weights=class_weights,
                    loss_fn=loss_fn,
                    scheduler=scheduler, lr_adjuster_on_val=lr_adjuster_on_val,
                    lr_div_factor=lr_div_factor, \
                    survival=survival, one_cycle_epochs=one_cycle_epochs, \
                    metric_from_whole=metric_from_whole, \
                    train=False, inference_on_holdout=False,
                    scaler=scaler, backward_every=backward_every, fp16=fp16)
        val_loss_list.append(loss_epoch)
        val_metric_list.append(metric_epoch)
        
        print(f"Epoch {epoch+1}/{epochs}, AUC_train: {train_metric_list[-1]:.2f}, AUC_valid: {val_metric_list[-1]:.2f}")
        
        if save_every is not None:
            # SAVING
            save_progress(epoch,
                        save_every,
                        model,
                        optimizer,
                        path_performance,
                        path_performance_and_model,
                        train_loss_list,
                        train_metric_list,
                        val_loss_list,
                        val_metric_list)

    if test_loader is not None:
        test_loss_list=[]
        test_metric_list=[]
        model, optimizer, scheduler, lr_adjuster_on_val, \
        loss_epoch, metric_epoch, pred_test_list, target_test_list=on_one_epoch(
                        epoch, test_loader, model, device, num_classes, bce_use, 
                        learning_rate, optimizer, metric_used, \
                        class_weights=class_weights,
                        loss_fn=loss_fn, 
                        scheduler=scheduler, lr_adjuster_on_val=lr_adjuster_on_val, 
                        lr_div_factor=lr_div_factor, \
                        survival=survival, one_cycle_epochs=one_cycle_epochs, \
                        metric_from_whole=metric_from_whole, \
                        train=False, inference_on_holdout=True,
                        scaler=scaler, backward_every=backward_every, fp16=fp16)
        test_loss_list.append(loss_epoch)
        test_metric_list.append(metric_epoch)

        print(f"AUC_test: {test_metric_list[-1]:.2f}")
        return pred_test_list, target_test_list