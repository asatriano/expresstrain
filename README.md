# Express Train
Pytorch training, testing and validation loops, half precision, and gradient accumulation. Made Easy.  

![alt text](https://github.com/as-deeplearning/expresstrain/blob/main/images/express_train_logo_20210322.png)

More documentation to come soon.

For now, first import:

```Python3
from expresstrain.modular import ExpressTrain
```

Then subclass and use as follows:

```Python3
class CustomExpressTrain(ExpressTrain):
    def __init__(self, **kwargs):
        super(CustomExpressTrain, self).__init__()
        self.initialize_all(kwargs)

    def on_train_epoch_start(self):
        print(f"pre-train message: You're doing great: epoch {self.epoch}")

trainer=CustomExpressTrain(train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
                        model=model, device=device, num_classes=num_classes,  
                        learning_rate=learning_rate, optimizer=optimizer, metric_used=metric_used, 
                        bce_use=bce_use, loss_fn=None,
                        class_weights=class_weights,
                        scheduler=scheduler, lr_adjuster_on_val=lr_adjuster, lr_div_factor=lr_div_factor, 
                        survival=survival, one_cycle_epochs=one_cycle_epochs, 
                        metric_from_whole=metric_from_whole, 
                        backward_every=backward_every, fp16=fp16,
                        save_every=save_every,
                        path_performance=path_performance,
                        path_performance_and_model=path_performance_and_model)

trainer.fit(epochs=epochs)
```

That's it :)
