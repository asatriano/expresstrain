# Express Train
Pytorch training, testing and validation loops, half precision, and gradient accumulation. Made Easy.  

![alt text](https://github.com/as-deeplearning/expresstrain/blob/main/images/express_train_logo.png)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asatriano/expresstrain/blob/main/examples/expresstrain_fashion_mnist_example.ipynb)

More documentation to come soon.

For now, first import:

```Python3
import expresstrain as et
```

Then subclass and use as follows:

```Python3
class CustomExpressTrain(et.ExpressTrain):
    def __init__(self, **kwargs):
        super(CustomExpressTrain, self).__init__()
        self.initialize_all(kwargs)

    def on_train_epoch_begin(self):
            print(f"Message before epoch {self.epoch+1} - Today is a great day :)")

    def on_train_epoch_end(self):
        self.scheduler_every_epoch.step()

trainer_kwargs={'train_loader': train_loader,
                'valid_loader': valid_loader,
                'model': model,
                'num_classes': 10,
                'device': device,
                'learning_rate': learning_rate,
                'optimizer': optimizer,
                'scheduler_every_epoch': scheduler_every_epoch,
                'metric_used': accuracy,
                'path_performance': path_performance,
                'path_performance_and_model': path_perf_model,
                'backward_every': backward_every}
if use_fp16==True:
    print("Using Automatic Mixed Precision")
    trainer_kwargs.update({'fp16': use_fp16})

trainer=CustomExpressTrain(**trainer_kwargs)

trainer.fit(epochs=epochs)
```


That's it! 🚂

Open the Fashion MNIST example in Colab!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asatriano/expresstrain/blob/main/examples/expresstrain_fashion_mnist_example.ipynb)

Customize your model using any of the following hooks (more to come!):

```Python3
on_batch_begin(self)        
on_batch_end(self)      
on_epoch_begin(self)
on_epoch_end(self)
on_train_epoch_begin(self)        
on_train_epoch_end(self)        
on_valid_epoch_begin(self)        
on_valid_epoch_end(self)        
on_test_begin(self)       
on_test_end(self)
print_progress_on_epoch(self, metric_epoch)
print_on_train_epoch_end(self)   
print_on_valid_epoch_end(self)
print_on_test_epoch_end(self)
compute_output_and_loss(self, data, target)
compute_pred_from_output(self, output)
lr_adjust_on_val(self)
```

or changing any of the default attributes when you instance your custom trainer:

```Python3
self.valid_loader=None # validation_loader to use for internal validation
self.test_loader=None # test_loader to assess performance at inference on holdout

self.bce_use=False # whether we should use BinaryCrossEntropyLoss

self.device=torch.device('cpu') # pytorch device to use for analysis

self.loss_fn=None   # if not None, specify desired loss function
self.class_weights=None # class_weights according to pytorch convention

self.scheduler=None # scheduler according to pytorch convention
self.lr_adjuster_on_val=None # pytorch scheduler depending on validation results
self.lr_div_factor=None # if not None, this activates one_cycle traing and divides lr
self.one_cycle_epochs=None # how many epochs each one-cycle trainign cycle should last

self.metric_from_whole=True # should metric be computed by single batch of whole epoch

self.backward_every=1 # backward pass every N epochs (for gradient accumulation)
self.fp16=False # save memory at training (for Automatic Mixed Precision)
self.save_every=5 # saing loss, metric and model happens every specified epochs

self.path_performance=None # path where loss and metrics are saved
self.path_performance_and_model=None # path where loss, metrics, and model params are saved

self.use_progbar=True # input True to use progress bar
```

