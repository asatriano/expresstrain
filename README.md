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

    def on_train_epoch_start(self):
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
on_batch_start(self)        
on_batch_end(self)      
on_epoch_start(self)
on_epoch_end(self)
on_train_epoch_start(self)        
on_train_epoch_end(self)        
on_valid_epoch_start(self)        
on_valid_epoch_end(self)        
on_test_start(self)       
on_test_end(self)    
```

or changing any of the default attributes :)
