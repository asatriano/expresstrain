# %% Imports
import torch
from torch.utils.tensorboard import SummaryWriter

# %% Generic logger class
class ExpressLogger(object):
    def __init__(self, **kwargs):
        self.initialize_all(kwargs)

    def initialize_all(self, kwargs):
        for key in kwargs: #all other parameyters are converted into attributes
            setattr(self, key, kwargs[key])
    
    def initialize_writer(self):
        return NotImplemented

    def log_one_var(self, phase, epoch, var_name, var_value):
        return NotImplemented
    
    def write_data(self, **kwargs):
        return NotImplemented

    def flush(self):
        return NotImplemented

    def close(self):
        return NotImplemented

    def check_and_log(self, phase, epoch, var_name, var_value):
        if isinstance(var_name, list):
            assert(isinstance(var_value, list))
            assert(len(var_name)==len(var_value))
            for var_idx, var in enumerate(var_name):
                self.log_one_var(phase, epoch,
                    var, var_value[var_idx])
        else:
            assert(isinstance(var_value, list)==False)
            self.log_one_var(phase, epoch,
                var_name, var_value)

# %%
class ExpressTensorBoard(ExpressLogger):
    def __init__(self, log_dir=None, comment='', purge_step=None,
                max_queue=10, flush_secs=120, filename_suffix='', **kwargs):
        super(ExpressTensorBoard, self).__init__()
        self.initialize_all(kwargs)

        self.writer=torch.utils.tensorboard.SummaryWriter(
            log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix)
        
    def log_one_var(phase, epoch, var_name, var_value)
        self.writer.add_scalar(f"{var_name}/{phase}",
                                var_value, epoch,
                                )

    def write_data(self, phase, epoch, loss_name=None, loss_value=None, metric_name=None, metric_value=None):
        self.check_and_log(phase, epoch, loss_name, loss_value)
        self.check_and_log(phase, epoch, metric_name, metric_value)

    def flush(self):
        self.writer.flush

    def close(self):
        self.writer.close()

# %%
