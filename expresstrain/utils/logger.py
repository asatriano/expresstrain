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
    
    def write_data(self, **kwargs):
        return NotImplemented

    def flush(self):
        return NotImplemented

    def close(self):
        return NotImplemented

# %%
class ExpressTensorBoard(ExpressLogger):
    def __init__(self, log_dir=None, comment='', purge_step=None,
                max_queue=10, flush_secs=120, filename_suffix='', **kwargs):
        super(ExpressTensorBoard, self).__init__()
        self.initialize_all(kwargs)

        self.writer=torch.utils.tensorboard.SummaryWriter(
            log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix)

    




# - generate logsession -> self.writer
# - Save to logsession(self, phase)
# - Flush logsession
# - Close logsession

# Class expresstensor(expresslog):
# — superinit
# -subclass the rest
