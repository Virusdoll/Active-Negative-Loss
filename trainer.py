import torch
from utils import batch_accuracy

# keys for batch info
EPOCH = 'epoch'
STEPS = 'steps'
LR = 'batch_lr'
ACC = 'batch_acc'
LOSS = 'batch_loss'

class Trainer():
    def __init__(self, train_dataloader, logger, writer,
                 device, is_anl, grad_bound=5.0):
        # setting trainer
        self.device = device
        self.train_dataloader = train_dataloader
        self.grad_bound = grad_bound
        self.logger = logger
        self.writer = writer
        self.is_anl = is_anl
        
        # setup log info
        self.setup_info()

    def train(self, model, optimizer, loss_function, epoch):
        model.train()
        self.info[EPOCH] = epoch

        for images, labels in self.train_dataloader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            logits, loss = self.train_batch(model, optimizer, loss_function,
                                            images, labels)
            
            self.update_info(optimizer, logits, labels, loss)
            self.log()
            self.write()
        
    def train_batch(self, model, optimizer, loss_function, images, labels):
        model.zero_grad()
        optimizer.zero_grad()
       
        logits = model(images)
        loss = loss_function(logits, labels, model) \
               if self.is_anl \
               else loss_function(logits, labels)
    
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_bound)
        optimizer.step()
        
        return logits, loss

    def setup_info(self):
        # dict for batch info
        self.info = {}
        self.info[EPOCH] = 0
        self.info[STEPS] = 0
        self.info[LR] = 0
        self.info[ACC] = 0
        self.info[LOSS] = 0

    def update_info(self, optimizer, logits, labels, loss):
        self.info[STEPS] += 1
        self.info[LR] = optimizer.param_groups[0]['lr']
        self.info[ACC] = batch_accuracy(logits, labels)
        self.info[LOSS] = loss.item() \
                          if not isinstance(loss, int) \
                          else loss

    def log(self):
        self.logger.info(self.info)

    def write(self):
        if self.writer is None:
            return

        self.writer.add_scalar('Learning_Rate/Train', self.info[LR], self.info[STEPS])
        self.writer.add_scalar('Accuracy/Train', self.info[ACC], self.info[STEPS])
        self.writer.add_scalar('Loss/Train', self.info[LOSS], self.info[STEPS])
