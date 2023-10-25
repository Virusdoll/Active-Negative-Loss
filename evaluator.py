import torch
from utils import AverageMeter, batch_accuracy

class Evaluator():
    def __init__(self, data_loader, logger, writer, device, is_anl) -> None:
        self.device = device
        self.data_loader = data_loader
        self.logger = logger
        self.loss_meters = AverageMeter()
        self.acc_meters = AverageMeter()
        self.writer = writer
        self.is_anl = is_anl

        self.best_acc = -1
        self.best_epoch = -1

    @torch.no_grad()
    def eval(self, model, loss_function, epoch):
        model.eval()
        for images, labels in self.data_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            pred, loss = self.eval_batch(model, images, labels, loss_function)
            self.update_meters(pred, labels, loss)
        self.update_best(epoch)
        self.log(epoch)
        self.write(epoch)
        self.reset_meters()
    
    def eval_batch(self, model, images, labels, loss_function):
        logits = model(images)
        loss = loss_function(logits, labels, model) \
               if self.is_anl \
               else loss_function(logits, labels)
        return logits, loss
    
    def update_meters(self, pred, labels, loss):
        batch_acc = batch_accuracy(pred, labels)
        self.loss_meters.update(loss.item(), labels.shape[0])
        self.acc_meters.update(batch_acc, labels.shape[0])

    def reset_meters(self):
        self.loss_meters.reset()
        self.acc_meters.reset()

    def log(self, epoch):
        display = {'epoch': epoch,
                   'eval_acc': self.acc_meters.avg,
                   'eval_loss': self.loss_meters.avg,
                   'best_acc': self.best_acc,
                   'best_epoch': self.best_epoch}
        self.logger.info(display)
    
    def write(self, epoch):
        if self.writer is None:
            return

        self.writer.add_scalar('Accuracy/Eval', self.acc_meters.avg, epoch)
        self.writer.add_scalar('Loss/Eval', self.loss_meters.avg, epoch)
    
    def update_best(self, epoch):
        if self.acc_meters.avg >= self.best_acc:
            self.best_acc = self.acc_meters.avg
            self.best_epoch = epoch
