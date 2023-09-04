import torch
from Metrics import accuracy
import time
from tqdm import tqdm

class BaseTrainer:
    def __init__(self, model, optimizer, train_loader, val_loader, monitor,criterion, scheduler, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.monitor = monitor
        self.device = device
        self.model.to(self.device)
        self.criterion=criterion
        self.scheduler=scheduler

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_corrects = 0

        self.monitor.track_process(self.train_loader)

        for batch in self.train_loader:
            inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            ave_acc=accuracy(outputs, labels)
            total_corrects += ave_acc * inputs.size(0)
            self.monitor.display_process(loss.data.cpu(),ave_acc)

        epoch_loss = total_loss / len(self.train_loader.dataset)
        epoch_acc = total_corrects / len(self.train_loader.dataset)

        return epoch_loss, epoch_acc

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total_corrects = 0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * inputs.size(0)
                total_corrects += accuracy(outputs, labels) * inputs.size(0)

        epoch_loss = total_loss / len(self.val_loader.dataset)
        epoch_acc = total_corrects / len(self.val_loader.dataset)

        return epoch_loss, epoch_acc

    def run(self, num_epochs):
        for epoch in range(num_epochs):
            self.monitor.num_epochs=num_epochs
            self.monitor.epoch=epoch
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()
            self.scheduler.step()
            self.monitor.update(epoch, train_loss, val_loss, train_acc, val_acc)
            self.monitor.display_stats()
            self.monitor.plot_progress()
