import time

import matplotlib.pyplot as plt
import csv
from tqdm import tqdm


class TrainingMonitor:
    def __init__(self, stats_save_path="training_stats.csv", plot_save_path="training_plot.png"):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.stats_save_path = stats_save_path
        self.plot_save_path = plot_save_path
        self.progress_bar=None
        self.epoch=0
        self.num_epochs=0

        # Initialize the CSV file with headers
        with open(self.stats_save_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Training Loss", "Validation Loss", "Training Accuracy", "Validation Accuracy"])

    def update(self, epoch, train_loss, val_loss, train_acc, val_acc):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)

        # Save the stats for the current epoch to the CSV file
        with open(self.stats_save_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, train_loss, val_loss, train_acc, val_acc])

    def display_stats(self):
        tqdm.write("")
        tqdm.write(f"Summary of Epoch {self.epoch}: Training Loss: {self.train_losses[-1]:.4f}, Training Acc: {self.train_accuracies[-1]:.2f}, Val Loss: {self.val_losses[-1]:.4f}, Val Acc: {self.val_accuracies[-1]:.2f}")
        tqdm.write("")

    def plot_progress(self):
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))

        # Plot loss
        ax[0].plot(self.train_losses, label='Training Loss', color='blue')
        ax[0].plot(self.val_losses, label='Validation Loss', color='red')
        ax[0].set_ylabel('Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].legend()
        ax[0].grid(True)

        # Plot accuracy
        ax[1].plot(self.train_accuracies, label='Training Accuracy', color='blue')
        ax[1].plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax[1].set_ylabel('Accuracy')
        ax[1].set_xlabel('Epoch')
        ax[1].legend()
        ax[1].grid(True)

        plt.tight_layout()

        # Save the plot to a file instead of displaying
        plt.savefig(self.plot_save_path)
        plt.close()

    def track_process(self,dataloader):
        if self.progress_bar:
            self.progress_bar.reset()
        else:
            self.progress_bar= tqdm(dataloader, desc="Training", dynamic_ncols=True)
        # time.sleep(1)
        self.progress_idx=0
    def display_process(self,loss,acc):
        if self.progress_idx%10==0:
            self.progress_bar.set_description(f"Epoch: {self.epoch}/{self.num_epochs}, Train Loss: {loss:.4f}, Train Acc: {acc:.4f}")
            self.progress_bar.update(10)
        self.progress_idx += 1
