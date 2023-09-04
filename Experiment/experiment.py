import json
import os
import time

import torch
import torch.optim as optim
from Datasets import get_dataset
from Models import get_model
from TrainingScripts.TrainingBase import BaseTrainer
from Utils import TrainingMonitor, get_optimizer,Criterion,CustomDataParallel,get_scheduler

class ExperimentManager:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.experiments = json.load(f)

    def _prepare_paths(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        return {
            "model": os.path.join(folder, "model.pth"),
            "plot": os.path.join(folder, "training_plot.png"),
            "stats": os.path.join(folder, "training_stats.csv"),
        }

    def run_experiment(self, experiment_name):
        config = self.experiments[experiment_name]
        print("\n".join(f"{key}: {config[key]}" for key in config))
        paths = self._prepare_paths(config["folder"])
        device = torch.device(f"cuda:{config['gpus'][0]}")

        # Load data

        trainloader,testloader,num_classes,normalize_layer=get_dataset(dataset_name=config["dataset"],batch_size=config["batch_size"], num_workers=config["num_worker"],device=device)

        # Initialize model

        model = get_model(model_name=config["model"], pretrained=False, num_classes=num_classes,normalize=normalize_layer,dataset=config["dataset"])

        #DataParallel
        if len(config["gpus"])>1 and torch.cuda.is_available():
            print(f"Using DataParallel for gpus: ", config["gpus"])
            model = CustomDataParallel(model, device_ids=config.get("gpus"))  # Use GPU 0 and 2

        model.to(device)

        # Load existing weights if specified
        if config.get("load_weights", False):
            model.load_state_dict(torch.load(paths["model"]))
            print("Weights loaded")

        # Set optimizer
        optimizer = get_optimizer(name=config["optimizer"],
                                  parameters=model.parameters(),
                                  lr=config["learning_rate"],
                                  momentum=config["momentum"],
                                  weight_decay=config["weight_decay"]
                                  )
        scheduler = get_scheduler(optimizer=optimizer,scheduler_type=config["scheduler_name"],T_max=config["epochs"])

        # Training
        monitor = TrainingMonitor(stats_save_path=paths['stats'], plot_save_path=paths['plot'])


        trainer = BaseTrainer(
            model=model, optimizer=optimizer,
            train_loader=trainloader, val_loader=testloader,
            monitor=monitor, device=device,
            criterion=Criterion.get_criterion(config.get("criterion", "crossentropy")),
            scheduler=scheduler
        )

        trainer.run(num_epochs=config["epochs"])

        # Save results
        torch.save(model.state_dict(), paths["model"])

    def run_all_experiments(self):
        for experiment_name in self.experiments:
            print(f"Running experiment: {experiment_name}")
            self.run_experiment(experiment_name)
            print(f"Completed experiment: {experiment_name}")
