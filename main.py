import time


from Experiment import ExperimentManager


if __name__ == '__main__':
    ExperimentManager=ExperimentManager(config_path='./Experiment/experiment_config.json')
    ExperimentManager.run_all_experiments()