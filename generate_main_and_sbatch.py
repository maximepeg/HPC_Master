# generate main.py and sbatch.sh

import os
import sys
import argparse
import yaml
from itertools import product

def generate_main(config):
    main_file_content = f"""
import pytorch_lightning as pl
import torch
from yaml import full_load
from project.data import SquadData
from project.squadmodel import SquadModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import os
        
if __name__ == '__main__':
    
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    config = full_load(open('config.yaml'))
    config.update({config})
    logger = WandbLogger()
    dataset_name = config.get('dataset_name', 'squad')
    num_workers = config.get('num_workers', 4)
    
    model_name = config.get('model_name', 'bert-base-uncased')
    batch_size = config.get('batch_size', 16)
    lr = float(config.get('lr', 1e-5))
    num_epochs = config.get('num_epochs', 3)
    
    devices = config.get('devices', 1)
    precision = config.get('precision', 32)
    accelerator = config.get('accelerator', 'cpu')
    strategy = config.get('strategy', 'ddp')
    enable_checkpoint = config.get('checkpoint', False)
    num_nodes = config.get('num_nodes', 1)
    model_nickname = model_name.split("-")[0]
    data = SquadData(model_name, dataset_name, batch_size, num_workers)
    data.prepare_data()
    data.setup()
    steps_per_epoch = len(data.train_data)
    model = SquadModule(model_name, lr, steps_per_epoch, num_epochs)
    callbacks = [LearningRateMonitor(logging_interval='step')]
    trainer = pl.Trainer(accelerator=accelerator,
                         devices=devices,
                         precision=precision,
                         strategy=strategy,
                         max_epochs=num_epochs,
                         logger=logger,
                         enable_checkpointing=enable_checkpoint,
                         callbacks=callbacks,
                         num_nodes=num_nodes)
    
    
    if trainer.global_rank == 0:
        logger.experiment.config.update(config)
        logger.experiment.name = logger.experiment.name + f"-devices_GPU-num_nodes_NODE-model_nickname_strategy"
    
    
    trainer.fit(model, data)
    # trainer.validate(model, data)
        """
    return main_file_content

def generate_configs(grid_config):
    keys, values = zip(*grid_config.items())
    configs = [dict(zip(keys, v)) for v in product(*values)]
    return configs

def generate_sbatch(filename, config):
    sbatch_content = f"""#!/bin/bash

#SBATCH -N {config['nodes']}             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks-per-node=2   # This needs to match Trainer(devices=...)
#SBATCH --mem=8G
#SBATCH -c 32
#SBATCH --time=0-02:00:00

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

#module load cuda

conda activate myenv
srun  python {filename}"""
    return sbatch_content








grid_config = {
    "model_name": ["bert-base-uncased", "distilbert-base-uncased"],
    "nodes": [1, 2, 3, 4, 5],
    "strategy": ["ddp", "ddp2", "ddp_spawn", "ddp_sharded", "dp", "fsdp", "horovod"],

}

configs = generate_configs(grid_config)
for i,config in enumerate(configs):
    # print(config)
    main_file_content = generate_main(config)
    with open(f"main{i}.py", "w") as f:
        f.write(main_file_content)
    sbatch_content = generate_sbatch(f"main{i}.py", config)
    with open(f"sbatch{i}.sh", "w") as f:
        f.write(sbatch_content)

#%%
