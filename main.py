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
    torch.set_float32_matmul_precision('medium' )
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config = full_load(open('config.yaml'))

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
    for param in model.parameters():
        param.requires_grad = True

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
        logger.experiment.name = logger.experiment.name + f"-{devices}_GPU-{num_nodes}_NODE-{model_nickname}_{strategy}"


    trainer.fit(model, data)
    # trainer.validate(model, data)

