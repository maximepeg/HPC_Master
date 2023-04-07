import pytorch_lightning as pl
from yaml import full_load
from project.data import SquadData
from project.squadmodel import SquadModule

if __name__ == '__main__':
    config = full_load(open('config.yaml'))

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






    data = SquadData(model_name, dataset_name, batch_size, num_workers)
    data.prepare_data()
    data.setup()

    model = SquadModule(model_name, lr)

    trainer = pl.Trainer(accelerator=accelerator,
                         gpus=devices,
                         precision=precision,
                         strategy=strategy,
                         max_epochs=num_epochs)

    trainer.fit(model, data)

    trainer.test(model, data)

