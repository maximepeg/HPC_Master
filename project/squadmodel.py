import pytorch_lightning as pl
import torch
import transformers


class SquadModel(pl.LightningModule):
    def __init__(self, model_name):
        super().__init__()
        self.output = None
        self.num_labels = None
        self.model_config = None
        self.model = None
        self.model_name = model_name
        self.init_model(model_name)

    def init_model(self, model_name):
        self.model = transformers.AutoModel.from_pretrained(model_name)
        self.model_config = transformers.AutoConfig.from_pretrained(model_name)
        self.num_labels = self.model_config.num_labels
        self.output = torch.nn.Linear(self.model_config.hidden_size, self.num_labels)

    def forward(self, batch):
        outputs = self.model(input_ids=batch['input_ids'],
                             attention_mask=batch['attention_mask'])

        logits = self.output(outputs[0])
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return {'start_logits': start_logits, 'end_logits': end_logits}


class SquadModule(pl.LightningModule):
    def __init__(self, model_name, lr, steps_per_epoch, max_epochs=1):
        super().__init__()
        self.hparams["learning_rate"] = lr
        self.hparams["steps_per_epoch"] = steps_per_epoch
        self.hparams["model_name"] = model_name
        self.hparams["max_epochs"] = max_epochs
        self.steps_per_epoch = steps_per_epoch
        self.save_hyperparameters()
        self.model = SquadModel(model_name)

    @staticmethod
    def compute_loss(start_positions, end_positions, start_logits, end_logits):
        total_loss = None
        if start_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_function = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_function(start_logits, start_positions)
            end_loss = loss_function(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return total_loss

    def common_step(self, batch, phase):
        start_positions = batch['start_positions']
        end_positions = batch['end_positions']

        output = self.model(batch)
        start_logits = output['start_logits']
        end_logits = output['end_logits']
        # compute precision
        precision = torch.nn.functional.softmax(start_logits, dim=1)
        precision = torch.argmax(precision, dim=1)
        precision = torch.sum(precision == start_positions) / len(start_positions)
        self.log(f'{phase}_precision', precision, sync_dist=True)

        total_loss = self.compute_loss(start_positions, end_positions, start_logits, end_logits)

        return total_loss

    def training_step(self, batch):

        total_loss = self.common_step(batch, "training")
        self.log('train_loss', total_loss, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_nf):
        total_loss = self.common_step(batch, "validation")
        self.log('val_loss', total_loss, sync_dist=True)
        return total_loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        output = self.model(batch)
        start_logits = output['start_logits']
        end_logits = output['end_logits']

        return {'start_logits': start_logits, 'end_logits': end_logits}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.hparams.learning_rate,
                                     eps=1e-08)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                             max_lr=self.hparams["learning_rate"],
                                                             steps_per_epoch=self.steps_per_epoch,
                                                             epochs=self.hparams["max_epochs"]),
            'interval': 'step'  # called after each training step
        }
        return [optimizer], [scheduler]
