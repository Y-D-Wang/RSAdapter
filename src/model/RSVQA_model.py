import torch
import numpy as np
import torchmetrics
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.optim.lr_scheduler import LambdaLR
from t.src.transformers.models.vilt.modeling_vilt_test import ViltModel

class VQAModel(pl.LightningModule):
    def __init__(self, batch_size=None, lr=None, number_outputs=None):
        super(VQAModel, self).__init__()

        self.save_hyperparameters()
        self.number_outputs = number_outputs
        self.loss = F.cross_entropy
        self.lr = lr
        self.batch_size = batch_size
        self.validation_step_outputs = []
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=number_outputs)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=number_outputs)
        self.results = {}
        self.res = []

        self.vilt = ViltModel.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

        for name, param in self.vilt.named_parameters():
            if 'adapter' not in name: 
                param.requires_grad = False
        
        self.classify_layer = torch.nn.Sequential(
            torch.nn.Linear(768, 1200),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1200, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, self.number_outputs)
        )  

    def forward(self, pixel_values, input_ids, token_type_ids, attention_mask):

        out = self.vilt(pixel_values=pixel_values, input_ids=input_ids, token_type_ids=token_type_ids,
                        attention_mask=attention_mask, output_attentions=True)

        out = torch.squeeze(out['last_hidden_state'][:, 0, :])
        out = self.classify_layer(out)

        return out

    def configure_optimizers(self):
        # configuration of the optimizer
        def rule(epoch):
            if self.number_outputs == 9:
                if epoch <= 3:
                    lamda = 1
                else:
                    lamda = 0.01
                return lamda 
            else:
                if epoch <= 2:
                    lamda = epoch + 1
                elif epoch <=10:
                    lamda = 0.2 
                else:
                    lamda = 0.2 * 0.2
                return lamda

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        scheduler = LambdaLR(optimizer, lr_lambda=rule)

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def training_step(self, batch, batch_idx):
        # performs the training steps
        pixel_values, input_ids, token_type_ids, attention_mask, answer = batch
        pred = self(pixel_values, input_ids, token_type_ids, attention_mask)

        self.train_acc(pred, answer)
        train_loss = self.loss(pred, answer)

        self.log("train_loss", train_loss, on_epoch=True, on_step=True, sync_dist=True, batch_size=self.batch_size)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=True, sync_dist=True, batch_size=self.batch_size)

        return train_loss

    def validation_step(self, batch, batch_idx):
        pixel_values, input_ids, token_type_ids, attention_mask, answer, question_type, img_id, question, answer_str = batch
        pred = self(pixel_values, input_ids, token_type_ids, attention_mask)

        self.valid_acc(pred, answer)
        valid_loss = self.loss(pred, answer)

        self.log("valid_loss", valid_loss, on_epoch=True, on_step=False, sync_dist=True, batch_size=self.batch_size)
        self.log("valid_acc", self.valid_acc, on_epoch=True, on_step=False, sync_dist=True, batch_size=self.batch_size)

        pred_arg = torch.argmax(pred, axis=1)
        for i in range(pred.shape[0]):
            if pred_arg[i] == answer[i]:
                self.validation_step_outputs.append([1, question_type[i]])
            else:
                self.validation_step_outputs.append([0, question_type[i]])

    def on_validation_epoch_end(self):
        outputs = np.stack(self.validation_step_outputs)

        total_rural_urban, total_presence, total_count, total_comp = 0, 0, 0, 0
        right_rural_urban, right_presence, right_count, right_comp = 0, 0, 0, 0
        acc_rural_urban, acc_presence, acc_count, acc_comp = 0, 0, 0, 0
        AA, OA, right, total = 0, 0, 0, 0
        
        for i in range(outputs.shape[0]):
            if outputs[i][1] == 'comp':
                total_comp += 1
                if outputs[i][0] == '1':
                    right_comp += 1
            elif outputs[i][1] == 'presence':
                total_presence += 1
                if outputs[i][0] == '1':
                    right_presence += 1
            elif outputs[i][1] == 'count':
                total_count += 1
                if outputs[i][0] == '1':
                    right_count += 1
            else:
                total_rural_urban += 1
                if outputs[i][0] == '1':
                    right_rural_urban += 1

        # Note that for RSVQA_HR, there's no 'rural_urban' question type 
        # so 'rural_urban' in RSVQA_HR represent for 'area' question type
        acc_rural_urban = right_rural_urban / total_rural_urban
        acc_presence = right_presence / total_presence
        acc_count = right_count / total_count
        acc_comp = right_comp / total_comp

        right = right_rural_urban + right_presence + right_count + right_comp
        total = total_rural_urban + total_presence + total_count + total_comp

        AA = (acc_rural_urban + acc_presence + acc_count + acc_comp) / 4
        OA = right / total

        self.log("acc_rural_urban", acc_rural_urban, sync_dist=True)
        self.log("acc_presence", acc_presence, sync_dist=True)
        self.log("acc_count", acc_count, sync_dist=True)
        self.log("acc_comp", acc_comp, sync_dist=True)
        # self.log("total", total, sync_dist=True)
        self.log('valid_AA', AA, sync_dist=True)
        self.log('valid_OA', OA, sync_dist=True)

        self.validation_step_outputs.clear()


