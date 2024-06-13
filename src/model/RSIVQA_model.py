import torch
import numpy as np
import math
import torchmetrics
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.optim.lr_scheduler import LambdaLR
from t.src.transformers.models.vilt.modeling_vilt_test import ViltModel

class VQAModel(pl.LightningModule):
    def __init__(self, batch_size, lr=None, number_outputs=None):
        super(VQAModel, self).__init__()

        self.save_hyperparameters()
        self.number_outputs = number_outputs
        self.loss = F.cross_entropy
        self.lr = lr
        self.batch_size = batch_size
        self.validation_step_outputs = []
        self.results = {}
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=number_outputs)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=number_outputs)

        self.vilt = ViltModel.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        
        for name, param in self.vilt.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False

        self.classify_layer = torch.nn.Sequential(
            torch.nn.Linear(768, 1200),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1200, 768),
            torch.nn.Tanh(),
            torch.nn.Linear(768, self.number_outputs)
        )

    def forward(self, pixel_values, input_ids, token_type_ids, attention_mask):

        out = self.vilt(pixel_values=pixel_values, input_ids=input_ids, token_type_ids=token_type_ids,
                        attention_mask=attention_mask, output_hidden_states=True, output_attentions=True)

        attention = out['attentions'][-1]
        out = self.classify_layer(torch.squeeze(out['last_hidden_state'][:, 0, :]))

        return out

    def configure_optimizers(self):
        # configuration of the optimizer
        def rule(epoch):
            if epoch <= 3:
                lamda = 1
            else:
                lamda = 0.01
            return lamda

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        scheduler = LambdaLR(optimizer, lr_lambda=rule)

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def training_step(self, batch, batch_idx):
        # performs the training steps
        pixel_values, input_ids, token_type_ids, attention_mask, answer, question_type, image = batch
        #pixel_values = pixel_values[torch.randperm(pixel_values.size(0))]
        pred = self(pixel_values, input_ids, token_type_ids, attention_mask)
        # answer = torch.tensor(answer)

        self.train_acc(pred, answer)
        train_loss = self.loss(pred, answer)

        self.log("train_loss", train_loss, on_epoch=True, on_step=True, sync_dist=True, batch_size=self.batch_size)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=True, sync_dist=True, batch_size=self.batch_size)

        return train_loss

    def validation_step(self, batch, batch_idx):
        pixel_values, input_ids, token_type_ids, attention_mask, answer, question_type, image = batch
        pixel_values = pixel_values[torch.randperm(pixel_values.size(0))]
        pred = self(pixel_values, input_ids, token_type_ids, attention_mask)
        # answer = torch.tensor(answer)
        #print(image)

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
        
        # print(answer, self.validation_step_outputs, image)

    def on_validation_epoch_end(self):
        outputs = np.stack(self.validation_step_outputs)

        total_yes_no, total_num, total_other = 0, 0, 0
        right_yes_no, right_num, right_other = 0, 0, 0
        acc_yes_no, acc_num, acc_other = 0, 0, 0
        
        for i in range(outputs.shape[0]):
            if outputs[i][1] == 'yes_no':
                total_yes_no += 1
                if outputs[i][0] == '1':
                    right_yes_no += 1
            elif outputs[i][1] == 'number':
                total_num += 1
                if outputs[i][0] == '1':
                    right_num += 1
            else:
                total_other += 1
                if outputs[i][0] == '1':
                    right_other += 1
        
        acc_yes_no = right_yes_no / total_yes_no
        acc_num = right_num / total_num
        acc_other = right_other / total_other

        self.log("acc_yes_no", acc_yes_no, sync_dist=True)
        self.log("acc_num", acc_num, sync_dist=True)
        self.log("acc_other", acc_other, sync_dist=True)

        self.validation_step_outputs.clear()