import os
import re
import random
import torch
random.seed(42)
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, [1]))
torch.manual_seed(42)
import typer
import glob
from tqdm import tqdm

import pytorch_lightning as pl

from torch.utils.data import random_split
from transformers import BertTokenizerFast, ViltImageProcessor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from model.RSIVQA_model import VQAModel
from dataloader.VQALoader_RSIVQA import VQALoader
from torch.utils.data import ConcatDataset


def main(num_workers: int = 12,
         sequence_length: int = 40,
         num_epochs: int = 50,
         batch_size: int = 64,
         lr: float = 1e-3,):

    data_path = '/YOUR/PATH'
    datasets = ['UCM', 'DOTA_train_new', 'AID', 'DOTA_val_new', 'Sydney', 'HRRSD_new',]

    tokenizer = BertTokenizerFast.from_pretrained('dandelin/vilt-b32-mlm')
    image_processor = ViltImageProcessor(do_resize=True, image_std=[0.229, 0.224, 0.225], image_mean=[0.485, 0.456, 0.406], 
                                         do_rescale=True, do_normalize=True, size=256, size_divisor=32)
    
    model = VQAModel(batch_size=batch_size, lr=lr, number_outputs=519)

    # loader for the training data
    list_ip = get_answer(data_path, datasets)

    dataset_yes = VQALoader(data_path,
                        datasets=datasets,
                        tokenizer=tokenizer,
                        image_processor=image_processor,
                        sequence_length=sequence_length,
                        type='yes_no',
                        list_ip=list_ip,)
    
    dataset_num = VQALoader(data_path,
                        datasets=datasets,
                        tokenizer=tokenizer,
                        image_processor=image_processor,
                        sequence_length=sequence_length,
                        type='number',
                        list_ip=list_ip,)
    
    dataset_other = VQALoader(data_path,
                        datasets=datasets,
                        tokenizer=tokenizer,
                        image_processor=image_processor,
                        sequence_length=sequence_length,
                        type='others',
                        list_ip=list_ip,)

    generator = torch.Generator().manual_seed(42)
    a = 0.8
    b = 0.9 - a
    train_yes, val_yes, test_yes = random_split(dataset=dataset_yes, lengths=[a, b, 0.1], generator=generator)
    train_num, val_num, test_num = random_split(dataset=dataset_num, lengths=[a, b, 0.1], generator=generator)
    train_other, val_other, test_other = random_split(dataset=dataset_other, lengths=[a, b, 0.1], generator=generator)
    
    train_dataset = ConcatDataset([train_yes, train_num, train_other,])
    valid_dataset = ConcatDataset([val_yes, val_num, val_other,])
    test_dataset = ConcatDataset([test_yes, test_num, test_other,])
 
    RSI_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                  num_workers=num_workers)

    # loader for the validation data
    RSI_val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    wandb_logger = WandbLogger(project='RSIVQA_adapter')

    # specify how to checkpoint
    checkpoint_callback = ModelCheckpoint(save_top_k=1,
                                          monitor="valid_acc",
                                          save_weights_only=True,
                                          mode="max",
                                          dirpath='/YOUR/PATH',
                                          filename=f"{{epoch}}_{{valid_acc:.5f}}")

    # early stopping
    early_stopping = EarlyStopping(monitor="valid_acc", patience=20, mode="max")

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(devices=1, 
                         accelerator='cuda',
                         fast_dev_run=False,
                         precision='16-mixed',
                         max_epochs=num_epochs,
                         logger=wandb_logger,
                         #strategy='ddp_find_unused_parameters_true',
                         num_sanity_val_steps=0,
                         callbacks=[checkpoint_callback, early_stopping, lr_monitor])

    trainer.fit(model, train_dataloaders=RSI_train_loader, val_dataloaders=RSI_val_loader)


def get_answer(data_path, dataset):
    # yes_no, number, others = 0, 0, 0
    answers = []

    for data in dataset:
        path = os.path.join(data_path, data)

        text_file = glob.glob(os.path.join(path, '*.txt'))
        print(data)

        # UCM we have 2 txt files
        for text in text_file:
            with open(text, 'r') as f:
                testData = f.readlines()

            for _, data in enumerate(tqdm(testData)):
                split_strings = re.split('([:?])', data.strip())

                if 'How many' in split_strings[2]:
                    if split_strings[4].isdigit():
                        answers.append(split_strings[-1])
                else:
                    answers.append(split_strings[-1])
    
    count_dist = {}

    for i in answers:
        if i in count_dist:
            count_dist[i] += 1
        else:
            count_dist[i] = 1
    
    list_ip = []

    for j in range(len(count_dist)):
        list_ip = list(count_dist.keys())

    return list_ip


if __name__ == "__main__":
    typer.run(main)