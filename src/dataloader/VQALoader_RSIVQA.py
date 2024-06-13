import os.path
import json
import re
import glob
import tqdm
import mxnet as mx

from torch.utils.data import Dataset
from skimage import io
from tqdm import tqdm
from PIL import Image
import numpy as np
from PIL import ImageFile
from skimage.transform import resize


class VQALoader(Dataset):
    """
    This class manages the Dataloading.
    """

    def __init__(self, dataPath, datasets, tokenizer, image_processor, transform=None, sequence_length=40, type=None, list_ip=None):
        
        self.images_questions_answers = []
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.transform = transform
        self.list_ip = list_ip

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        for dataset in datasets:
            path = os.path.join(dataPath, dataset)
            text_file = glob.glob(os.path.join(path, '*.txt'))
            print(dataset)
            # UCM we have 2 txt files
            for text in text_file:
                with open(text, 'r') as f:
                    testData = f.readlines()

                for _, data in enumerate(tqdm(testData)):
                    question_type = None
                    split_strings = re.split('([:?])', data.strip())

                    if split_strings[-1] in ['yes', 'no']:
                        question_type = 'yes_no'
                    elif 'how many' in split_strings[2].lower():
                        if split_strings[4].isdigit():
                            question_type = 'number'
                        else:
                            continue
                    else:
                        question_type = 'others'

                    if dataset == 'HRRSD_new' :
                    # if dataset == 'HRRSD' :
                        # image = io.imread(os.path.join(path, split_strings[0] + '.tif'))
                        image = mx.img.imread(os.path.join(path, split_strings[0] + '.jpg'))
                    elif dataset == 'DOTA_val_new' or dataset == 'DOTA_train_new':
                    # elif dataset == 'DOTA_val' or dataset == 'DOTA_train':
                        # image = io.imread(os.path.join(path, split_strings[0] + '.tif'))
                        image = mx.img.imread(os.path.join(path, split_strings[0] + '.png'))
                    else:
                        # image = io.imread(os.path.join(path, split_strings[0]))
                        image = mx.img.imread(os.path.join(path, split_strings[0]))
                        # image = resize(image, (256,256))
                        image = mx.img.imresize(image, 256, 256)
                        # image = image.asnumpy()

                    answer = self.list_ip.index(split_strings[-1])
                    image = image.asnumpy()

                    if type is not None:
                        if question_type == type:

                            self.images_questions_answers.append([image, split_strings[2], answer, question_type, split_strings[0]])
                    else:
                        self.images_questions_answers.append([image, split_strings[2], answer, question_type])

        print("Done.")

    def __len__(self):
        # return the number of image-question-answer pairs, which are selected
        return len(self.images_questions_answers)

    def __getitem__(self, idx):
        # load the features of the index
        data = self.images_questions_answers[idx]

        language_feats = self.tokenizer(data[1], return_tensors='pt', padding='max_length',
                                        max_length=self.sequence_length)

        if self.transform is not None:
            img_unaugment = Image.fromarray(np.uint8(data[0])).convert('RGB')
            img_augment = self.transform(img_unaugment)
            imgT = self.image_processor(img_augment, return_tensors="pt")
            # img_augment = self.transform(data[0])
            # imgT = self.image_processor(img_augment, return_tensors="pt")
        else:
            imgT = self.image_processor(data[0], return_tensors="pt")

        return imgT['pixel_values'][0], language_feats['input_ids'][0], language_feats['token_type_ids'][0], language_feats['attention_mask'][0], data[2], data[3], data[-1]