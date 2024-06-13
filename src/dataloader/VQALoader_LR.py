import os.path
import json

from torch.utils.data import Dataset
from skimage import io
from tqdm import tqdm
from PIL import Image
import numpy as np


class VQALoader(Dataset):
    """
    This class manages the Dataloading.
    """

    def __init__(self,
                 imgFolder,
                 images_file,
                 questions_file,
                 answers_file,
                 tokenizer,
                 image_processor,
                 Dataset,
                 train=True,
                 ratio_images_to_use=0.1,
                 selected_answers=None,
                 sequence_length=40,
                 transform=None,
                 label=None):

        self.train = train
        self.imgFolder = imgFolder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        LR_number_outputs = 9
        HR_number_outputs = 98
        self.Dataset = Dataset
        self.transform = transform
        self.dict = {}

        # sequence length of the tokens
        self.sequence_length = sequence_length

        # loading the json files for the question, answers and images
        print("Loading JSONs...")
        with open(questions_file) as json_data:
            questionsJSON = json.load(json_data)

        with open(answers_file) as json_data:
            answersJSON = json.load(json_data)

        with open(images_file) as json_data:
            imagesJSON = json.load(json_data)
        print("Done.")

        # select only the active images
        images = [img['id'] for img in imagesJSON['images'] if img['active']]

        # select the requested amount of images
        # images = images[:int(len(images) * ratio_images_to_use)]
        self.img_ids = images
        if self.Dataset == 'LR':
            self.images = np.empty((len(images), 256, 256, 3))
        else:
            self.images = np.empty((len(images), 512, 512, 3))

        print("Construction of the Dataset")

        # when training we construct the answer set
        if train:

            # this dict will store as keys the answers and the values are the frequencies they occur
            self.freq_dict = {}

            for i, image in enumerate(tqdm(images)):

                # select the questionids, aligned to the image
                for questionid in imagesJSON['images'][image]['questions_ids']:

                    # select question with the id
                    question = questionsJSON['questions'][questionid]

                    # get the answer str with the answer id from the question
                    answer_str = answersJSON['answers'][question["answers_ids"][0]]['answer']

                    # group the counting answers
                    if self.Dataset == 'LR':
                        if answer_str.isdigit():
                            num = int(answer_str)
                            if num > 0 and num <= 10:
                                answer_str = "between 0 and 10"
                            if num > 10 and num <= 100:
                                answer_str = "between 10 and 100"
                            if num > 100 and num <= 1000:
                                answer_str = "between 100 and 1000"
                            if num > 1000:
                                answer_str = "more than 1000"
                    else:
                        if 'm2' in answer_str:
                            num = int(answer_str[:-2])
                            if num > 0 and num <= 10:
                                answer_str = "between 0m2 and 10m2"
                            if num > 10 and num <= 100:
                                answer_str = "between 10m2 and 100m2"
                            if num > 100 and num <= 1000:
                                answer_str = "between 100m2 and 1000m2"
                            if num > 1000:
                                answer_str = "more than 1000m2"

                    # update the dictionary
                    if answer_str not in self.freq_dict:
                        self.freq_dict[answer_str] = 1
                    else:
                        self.freq_dict[answer_str] += 1

            # sort the dictionary by the most common
            # so that position 0 contains the most frequent word
            self.freq_dict = sorted(self.freq_dict.items(), key=lambda x: x[1], reverse=True)

            self.selected_answers = []
            self.non_selected_answers = []

            # store the total number of selected answers
            coverage = 0

            # store the total number of answers
            total_answers = 0

            # this loop creates the list with selected answers
            # with respect to the number of outputs
            for i, key in enumerate(self.freq_dict):

                # select the answer if not have enough
                if self.Dataset == 'LR':
                    if i < LR_number_outputs:

                        # append the answer string
                        self.selected_answers.append(key[0])

                        # add the frequency of the answer
                        coverage += key[1]

                    # if we have enough answers we append them to non_selected_answers
                    else:
                        self.non_selected_answers.append(key[0])

                    # add all frequencies occuring
                    total_answers += key[1]
                else:
                    if i < HR_number_outputs:

                        # append the answer string
                        self.selected_answers.append(key[0])

                        # add the frequency of the answer
                        coverage += key[1]

                    # if we have enough answers we append them to non_selected_answers
                    else:
                        self.non_selected_answers.append(key[0])

                    # add all frequencies occuring
                    total_answers += key[1]
        # if we are not in training
        else:

            # if we are not in training we constructed already the answer set and provide them as input
            self.selected_answers = selected_answers

        # list for storing the image-question-answer pairs
        self.images_questions_answers = []

        # we go through all img ids
        for i, image in enumerate(tqdm(images)):

            img = io.imread(os.path.join(imgFolder, str(image) + '.tif'))
            self.images[i, :, :, :] = img

            # use img id to get the question id for corresponding to the img
            for questionid in imagesJSON['images'][image]['questions_ids']:

                # question id gives the dict of the question
                question = questionsJSON['questions'][questionid]

                # get the question str and the question type (e.g. Y/N)
                question_str = question["question"]
                type_str = question["type"]

                # get the answer str with the answer id from the question
                answer_str = answersJSON['answers'][question["answers_ids"][0]]['answer']

                # group the counting answers
                if self.Dataset == 'LR':
                    if answer_str.isdigit():
                        num = int(answer_str)
                        if num > 0 and num <= 10:
                            answer_str = "between 0 and 10"
                        if num > 10 and num <= 100:
                            answer_str = "between 10 and 100"
                        if num > 100 and num <= 1000:
                            answer_str = "between 100 and 1000"
                        if num > 1000:
                            answer_str = "more than 1000"
                else:
                    if 'm2' in answer_str:
                        num = int(answer_str[:-2])
                        if num > 0 and num <= 10:
                            answer_str = "between 0m2 and 10m2"
                        if num > 10 and num <= 100:
                            answer_str = "between 10m2 and 100m2"
                        if num > 100 and num <= 1000:
                            answer_str = "between 100m2 and 1000m2"
                        if num > 1000:
                            answer_str = "more than 1000m2"

                answer = self.selected_answers.index(answer_str)
                if label is not None:
                    if type_str == label:
                        self.images_questions_answers.append([question_str, answer, i, type_str, answer_str])
                else:
                    self.images_questions_answers.append([question_str, answer, i, type_str, answer_str])
            
        print("Done.")

    def __len__(self):
        # return the number of image-question-answer pairs, which are selected
        return len(self.images_questions_answers)

    def __getitem__(self, idx):
        # load the features of the index
        data = self.images_questions_answers[idx]

        language_feats = self.tokenizer(data[0], return_tensors='pt', padding='max_length',
                                        max_length=self.sequence_length)
        img = self.images[data[2], :, :, :]

        if self.transform is not None:
            img_unaugment = Image.fromarray(np.uint8(img)).convert('RGB')
            img_augment = self.transform(img_unaugment)
            imgT = self.image_processor(img_augment, return_tensors="pt")
        else:
            imgT = self.image_processor(img, return_tensors="pt")

        if self.train:
            return imgT['pixel_values'][0], language_feats['input_ids'][0], language_feats['token_type_ids'][0], language_feats['attention_mask'][0], data[1]
        else:
            return imgT['pixel_values'][0], language_feats['input_ids'][0], language_feats['token_type_ids'][0], language_feats['attention_mask'][0], data[1], \
                   data[3], data[2], data[0], data[-1]