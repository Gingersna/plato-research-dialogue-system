"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project.

See the License for the specific language governing permissions and
limitations under the License.
"""

__author__ = "jiang"


#from ludwig.api import LudwigModel
from plato.agent.component.conversational_module import ConversationalModule
import os.path
import pandas as pd

import os
import zipfile
import json
import torch
import sys

from convlab2.util.file_util import cached_path

from .crosswoz.dataloader import Dataloader
from .crosswoz.jointBERT import JointBERT
from .crosswoz.postprocess import recover_intent
from .crosswoz.preprocess import preprocess

"""
CrossWOZNLU is an CrossWOZ class that defines an interface to PyTorch models.
"""


class CrossWOZNLU(ConversationalModule):
    def __init__(self, args):
        """
        Load the Ludwig MetalWOZ model.

        :param args: a dictionary containing the path to the model.
        """
        super(CrossWOZNLU, self).__init__()


        if 'mode' in args:
            mode = args['mode']
        assert mode == 'usr' or mode == 'sys' or mode == 'all'

        if 'config_file' in args:
            config_file = args['config_file']

        model_file = None
        if 'model_file' in args:
            model_file = args['model_file']

        self.model = None



        config = json.load(open(config_file))
        DEVICE = config['DEVICE']
        data_dir = config['data_dir']
        print(data_dir)

        # if not os.path.exists(os.path.join(data_dir, 'intent_vocab.json')):
        #     print("不存在意图标签")
        #     preprocess(mode)
        print("载入意图标签")
        intent_vocab = json.load(open(os.path.join(data_dir, 'intent_vocab.json')))

        tag_vocab = json.load(open(os.path.join(data_dir, 'tag_vocab.json')))
        dataloader = Dataloader(intent_vocab=intent_vocab, tag_vocab=tag_vocab,
                                pretrained_weights=config['model']['pretrained_weights'])

        print('intent num:', len(intent_vocab))
        print('tag num:', len(tag_vocab))

        best_model_path = model_file
        # if not os.path.exists(best_model_path):
        #     if not os.path.exists(output_dir):
        #         os.makedirs(output_dir)
        #     print('Load from model_file param')
        #     archive_file = cached_path(model_file)
        #     archive = zipfile.ZipFile(archive_file, 'r')
        #     archive.extractall(root_dir)
        #     archive.close()
        print('Load from', best_model_path)
        model = JointBERT(config['model'], DEVICE, dataloader.tag_dim, dataloader.intent_dim)
        model.load_state_dict(torch.load(os.path.join(best_model_path, 'pytorch_model.bin'), DEVICE))
        model.to(DEVICE)
        model.eval()

        self.model = model
        self.dataloader = dataloader
        print("BERTNLU loaded")



    def __del__(self):
        """
        Close the Ludwig model.

        :return:
        """

        if self.model:
            self.model.close()

    def initialize(self, args):
        """
        Nothing to do here.

        :param args:
        :return:
        """
        pass

    def generate_output(self, args=None):
        """
        Generate output, used by the Generic Agent.

        :param args:
        :return:
        """
        if not args:
            print('WARNING! MetalWOZ called without arguments!')
            return ''

        utterance = args['args']
        context = []

        ori_word_seq = self.dataloader.tokenizer.tokenize(utterance)
        ori_tag_seq = ['O'] * len(ori_word_seq)
        context_seq = self.dataloader.tokenizer.encode('[CLS] ' + ' [SEP] '.join(context[-3:]))
        intents = []
        da = {}

        word_seq, tag_seq, new2ori = ori_word_seq, ori_tag_seq, None
        batch_data = [[ori_word_seq, ori_tag_seq, intents, da, context_seq,
                       new2ori, word_seq, self.dataloader.seq_tag2id(tag_seq), self.dataloader.seq_intent2id(intents)]]

        pad_batch = self.dataloader.pad_batch(batch_data)
        pad_batch = tuple(t.to(self.model.device) for t in pad_batch)
        word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor, context_seq_tensor, context_mask_tensor = pad_batch
        slot_logits, intent_logits = self.model.forward(word_seq_tensor, word_mask_tensor,
                                                        context_seq_tensor=context_seq_tensor,
                                                        context_mask_tensor=context_mask_tensor)
        intent = recover_intent(self.dataloader, intent_logits[0], slot_logits[0], tag_mask_tensor[0],
                                batch_data[0][0], batch_data[0][-4])
        sys_text = str(intent)

        return sys_text

    def train(self, data):
        """
        Not implemented.

        We can use Ludwig's API to train the model given the experience.

        :param data: dialogue experience
        :return:
        """
        pass

    def train_online(self, data):
        """
        Not implemented.

        We can use Ludwig's API to train the model online (i.e. for a single
        dialogue).

        :param data: dialogue experience
        :return:
        """
        pass

    def save(self, path=None):
        """
        Save the Ludwig model.

        :param path: the path to save to
        :return:
        """
        if not path:
            print('WARNING: Ludwig MetalWOZ model not saved '
                  '(no path provided).')
        else:
            self.model.save(path)

    def load(self, model_path):
        """
        Loads the Ludwig model from the given path.

        :param model_path: path to the model
        :return:
        """

        pass

