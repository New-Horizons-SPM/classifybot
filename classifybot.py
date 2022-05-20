#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 16:15:19 2022

@author: jack
"""

import os

import tensorflow as tf
from tf_custom_metric import macro_soft_f1

import wget

import pickle

import numpy as np

import zulip

class classifybot(object):
    
    def __init__(self):
        self.model = tf.keras.models.load_model('kf_model.model', custom_objects={'macro_soft_f1':macro_soft_f1})
        self.class_names = pickle.load(open('class_names.pkl', 'rb'))
        self.client = zulip.Client(config_file='zuliprc')
        self.threshold = .8
        
    def handle_message(self, message, bot_handler=None):
        self.bot_handler = bot_handler
        
        if 'set_threshold' in message['content']:
            self.threshold = float(message['content'].split(' ')[1])
        
        if(bot_handler):
            ## get the png
            print(message['content'])
            
            # url = message['content'].split('<a href="')[1].split('">')[0].replace('&amp;', '&')
            url = message['content'].split('(')[1].split(')')[0] #.replace('&amp;', '&')
            file = wget.download(url)
            
            ## from https://www.tensorflow.org/tutorials/images/classification
            img_height = 224
            img_width = 224
            
            img = tf.keras.utils.load_img(file, target_size=(img_height, img_width))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            
            predictions = self.model.predict(img_array)
            score = np.array(tf.nn.softmax(predictions[0]))
            
            ## send reactions for categories above 80%:
            score /= np.amax(np.array(score))
            class_names = np.array(self.class_names)
            emojis = class_names[score > self.threshold]
            
            for emoji in emojis:
                react_request = {
                    'message_id': message['id'],
                    'emoji_name': emoji,
                    }
                _ = self.client.add_reaction(react_request)
            
            os.remove(file)
            
handler_class = classifybot