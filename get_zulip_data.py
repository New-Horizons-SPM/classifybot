#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:05:48 2022

pull in image pngs on firebase indexed on zulip thread, and get associated emoji reactions

@author: jack
"""

import os

import zulip


## specify zuliprc file
zuliprc_path = os.getcwd() + '/zuliprc'
client = zulip.Client(config_file=zuliprc_path)

## specify stream & scanbot email address to read messages:
    
request = {
    "narrow" : [
        {"operator": "sender", "operand": "scanbot-bot@zulip.schiffrin-zulip.cloud.edu.au"},
        {"operator": "stream", "operand": "scanbot"},
        {"operator": "topic", "operand": "survey"},
    ]    
}

result = client.get_messages(request)

