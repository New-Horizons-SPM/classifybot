#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:05:48 2022

pull in image pngs on firebase indexed on zulip thread, and get associated emoji reactions

@author: jack
"""

import os

import time

import zulip

import wget


# make database files
try:
    os.mkdir('image_data')
except:
    print('image_data dir already exists')



## specify zuliprc file
zuliprc_path = os.getcwd() + '/zuliprc'
client = zulip.Client(config_file=zuliprc_path)


    

## specify stream & scanbot email address to read messages:
request = {}
## define the narrow
request['narrow'] = [
        {"operator": "sender", "operand": "scanbot-bot@zulip.schiffrin-zulip.cloud.edu.au"},
        {"operator": "stream", "operand": "scanbot"},
        # {"operator": "topic", "operand": "survey"},
    ]  

# get id of first unread message
request['anchor'] = 'first_unread'
request['num_before'] = 0
request['num_after'] = 0

# ##go for oldest
# request['anchor'] = 'oldest'
# request['num_before'] = 0
# request['num_after'] = 1

result = client.get_messages(request)

first_unread_id = result['messages'][0]['id']

## get id of newest message
request['anchor'] = 'newest'
request['num_before'] = 1
request['num_after'] = 0

result = client.get_messages(request)

newest_message_id = result['messages'][0]['id']

to_mark_read = []
for message_id in range(first_unread_id, newest_message_id, 100):
    request['anchor'] = message_id
    request['num_before'] = 0
    request['num_after'] = 100
    
    results = client.get_messages(request)
    if results['result'] == 'success':
        for message in results['messages']:
            if message['content'].startswith('<div class="message_inline_image">') and 'read' not in message['flags']:
                url = message['content'].split('<a href="')[1].split('">')[0].replace('&amp;', '&')
                for reaction in message['reactions']:
                    path = 'image_data/'+reaction['emoji_name']
                    try:
                        os.mkdir(path)
                    except:
                        pass
                    
                    try:
                        ## down wget a duplicate png
                        if not url.split('/scanbot/')[1].split('?')[0] in os.listdir(path):
                            wget.download(url=url, out=path)
                    except:
                        pass
                ## mark the message as read
                to_mark_read.append(message['id'])

    else:
        print(results)
        
## now mark all read
if len(to_mark_read) > 0:
    mkrd_request = {
        'messages': to_mark_read,
        'op': 'add',
        'flag': 'read',
        }
    result = client.update_message_flags(mkrd_request)
    
    ## put a sleep in here to not hit the API rate limit
    for msg_id in to_mark_read:
        react_request = {
            'message_id': msg_id,
            'emoji_name': 'eyes',
            }
        result = client.add_reaction(react_request)
        time.sleep(.3)
    
    

        