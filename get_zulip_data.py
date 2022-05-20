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
import pickle


# make database files
try:
    os.mkdir('image_data')
except:
    print('image_data dir already exists')
    
if not 'batch_0' in os.listdir('image_data'):
    os.mkdir('image_data/batch_0')



## specify zuliprc file
zuliprc_path = os.getcwd() + '/zuliprc'
client = zulip.Client(config_file=zuliprc_path)

## specify hard-coded classifybot name
classifybot_name = 'classifybot'


    

## specify stream & scanbot email address to read messages:
request = {}
## define the narrow
request['narrow'] = [
        {"operator": "sender", "operand": "scanbot-bot@zulip.schiffrin-zulip.cloud.edu.au"},
        {"operator": "stream", "operand": "scanbot"},
        # {"operator": "topic", "operand": "survey"},
    ]  

## get id of first unread message
request['anchor'] = 'first_unread'
request['num_before'] = 0
request['num_after'] = 0

##go for oldest
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
    
    ## check which batch folder to put images in
    folders = os.listdir('image_data')
    max_batch_index = 0
    for name in folders:
        if name.split('_')[0] == 'batch':
            batch_index = int(name.split('_')[1])
            if batch_index > max_batch_index:
                max_batch_index = batch_index
    
    batch_path = 'image_data/batch_' + str(max_batch_index) + '/'
    
    if len(os.listdir('image_data/' + 'batch_' + str(max_batch_index))) > 256:
        batch_path = 'image_data/batch_' + str(max_batch_index+1) + '/'
        os.mkdir(batch_path)
    
    try:
        batch_labels = pickle.load(open(batch_path + 'file_labels.pkl', 'rb'))
    except:
        print('no batch labels ' + batch_path)
        batch_labels = {}
    
    
    results = client.get_messages(request)
    if results['result'] == 'success':
        for message in results['messages']:
            if '<div class="message_inline_image">' in message['content'] and 'read' not in message['flags']:
                url = message['content'].split('<a href="')[1].split('">')[0].replace('&amp;', '&')
                labels = []
                for reaction in message['reactions']:
                    if reaction['user']['full_name'] != classifybot_name:
                        labels.append(reaction['emoji_name'])
                
                if len(labels) > 0:
                    try:
                        if not url.split('/scanbot/')[1].split('?')[0] in os.listdir(batch_path):
                            filename = wget.download(url=url, out=batch_path)
                            batch_labels[filename] = labels
                    except:
                        pass

                
                ## mark the message as read
                to_mark_read.append(message['id'])

    else:
        print(results)
    
    ## dump the batch_labels
    pickle.dump(batch_labels, open(batch_path + 'file_labels.pkl', 'wb'))
        
    
        
## now mark all read
if len(to_mark_read) > 0:
    mkrd_request = {
        'messages': to_mark_read,
        'op': 'add',
        'flag': 'read',
        }
    result = client.update_message_flags(mkrd_request)
    
    # ## put a sleep in here to not hit the API rate limit
    # print('the slow loop to add eyes to all mark-read files')
    # for msg_id in to_mark_read:
    #     react_request = {
    #         'message_id': msg_id,
    #         'emoji_name': 'eyes',
    #         }
    #     result = client.add_reaction(react_request)
    #     time.sleep(.3)
    
    

        