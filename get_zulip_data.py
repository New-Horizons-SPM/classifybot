#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:05:48 2022

pull in image pngs on firebase indexed on zulip thread, and get associated emoji reactions

@author: jack
"""

import os
# import time
import zulip
import wget
import pickle


# make database files
with open('config.ini', 'r') as f:
    config = f.read()
    image_data_path = config.split('image_database_directory=')[1].split('\n')[0]
    zuliprc_name = config.split('zuliprc_name=')[1].split('\n')[0]
    classifybot_name = config.split('classifybot_name=')[1].split('\n')[0]
    scanbot_address = config.split('scanbot_address=')[1].split('\n')[0]
    scanbot_stream = config.split('scanbot_stream=')[1].split('\n')[0]

try:
    os.mkdir(image_data_path)
except:
    print('image_data dir already exists')
    
if not 'batch_0' in os.listdir(image_data_path):
    os.mkdir(os.path.join(image_data_path, 'batch_0'))



## specify zuliprc file
zuliprc_path = os.path.join(os.getcwd(), zuliprc_name)
client = zulip.Client(config_file=zuliprc_path)

## specify hard-coded classifybot name
# classifybot_name = 'classifybot'


    

## specify stream & scanbot email address to read messages:
request = {}
## define the narrow
request['narrow'] = [
        {"operator": "sender", "operand": scanbot_address},
        {"operator": "stream", "operand": scanbot_stream},
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
print('first_unread anchor')
print(result)

first_unread_id = result['messages'][0]['id']

## get id of newest message
request['anchor'] = 'newest'
request['num_before'] = 1
request['num_after'] = 0

result = client.get_messages(request)
print('newest anchor')
print(result)

newest_message_id = result['messages'][0]['id']

to_mark_read = []
keep_unread = []
for message_id in range(first_unread_id, newest_message_id, 100):
    request['anchor'] = message_id
    request['num_before'] = 0
    request['num_after'] = 100
    
    ## check which batch folder to put images in
    folders = os.listdir(image_data_path)
    max_batch_index = 0
    for name in folders:
        if name.split('_')[0] == 'batch':
            batch_index = int(name.split('_')[1])
            if batch_index > max_batch_index:
                max_batch_index = batch_index
    
    batch_path = os.path.join(image_data_path, 'batch_' + str(max_batch_index))
    
    if len(os.listdir(batch_path)) > 256:
        batch_path = os.path.join(image_data_path, 'batch_' + str(max_batch_index+1))
        os.mkdir(batch_path)
        pickle.dump(True, open('retrain_flag.pkl', 'wb'))
    
    try:
        batch_labels = pickle.load(open(os.path.join(batch_path,'file_labels.pkl'), 'rb'))
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
                
                if len(labels) > 0 and '.png' in url:
                    try:
                        if not url.split('/scanbot/')[1].split('?')[0] in os.listdir(batch_path):
                            filename = wget.download(url=url, out=batch_path)
                            keyname = str(os.path.join(batch_path.split('/')[-1], filename.split('/')[-1]))
                            batch_labels[keyname] = labels
                    except:
                        pass
                                    
                    ## mark the message as read
                    to_mark_read.append(message['id'])
                    
                else:
                    keep_unread.append(message['id'])
                    
            else: ## message doesn't have an image in it; mark read
                to_mark_read.append(message['id'])

    else:
        print(results)
    
    ## dump the batch_labels
    pickle.dump(batch_labels, open(os.path.join(batch_path, 'file_labels.pkl'), 'wb'))
        
    
        
## now mark all read
if len(to_mark_read) > 0:
    mkrd_request = {
        'messages': to_mark_read,
        'op': 'add',
        'flag': 'read',
        }
    result = client.update_message_flags(mkrd_request)
    print(result)
    
    # ## put a sleep in here to not hit the API rate limit
    # print('the slow loop to add eyes to all mark-read files')
    # for msg_id in to_mark_read:
    #     react_request = {
    #         'message_id': msg_id,
    #         'emoji_name': 'eyes',
    #         }
    #     result = client.add_reaction(react_request)
    #     time.sleep(.3)
    
## mark messages that haven't been labelled unread
mkunread_request = {
        'messages': keep_unread,
        'op': 'remove',
        'flag': 'read',
    }
result = client.update_message_flags(mkunread_request)
print(result)
    
    

        