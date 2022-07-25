#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:05:48 2022

config.ini must contain
run_name=<run_name>                                                             # Name given to the run. Outut is saved as <run_name>.pkl
zulip_rc_path=<path to bot rc file>
scanbot_address=<scanbot email address>
scanbot_stream=<stream scanbot is sending data to>
last_message_id=<id of the oldest message to anchor to>                         # This gets overwritten after the run. set to 0 to start from begining of time
label_dict=<map emoji to label>                                                 # In the form of emoji1:label1,emoji2:label2. useful for same emoji with multiple names (e.g. -1 and thumbs_down)

This code scans <scanbot_stream> beginning at <last_message_id>. Selects all
messages sent from <scanbot_address> directed at @<classifybot_handle>

@author: Julian Ceddia & Jack Hellerstedt
"""

import zulip
import pickle
import wget
import os
import imageio as iio

def getZulipData():
    with open('config.ini','r') as f:
        config          = f.read()                                                  # config.ini must contain all of the following:
        runName         = config.split('run_name=')[1].split('\n')[0]               # Name of the run. pkl saved as zulipData-runName-batch_x.pkl
        zulipRcPath     = config.split('zulip_rc_path=')[1].split('\n')[0]          # Path to the zulip bot's rc file
        scanbotAddress  = config.split('scanbot_address=')[1].split('\n')[0]        # scanbot's email address
        scanbotStream   = config.split('scanbot_stream=')[1].split('\n')[0]         # stream to walk through
        lastMsgID       = int(config.split('last_message_id=')[1].split('\n')[0])   # Start searching from this message ID. Doesn't have to be an exact message ID match. 0 starts from begining. Autoupdated at the end of a run to pick up from where you left off
        batchSize       = int(config.split('batch_size=')[1].split('\n')[0])        # Number of images per pkl file
        nbatch          = int(config.split('nbatch=')[1].split('\n')[0])            # Number of batches to process -1 for entire stream. 1 batch per pkl
        userList        = config.split('user_list=')[1].split('\n')[0].split(',')   # Users to read labels from
        labelDictIni    = config.split('label_dict=')[1].split('\n')[0]             # Map emojis to labels... In the form of emoji1:label1,emoji2:label2. useful for same emoji with multiple names (e.g. -1 and thumbs_down)
        pklPath         = config.split('pkl_path=')[1].split('\n')[0]               # Output path for pkl'd data
    
    labelDict = {}                                                                  # Convert labelDictini to python dictionary...
    labelDictIni = labelDictIni.split(',')
    for ld in labelDictIni:
        key,value = ld.split(':')
        labelDict[key] = value
    
    if(not pklPath.endswith('/')): pklPath += '/'
    pklPath += runName + '/'
    pklParams = {"runName":         runName,                                        # Store config params in the final pickle file
                 "zulipRcPath":     zulipRcPath,
                 "scanbotAddress":  scanbotAddress,
                 "scanbotStream":   scanbotStream,
                 "lastMsgID":       lastMsgID,
                 "userList":        userList,
                 "batchSize":       batchSize,
                 "nbatch":          nbatch,
                 "labelDict":       labelDict}
    
    client = zulip.Client(config_file=zulipRcPath)                                  # Zulip Client
    handle = client.get_profile()['full_name']                                      # Bot's handle
    
    try: os.mkdir(pklPath)
    except: pass
    try: os.mkdir(pklPath + "labelled")
    except: pass
    try: os.mkdir(pklPath + "unlabelled")
    except: pass
    try:
        labelledBatchNo = max([int(b.split('batch_')[1].split('.pkl')[0]) for b in  # If there are any pkl's with this runName, check what batch number we're up to
                               [f for f in os.listdir(pklPath + 'labelled/')
                                if('zulipData-' + runName + '-labelled-' in f)]])
        labelledData = pickle.load(open(pklPath + 'labelled/zulipData-' + runName + '-labelled-batch_' # load in the batch file to continue adding to it
                                        + str(labelledBatchNo) + '.pkl','rb'))
        labelledData = labelledData['data']                                         # just get the data from it
        if(len(labelledData) >= batchSize):                                         # If this batch is full, start from the next one
            labelledBatchNo += 1
            labelledData = {}
    except:                                                                         # If no batches with runName, start from 0
        labelledBatchNo = 0
        labelledData   = {}
    
    try:
        unlabelledBatchNo = max([int(b.split('batch_')[1].split('.pkl')[0]) for b in#  If there are any pkl's with this runName, check what batch number we're up to
                                 [f for f in os.listdir(pklPath + 'unlabelled/')
                                  if('zulipData-' + runName + '-unlabelled-' in f)]])
        unlabelledData = pickle.load(open(pklPath + 'unlabelled/zulipData-' + runName + '-unlabelled-batch_' # load in the batch file to continue adding to it
                                          + str(unlabelledBatchNo) + '.pkl','rb'))
        unlabelledData = unlabelledData['data']                                     # just get the data from it
        if(len(unlabelledData) >= batchSize):                                       # If this batch is full, start from the next one
            unlabelledBatchNo += 1
            unlabelledData = {}
    except:                                                                         # If no batches with runName, start from 0
        unlabelledBatchNo = 0
        unlabelledData = {}
        
    pbatch = 0                                                                      # Number of batches completed so far
    result = {'found_newest': False}                                                # Initialise result
    while(not result['found_newest'] and (pbatch < nbatch or nbatch < 0)):          # Keep going until we run out of messages in the stream, or until we hit our batch limit. If nbatch=-1 then process all messages
        request = {}                                                                # Request to pull messages
        request['narrow'] = [                                                       # Filter search on...
                {"operator": "sender", "operand": scanbotAddress},                  # Sender being scanbot (scanbot's email address)
                {"operator": "stream", "operand": scanbotStream}]                   # Stream being 'scanbot'
                # {"operator": "topic", "operand": "survey"},                       # Topic being 'survey'
        request['anchor'] = lastMsgID                                               # Start search from this message ID.
        request['num_before'] = 0                                                   # 0 messages before lastMsgID
        request['num_after'] = 100                                                  # Grab the 100 messages after lastMsgID
        result = client.get_messages(request)                                       # Perform search
        
        messages = result['messages']                                               # All messages returned from search
        
        for message in messages:
            if(pbatch == nbatch): break                                             # Stop processing if we've hit our batch limit
            if '.sxm' in message['content']:                                        # If there's an sxm filename in the message content
                try:
                    sxmFile = message['content'].split('.sxm')[0].split('/')[-1] + ".sxm"
                    if(sxmFile in labelledData):   continue                         # Don't pickup the same sxm file twice
                    if(sxmFile in unlabelledData): continue                         # Don't pickup the same sxm file twice
                    
                    try:
                        url = message['content'].split('<a href="')[1].split('">')[0].replace('&amp;', '&')
                    except:
                        url = message['content'].split('(')[1].split(')')[0]
                    
                    labels = []
                    for reaction in message['reactions']:
                        if reaction['user']['email'] in userList:                   # Only look at labels from users in the list
                            label = reaction['emoji_name']
                            if(label not in labelDict):     continue                # Only process labels listed in the config file
                            if(labelDict[label] in labels): continue                # Only add the label if it (or an equivalent one) hasn't been added yet
                            labels.append(labelDict[label])                         # Append the label to the list for this sxm
                    
                    if(len(labels)):
                        filename = wget.download(url=url)
                        im = iio.imread(filename)
                        os.remove(filename)
                        labelledData[sxmFile] = [im,labels]                         # Only data with more than zero labels goes in this list
                        if(len(labelledData) == batchSize):
                            print("Batch " + str(pbatch) + " complete")
                            pklParams['data'] = labelledData
                            pklName  = 'zulipData-' + runName + '-labelled-'
                            pklName += 'batch_' + str(labelledBatchNo) + '.pkl'
                            pickle.dump(pklParams, open(pklPath + 'labelled/' + pklName, 'wb')) # Pickle containing config settings and labelled data
                            labelledBatchNo += 1
                            labelledData   = {}
                            pbatch += 1
                    # else:
                    #     unlabelledData[sxmFile] = [im,labels]                       # Unlabelled data goes in this list
                    #     if(len(unlabelledData) == batchSize):
                    #         pklParams['data'] = unlabelledData
                    #         pklName  = 'zulipData-' + runName + '-unlabelled-'
                    #         pklName += 'batch_' + str(unlabelledBatchNo) + '.pkl'
                    #         pickle.dump(pklParams, open(pklPath + 'unlabelled/' + pklName, 'wb')) # Pickle containing config settings and unlabelled data
                    #         unlabelledBatchNo += 1
                    #         unlabelledData = {}
                    #         pbatch += 1
                    
                    lastMsgID = message['id'] + 1                                   # Remember this number for next time, so we don't need to go through entire message history
                except:
                    pass
                    
    if(len(labelledData)):
        pklParams['data'] = labelledData
        pklName  = 'zulipData-' + runName + '-labelled-'
        pklName += 'batch_' + str(labelledBatchNo) + '.pkl'
        pickle.dump(pklParams, open(pklPath + 'labelled/' + pklName, 'wb'))         # Pickle containing config settings and labelled data
        
    # if(len(unlabelledData)):
    #     pklParams['data'] = unlabelledData
    #     pklName  = 'zulipData-' + runName + '-unlabelled-'
    #     pklName += 'batch_' + str(unlabelledBatchNo) + '.pkl'
    #     pickle.dump(pklParams, open(pklPath + 'unlabelled/' + pklName, 'wb'))       # Pickle containing config settings and unlabelled data
    #     unlabelledData = {}
    
    with open('config.ini','r+') as f:
        config = str(f.read())
        oldMsgID = config.split('last_message_id=')[1].split('\n')[0]
        config = config.replace('last_message_id=' + str(oldMsgID),
                                'last_message_id=' + str(lastMsgID))
        
    with open('config.ini','w') as f:
        f.write(config)                                                             # Update config.ini with lastMsgID to pick up from where we left off