# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 11:32:58 2022

@author: jced0001
"""

import zulip
import torch
import numpy as np
import train_model as cnn
from train_model import ConvNet
from get_zulip_data import getZulipData
import imageio as iio
import wget
import os

class classifybot(object):
###############################################################################
# Constructor
###############################################################################
    def __init__(self):
        self.init()
        
        self.getWhitelist()                                                     # Load in whitelist file if there is one
        self.initCommandDict()                                                  # Initialise dictionary containing all of classifybot's commands
    
###############################################################################
# Initialisation
###############################################################################
    def init(self):
        with open('config.ini','r') as f:                                       # Open the config file
            config = f.read()                                                   # config.ini must contain all of the following:
            self.zuliprc         = config.split('zulip_rc_path=')[1].split('\n')[0]   # Path to the zulip bot's rc file
            self.scanbotAddress  = config.split('scanbot_address=')[1].split('\n')[0] # scanbot's email address
            self.scanbotStream   = config.split('scanbot_stream=')[1].split('\n')[0]  # stream to walk through
            
        self.notifications = True
        self.bot_message = ""
        
        self.zulipClient = []
        if(self.zuliprc):
            self.zulipClient = zulip.Client(config_file=self.zuliprc)
            
    def getWhitelist(self):
        self.whitelist = []
        try:
            with open('whitelist.txt', 'r') as f:
                d = f.read()
                self.whitelist = d.split('\n')[:-1]
        except:
            print('No whitelist... add users to create one')

    def initCommandDict(self):
        self.commands = {'list_commands'    : self.listCommands,
                         'help'             : self._help,
                         'add_user'         : self.addUsers,
                         'list_users'       : lambda args: str(self.whitelist),
                         'stop'             : self.stop,
                         'get_model'        : self.getModel,
                         'train_model'      : self.trainModel,
                         'load_model'       : self.loadModel,
                         'load_zulip_data'  : self.loadZulipData,
                         'predict'          : self.predict
        }
        
###############################################################################
# Zulip
###############################################################################
    def handle_message(self, message, bot_handler=None):
        messageContent   = message
        self.bot_message = []
        self.bot_handler = bot_handler
        if(bot_handler):
            if message['sender_email'] not in self.whitelist and self.whitelist:
                self.sendReply(message['sender_email'])
                self.sendReply('access denied')
                return
            
            self.bot_message = message
            messageContent = message['content']
            
        command = messageContent.split(' ')[0].lower()
        args    = messageContent.split(' ')[1:]
        
        if(not command in self.commands):
            reply = "Invalid command. Run *list_commands* to see command list"
            self.sendReply(reply)
            return
        
        reply = self.commands[command](args)
        
        if(reply): self.sendReply(reply)
    
    def sendReply(self,reply,message=""):
        """
        Send reply text. Currently only supports zulip and console.

        Parameters
        ----------
        reply   : Reply string
        message : Zulip: message params for the specific message to reply ro.
                  If not passed in, replies to the last message sent by user.

        Returns
        -------
        message_id : Returns the message id of the sent message. (zulip only)

        """
        if(not reply): return                                                   # Can't send nothing
        if(self.notifications):                                                 # Only send reply if notifications are turned on
            if(self.bot_handler):                                               # If our reply pathway is zulip
                replyTo = message                                               # If we're replying to a specific message
                if(not replyTo): replyTo = self.bot_message                     # If we're just replying to the last message sent by user
                self.bot_handler.send_reply(replyTo, reply)                     # Send the message
                return
        
        print(reply)                                                            # Print reply to console
    
    def reactToMessage(self,reaction,message=""):
        """
        Scanbot emoji reaction to message

        Parameters
        ----------
        reaction : Emoji name (currently zulip only)
        message  : Specific zulip message to react to. If not passed in, reacts
                   to the last message sent by user.

        """
        if(not self.bot_handler):                                               # If we're not using zulip
            print("Scanbot reaction: " + reaction)                              # Send reaction to console
            return
        reactTo = message                                                       # If we're reacting to a specific zulip message
        if(not reactTo): reactTo = self.bot_message                             # Otherwise react to the last user message
        react_request = {
            'message_id': reactTo['id'],                                        # Message ID to react to
            'emoji_name': reaction,                                             # Emoji scanbot reacts with
            }
        self.zulipClient.add_reaction(react_request)                            # API call to react to the message
        
###############################################################################
# Classifybot Commands
###############################################################################
    def listCommands(self,args):
        return "\n". join([c for c in self.commands])
    
    def addUsers(self,user,_help=False):
        arg_dict = {'' : ['', 0, "(string) Add user email to whitelist (one at a time)"]}
        
        if(_help): return arg_dict
        
        if(len(user) != 1): self.reactToMessage("cross_mark"); return
        if(' ' in user[0]): self.reactToMessage("cross_mark"); return
        try:
            self.whitelist.append(user[0])
            with open('whitelist.txt', 'w') as f:
                for w in self.whitelist:
                    f.write(w+'\n')
        except Exception as e:
            return str(e)
        
        self.reactToMessage("+1")
    
    def stop(self,args):
        pass
    
    def trainModel(self,user_args,_help=False):
        arg_dict = {'-name'     : ['-default', lambda x: str(x), "(str) Name the model"],
                    '-target'   : ['-default', lambda x: str(x), "(str) Target label (binary implementation for now)"],
                    '-pklpath'  : ['-default', lambda x: str(x), "(str) Path to pickled data"],
                    '-augment'  : ['-default', lambda x: int(x), "(int) Augment data (reflections)"],
                    '-load'     : ['1',        lambda x: int(x), "(int) Auto load model after training. 0=No, 1=Yes"]}
        
        if(_help): return arg_dict
        
        error,user_arg_dict = self.userArgs(arg_dict,user_args)
        if(error): return error + "\nRun ```help train_model``` if you're unsure."
        
        with open('config.ini','r') as f:                                       # Open the config file
            config          = f.read()                                          # config.ini must contain all of the following:
            default_args = {'-name'    : config.split('run_name=')[1].split('\n')[0],       # Name of the run. pkl saved as zulipData-runName-batch_x.pkl
                            '-target'  : config.split('target_label=')[1].split('\n')[0],   # The label with respect to which we try and classify. Binary for now
                            '-pklpath' : config.split('pkl_path=')[1].split('\n')[0],       # Output path for pkl'd data
                            '-augment' : config.split('augment_data=')[1].split('\n')[0]}   # Flag to augment data during training
        
        for key in user_arg_dict:
            if(user_arg_dict[key][0] == "-default"):
                user_arg_dict[key][0] = default_args[key]
        
        args = self.unpackArgs(user_arg_dict)
        
        self.reactToMessage("gym")
        try:
            modelPath = cnn.trainNewCNN(*args[0:4])
            self.reactToMessage("muscle")
            if(user_arg_dict['-load'][0] == '0'): return
            self.model = ConvNet(load_model=modelPath)
            self.reactToMessage("computer")
        except Exception as e:
            return str(e)
        
        with open('config.ini','r+') as f:
            config = str(f.read())
            oldModel  = config.split('run_name=')[1].split('\n')[0]
            oldTarget = config.split('target_label=')[1].split('\n')[0]
            oldpkl    = config.split('pkl_path=')[1].split('\n')[0]
            oldaug    = config.split('augment_data=')[1].split('\n')[0]
            
            config = config.replace('run_name=' + oldModel,
                                    'run_name=' + user_arg_dict['-name'][0])
            config = config.replace('target_label=' + oldTarget,
                                    'target_label=' + user_arg_dict['-target'][0])
            config = config.replace('pkl_path=' + oldpkl,
                                    'pkl_path=' + user_arg_dict['-pklpath'][0])
            config = config.replace('augment_data=' + oldaug,
                                    'augment_data=' + user_arg_dict['-augment'][0])
            
        with open('config.ini','w') as f:
            f.write(config) 
    
    def loadModel(self,user_args,_help=False):
        arg_dict = {'-name': ['-default', lambda x: str(x), "(str) Name the model"]}

        if(_help): return arg_dict
        
        error,user_arg_dict = self.userArgs(arg_dict,user_args)
        if(error): return error + "\nRun ```help laod_model``` if you're unsure."
        
        with open('config.ini','r') as f:                                       # Open the config file
            config          = f.read()                                          # config.ini must contain all of the following:
            default_args = {'-name': config.split('run_name=')[1].split('\n')[0]} # Name of the run. pkl saved as zulipData-runName-batch_x.pkl
        
        for key in user_arg_dict:
            if(user_arg_dict[key][0] == "-default"):
                user_arg_dict[key][0] = default_args[key]
        
        args = self.unpackArgs(user_arg_dict)
        
        try:
            self.model = ConvNet(*args)
            print(self.model)
        except Exception as e:
            return str(e)
        
        with open('config.ini','r+') as f:
            config = str(f.read())
            oldModel = config.split('run_name=')[1].split('\n')[0]
            config = config.replace('run_name=' + oldModel,
                                    'run_name=' + user_arg_dict['-name'][0])
            
        with open('config.ini','w') as f:
            f.write(config) 
            
        self.reactToMessage("computer")
        
    def predict(self,user_args,_help=False):
        try:
            try:    url = self.bot_message['content'].split('<a href="')[1].split('">')[0].replace('&amp;', '&')
            except: url = self.bot_message['content'].split('(')[1].split(')')[0]
            filename = wget.download(url=url)
            im = iio.imread(filename)
            os.remove(filename)
            im = np.array(im/np.max(im),dtype=np.float32)                       # Normalise the data and force to be float32
            x = []
            x.append(np.transpose(im[:,:,:3], (2,0,1)))                               # Append the image to x
            x = np.array(x)                                                     # Convert to numpy array
            x = torch.tensor(x)                                                 # Convert to a pytorch compatible tensor
            print("predicting")
            prediction = self.model(x[0:1])
            self.reactToMessage(["cross_mark","bulls_eye"][np.argmax(prediction.detach())])
        except Exception as e:
            self.sendReply(str(e))
            self.reactToMessage("no_entry")
    
    def loadZulipData(self,user_args,_help=False):
        arg_dict = {'-name'     : ['-default', lambda x: str(x), "(str) Name the model"],
                    '-msgid'    : ['-default', lambda x: str(x), "(str) Last message ID"],
                    '-target'   : ['-default', lambda x: str(x), "(str) Target label (binary implementation for now)"],
                    '-pklpath'  : ['-default', lambda x: str(x), "(str) Path to pickled data"],
                    '-augment'  : ['-default', lambda x: int(x), "(int) Augment data (reflections)"]}
        
        if(_help): return arg_dict
        
        error,user_arg_dict = self.userArgs(arg_dict,user_args)
        if(error): return error + "\nRun ```help train_model``` if you're unsure."
        
        with open('config.ini','r') as f:                                       # Open the config file
            config          = f.read()                                          # config.ini must contain all of the following:
            default_args = {'-name'    : config.split('run_name=')[1].split('\n')[0],       # Name of the run. pkl saved as zulipData-runName-batch_x.pkl
                            '-msgid'   : config.split('last_message_id=')[1].split('\n')[0],# Message ID anchor (get messages from this message ID)
                            '-target'  : config.split('target_label=')[1].split('\n')[0],   # The label with respect to which we try and classify. Binary for now
                            '-pklpath' : config.split('pkl_path=')[1].split('\n')[0],       # Output path for pkl'd data
                            '-augment' : config.split('augment_data=')[1].split('\n')[0]}   # Flag to augment data during training
        
        for key in user_arg_dict:
            if(user_arg_dict[key][0] == "-default"):
                user_arg_dict[key][0] = default_args[key]
        
        config_bk = []
        with open('config.ini','r+') as f:
            config = str(f.read())
            config_bk = config                                                  # Keep a backup of this config in case we error, then put back the old config
            oldModel  = config.split('run_name=')[1].split('\n')[0]
            oldMsgID  = config.split('last_message_id=')[1].split('\n')[0]
            oldTarget = config.split('target_label=')[1].split('\n')[0]
            oldpkl    = config.split('pkl_path=')[1].split('\n')[0]
            oldaug    = config.split('augment_data=')[1].split('\n')[0]
            
            config = config.replace('run_name=' + oldModel,
                                    'run_name=' + user_arg_dict['-name'][0])
            config = config.replace('last_message_id=' + oldMsgID,
                                    'last_message_id=' + user_arg_dict['-msgid'][0])
            config = config.replace('target_label=' + oldTarget,
                                    'target_label=' + user_arg_dict['-target'][0])
            config = config.replace('pkl_path=' + oldpkl,
                                    'pkl_path=' + user_arg_dict['-pklpath'][0])
            config = config.replace('augment_data=' + oldaug,
                                    'augment_data=' + user_arg_dict['-augment'][0])
            
        with open('config.ini','w') as f:
            f.write(config) 
        
        try:
            self.reactToMessage("working_on_it")
            getZulipData()
            self.reactToMessage("computer")
        except Exception as e:
            with open('config.ini','w') as f:
                f.write(config_bk) 
            self.reactToMessage("-1")
            return str(e)
    
    def getModel(self,user_args,_help=False):
        if(_help): return
        
        with open('config.ini','r+') as f:
            config = str(f.read())
            model = config.split('run_name=')[1].split('\n')[0]
        
        return(model)
###############################################################################
# Utilities
###############################################################################  
    def userArgs(self,arg_dict,user_args):
        error = ""
        for arg in user_args:                                                   # Override the defaults if user inputs them
            try:
                key,value = arg.split('=')
            except:
                error = "Invalid argument"
                break
            if(not key in arg_dict):
                error = "invalid argument: " + key                              # return error message
                break
            try:
                arg_dict[key][1](value)                                         # Validate the value
            except:
                error  = "Invalid value for arg " + key + "."                   # Error if the value doesn't match the required data type
                break
            
            arg_dict[key][0] = value
        
        return [error,arg_dict]
    
    def unpackArgs(self,arg_dict):
        args = []
        for key,value in arg_dict.items():
            if(value[0] == "-default"):                                         # If the value is -default...
                args.append("-default")                                         # leave it so the function can retrieve the value from nanonis
                continue
            
            args.append(value[1](value[0]))                                     # Convert the string into data type
        
        return args
    
    def _help(self,args):
        if(not len(args)):
            helpStr = "Type ```help <command name>``` for more info\n"
            return helpStr + self.listCommands(args=[])
        
        command = args[0]
        if(not command in self.commands):
            return "Run ```list_commands``` to see valid commands"
        
        try:
            helpStr = "**" + command + "**\n"
            arg_dict = self.commands[command](args,_help=True)
            for key,value in arg_dict.items():
                if(key):
                    helpStr += "```"
                    helpStr += key + "```: " 
                helpStr += value[2] + ". "
                if(value[0]):
                    helpStr += "Default: ```" +  value[0].replace("-default","config file") 
                    helpStr += "```"
                helpStr += "\n"
        except:
            return "No help for this command"
        
        return helpStr
    
handler_class = classifybot