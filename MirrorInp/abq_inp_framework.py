#/usr/bin/python
# abq_inp_framework.py
#  
# ABAQUS input file framework by Dr. Fabio Geraci
# 
# This script takes as an input, a folder structure containing *inp files in any number of subfolders.
# The user can then define custom logic in acting on the different ABAQUS cards using IF statements.
# This script uses the concept of 'stream processing' similar to UNIX 'sed' but retains certain memories or states,
# most important being the current ABAQUS card.
#
# Running scripts:
# Use UNIX or Windows command prompt. Script and INP folder should be in the same place.
# Navigate to this folder and run:
# python abq_inp_framework.py foldername
#
# replace abq_inp_framework.py with the python script name if you rename this file.
# replace foldername with the actual folder name where the INP files are located.

import sys
from math import *
import shutil
import os
import re
import fileinput
import operator

#add keywords to both flags and flag_regex.

flags = {'nodeflag':False,
        'orientflag':False,
        'elementflag':False,
        'cload_flag': False,
        'pretension_f': False,
        'tx_flag': False,}

# You can keep these variables as examples on how to write Regular Expressions, or remove them for faster performance.
# But make sure you add the keyword you're interested in!        
flag_regex = {'nodeflag':re.compile('\*node\s*,|\*node\s*$', re.IGNORECASE), # *node (zero or more whitespace) + comma OR *node (zero or more whitespace) +newline (to avoid *NODE OUTPUT and similar cards)
              'orientflag': re.compile('\*ORIENTATION',re.IGNORECASE),
              'elementflag': re.compile('\*element\s*,',re.IGNORECASE), # *element (zero or more whitespace) + comma (just to avoid detecting *ELEMENT OUTPUT or similar cards)
              'cload_flag': re.compile('\*CLOAD', re.IGNORECASE),
              'pretension_f': re.compile('\*Pre-tension Section', re.IGNORECASE),
              'tx_flag': re.compile('\*TRANSFORM', re.IGNORECASE)}

# dictionary to store the parameters in. i.e. keyword_parameters['parameter'] = value (all stored as strings)             
keyword_parameters = {}
              
def unsetflags():
    '''uset all state flags'''
    for keyword in flags.keys():
        flags[keyword] = False
    keyword_parameters = {}
    
def setflags(line):
    '''sets state flags based on regular expression'''
    for keyword in flags.keys():
        if flag_regex[keyword].search(line):
            flags[keyword] = True
    if len(line.split('=')) > 1:
        for params in line.split(',')[1:]:
            paramname = params.split('=')[0]
            keyword_parameters[paramname] = params.split('=')[1]
    
if len(sys.argv) < 2:
    print 'Python Abaqus input file framework'
    print 'usage: python abq_inp_framework.py [folder]'
    sys.exit()

src_dir = sys.argv[1]

dst_dir = src_dir + '-python'
shutil.copytree(src_dir,dst_dir)  # copy the folder structure
path = dst_dir

# regex to detect a keyword
keyword_re = re.compile('^\*\w')

totalfiles = 0

for (path, dirs, files) in os.walk(path):
    for file in files:
        #print file
        if '.inp' in file:
            totalfiles += 1
            sys.stderr.write('processing: ' + file + '\n')
            for line in fileinput.input(os.path.join(path,file),inplace=1):
                if (len(line.strip()) > 0):
                #-----------------------Finite State Machine start-------------------------
                    if keyword_re.match(line):
                        unsetflags()
                        setflags(line)
                        counter = 0
                        sline = []
                    elif line.strip()[0:2] == '**':
                         pass #do nothing, this is a comment
                    elif flags['orientflag']:
                        counter = counter + 1 # use the counter variable to keep track of lines AFTER the keyword
                        #add logic for *ORIENTATION here
                        # the variable 'line' is a string containing the current line we're processing.
                    elif flags['someflag']:
                        #another flag, do something else
                #----------------------------Finite State Machine end ----------------------------
                print line.rstrip()

sys.stderr.write('total files processed: ' + str(totalfiles)+ '\n')
