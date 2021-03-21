#/usr/bin/python
# reflect_fsm.py
#  
# Wing model reflection script by Dr. Fabio Geraci
# 
# - currently takes a wing model formatted to A350 standards and reflects it around global y-axis
#
# - local coordinate systems (*orientation and *transform) are reflected, then rotated 180 degrees 
#   around local x-axis to maintain right-handedness
#
# - ALL element normals are reversed to maintain correct property orientations.
#
# - Element types currently supported are:
#    - S3/S3R
#    - S4/S4R
#    - SC6/SC6R/ any pentahedral element
#    - SC8/SC8R/ any hexahedral element
#    - first-order tetrahedral elements
#    - any 2-noded elements have their nodes simply reflected (including 3rd section node in B31 elements)
# - All PART names have a suffix '-RH' added
# - All *INSTANCEs of parts are also renamed with a '-RH' suffix
# - All CLOADs are reflected around the Y-axis:
#    - for the 6 directions: [ +, -, +, -, +, -]
# - Because of the rotated local coordinate systems, ply orientations are not rotated (rotating them 180 degrees
#    around x-axis will not change properties: i.e. properties of fibre at 45 = 135 degrees, etc.)

# ####################################################
# This script uses a finite-state machine and a stream-processing approach to minimise memory use and maximise speed.
# It has been tested on Python 2.4.3, and will likely work on newer 2.x versions of Python.
# ####################################################

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
        'n3_flag': False,        
        'n4_flag': False,
        'tet_flag': False,
        'n6_flag':False,
        'n8_flag':False,
        'cload_flag': False,
        'tx_flag': False,}
        
flag_regex = {'nodeflag':re.compile('\*node\s*,|\*node\s*$', re.IGNORECASE), # *node (zero or more whitespace) +comma or *node (zero or more whitespace) +newline
              'orientflag': re.compile('\*ORIENTATION',re.IGNORECASE),
              'elementflag': re.compile('\*element\s*,',re.IGNORECASE),
              'n3_flag': re.compile('TYPE\s*=\s*S3|TYPE\s*=\s*S3R'),                    # 4-noded elements
              'n4_flag': re.compile('TYPE\s*=\s*S4|TYPE\s*=\s*S4R'),                    # 4-noded elements
              'tet_flag': re.compile('TYPE\s*=\s*C3D4'),                          # tetrahedral elements
              'n6_flag': re.compile('TYPE\s*=\s*SC6R|TYPE\s*=\s*C3D6'),                            # 6-noded elements
              'n8_flag': re.compile('TYPE\s*=\s*SC8R|TYPE\s*=\s*C3D8R|TYPE\s*=\s*C3D8I|TYPE\s*=\s*C3D8'),        # 8-noded elements
              'cload_flag': re.compile('\*CLOAD', re.IGNORECASE),
              'tx_flag': re.compile('\*TRANSFORM', re.IGNORECASE)}        

class quaternion:
    ''' quaternion class using format w + xi + yj + xk
        q[0] = w
        q[1] = x
        q[2] = y
        q[3] = z
        Brief description: Quaternions are a number system that extends the complex number system.
        Hamilton defined quaternions as the quotient between two vectors, and they are commonly used in
        3-dimensional rotations, one benefit being that they are immune to singularites or gimbal lock,
        which affect systems based on Euler angles and rotations.'''
    def __init__(self, value=[0.0,0.0,0.0,0.0]):
        self.value = map(float,value)
        self.dimx = len(value)
        
    def show(self):
        for i in range(self.dimx):
            print self.value[i]
        print ' '
        
    def __add__(self, other):
        # check if correct dimensions
        if self.dimx != other.dimx:
            raise ValueError, "Quaternions must be of equal dimensions to add"
        else:
            # add if correct dimensions
            res = quaternion()
            for i in range(self.dimx):
                res.value[i] = self.value[i] + other.value[i]
            return res

    def __sub__(self, other):
        # check if correct dimensions
        if self.dimx != other.dimx:
            raise ValueError, "Quaternions must be of equal dimensions to subtract"
        else:
            # subtract if correct dimensions
            res = quaternion()
            for i in range(self.dimx):
                res.value[i] = self.value[i] - other.value[i]
            return res

    def __mul__(self, other):
        # check if correct dimensions
        if self.dimx != other.dimx:
            raise ValueError, "Can only multiply 2 quaternions together"
        else:
            # quaternion multiplication
            res = quaternion()
            res.value[0] = self.value[0]*other.value[0] - self.value[1]*other.value[1] - self.value[2]*other.value[2] - self.value[3]*other.value[3]
            res.value[1] = self.value[0]*other.value[1] + self.value[1]*other.value[0] + self.value[2]*other.value[3] - self.value[3]*other.value[2]
            res.value[2] = self.value[0]*other.value[2] - self.value[1]*other.value[3] + self.value[2]*other.value[0] + self.value[3]*other.value[1]
            res.value[3] = self.value[0]*other.value[3] + self.value[1]*other.value[2] - self.value[2]*other.value[1] + self.value[3]*other.value[0]
            return res

    def conjugate(self):
        # compute quaternion conjugate
        res = quaternion()
        res.value[0] = self.value[0]
        res.value[1] = -self.value[1]
        res.value[2] = -self.value[2]
        res.value[3] = -self.value[3]
        return res
        
    def magnitude(self):
        res = sqrt(self.value[1]**2 + self.value[2]**2 + self.value[3]**2)
        return res
        
    def normalise(self):
        res = quaternion()
        res.value[0] = self.value[0]
        res.value[1] = self.value[1]/self.magnitude()
        res.value[2] = self.value[2]/self.magnitude()
        res.value[3] = self.value[3]/self.magnitude()
        return res
        
    def __repr__(self):
        return repr(self.value)
              
def unsetflags():
    #uset all state flags
    for keyword in flags.keys():
        flags[keyword] = False
    
def setflags(line):
    #sets state flags based on regular expression
    for keyword in flags.keys():
        if flag_regex[keyword].search(line):
            flags[keyword] = True
    
def subvec(a,b):
    '''subtract vector b from vector a'''
    c = [a[0] - b[0], a[1]-b[1], a[2]-b[2]]
    return c
    
def reverse_b(vec,axis):
    '''rotate vec 180 degrees around axis'''
    rquat = quaternion([ cos(pi/2), axis[0]*sin(pi/2), axis[1]*sin(pi/2), axis[2]*sin(pi/2)])
    #normalize quaternion
    rquat = rquat.normalise()
    #get quaternion conjugate
    qm1 = rquat.conjugate()
    #convert vector to quaternion with w=0
    qvec = quaternion([ 0.0, vec[0], vec[1], vec[2]])
    v_rot = (rquat*qvec)*qm1
    brev = [v_rot.value[1], v_rot.value[2], v_rot.value[3]]
    return brev
    
def reflect_point(point, origin):
    #reflects a point (x,y,z). TODO: add translation/rotation
    pass
    
def reflect_orient(line):
    a_vec = map(float,line.split(',')[0:3])
    b_vec = map(float,line.split(',')[3:6]) #python slices exclude last element
    c_vec = map(float,line.split(',')[6:9])
    a_vec[1] = -a_vec[1]       #need to reflect ORIENTATION coordinates as well, as they're not linked to nodes.
    b_vec[1] = -b_vec[1]                            
    c_vec[1] = -c_vec[1] 
    y_vec = subvec(b_vec,c_vec) # y = b - c (not necessarily the y axis)
    x_vec = subvec(a_vec,c_vec) # the x-axis
    bnew = reverse_b(y_vec,x_vec) 
    bnew[0] = bnew[0]+c_vec[0]; bnew[1] = bnew[1]+c_vec[1]; bnew[2] = bnew[2] + c_vec[2] #need to add to c_vec
    line = str(a_vec[0]) + ', '  + str(a_vec[1]) + ', '  + str(a_vec[2]) + ', ' + \
            str(bnew[0]) + ', '  + str(bnew[1]) + ', '  + str(bnew[2]) +', ' + \
            str(c_vec[0]) + ', '  + str(c_vec[1]) + ', '  + str(c_vec[2])
    return line
    
def rearrange_el_nodes(line,sline):
    ############# NOT USED #########################
    #this function is not used yet - still doesn't work correctly on the skipping lines method.
    cont_bool = False
    if flags['n6_flag']:
        s = line.rstrip().split(',')
        b = [0,4,5,6,1,2,3] #the new node ordering
        rearranged = map(operator.getitem,[s]*len(b),b)
        line = "    " + ",    ".join(rearranged)
    elif flags['n3_flag']:
        s = line.rstrip().split(',')
        b = [0,1,3,2] #the new node ordering
        rearranged = map(operator.getitem,[s]*len(b),b)
        line = "    " + ",    ".join(rearranged)    
    elif flags['n4_flag']:
        s = line.rstrip().split(',')
        b = [0,1,4,3,2] #the new node ordering
        rearranged = map(operator.getitem,[s]*len(b),b)
        line = "    " + ",    ".join(rearranged)                            
    elif flags['n8_flag']:
        row = filter(len,line.strip().split(',')) #filter out entries with len = 0
        sline = sline + row
        if len(sline) < 9:
            cont_bool = True
            #continue #store ids, continue to next line
        elif len(sline) == 9:
            b = [0,5,6,7,8,1,2,3,4] #the new node ordering. 0 is the element ID
            rearranged = map(operator.getitem,[sline]*len(b),b)
            line = "    " + ",    ".join(rearranged)
            sline = [] #reset list
        elif len(sline) > 9:
            sys.stderr.write('Error with SC8R elements.\n')
            sys.exit()    
    return line, cont_bool
    
if len(sys.argv) < 2:
    print 'Python Abaqus model reflecting script'
    print 'usage: python reflect.py [folder]'
    sys.exit()

src_dir = sys.argv[1]

dst_dir = src_dir + '-RH'
shutil.copytree(src_dir,dst_dir)         #copy the folder structure
path = dst_dir

#Regular expressions
instre = re.compile(r'\*instance', re.IGNORECASE)
partre = re.compile(r'\*part', re.IGNORECASE)
part_re = re.compile('part=\w+', re.IGNORECASE)
name_re = re.compile('name=\S+,', re.IGNORECASE)
include_re = re.compile(r'\*include', re.IGNORECASE)
transform_re = re.compile(r'\*transform', re.IGNORECASE)
keyword_re = re.compile('^\*\w')
inst_cross_re = re.compile('instance=\S+', re.IGNORECASE)
lh_search_re = re.compile('LH_|LH\-', re.IGNORECASE)

instancenames = []
partnames = []
totalfiles = 0
cload_m = [ '', '-', '', '-', '', '-']

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
                        #getkeyword_params(line)
                    elif line.strip()[0:2] == '**':
                         pass #do nothing, this is a comment
                    elif flags['orientflag']:
                        counter = counter + 1
                        if (counter == 1) and (len(line.split(','))==9):
                            line = reflect_orient(line) #modify the orientation coordinates to x, -y
                    elif flags['tx_flag']:
                        counter = counter + 1
                        if (counter == 1) and (len(line.split(','))==6):
                            line = line + ', 0.0 , 0.0, 0.0' #add the origin vector
                            line = reflect_orient(line) #modify the orientation coordinates to x, -y
                            line = ",    ".join(line.split(',')[:-3])
                    elif flags['elementflag']:
                        if flags['n6_flag']:
                            #note: 456,123 results in positive elements but surfaces are reversed
                            s = line.rstrip().split(',')
                            b = [0,2,1,3,5,4,6] #the new node ordering
                            rearranged = map(operator.getitem,[s]*len(b),b)
                            line = "    " + ",    ".join(rearranged)
                        elif flags['n3_flag']:
                            s = line.rstrip().split(',')
                            b = [0,1,3,2] #the new node ordering
                            rearranged = map(operator.getitem,[s]*len(b),b)
                            line = "    " + ",    ".join(rearranged)    
                        elif flags['n4_flag']:
                            s = line.rstrip().split(',')
                            b = [0,1,4,3,2] #the new node ordering
                            rearranged = map(operator.getitem,[s]*len(b),b)
                            line = "    " + ",    ".join(rearranged)
                        elif flags['tet_flag']:
                            s = line.rstrip().split(',')
                            b = [0,1,3,2,4] #the new node ordering
                            rearranged = map(operator.getitem,[s]*len(b),b)
                            line = "    " + ",    ".join(rearranged)
                        elif flags['n8_flag']:
                            #notes: 5678,1234 results in positive elements but surfaces messed up
                            row = filter(len,line.strip().split(',')) #filter out entries with len = 0
                            sline = sline + row
                            if len(sline) < 9:
                                continue #store ids, continue to next line
                            elif len(sline) == 9:
                                b = [0,2,1,4,3,6,5,8,7] #the new node ordering. 0 is the element ID
                                rearranged = map(operator.getitem,[sline]*len(b),b)
                                line = "    " + ",    ".join(rearranged)
                                sline = [] #reset list
                            elif len(sline) > 9:
                                sys.stderr.write('Error with SC8R elements.\n')
                                sys.exit()
                ############################################################################            
                    elif flags['nodeflag']:
                        node_c = [x.strip() for x in line.split(',')] #map(float,line.split(','))
                        if node_c[2][0] == '-':
                            node_c[2] = node_c[2][1:] #remove negative sign
                        else:
                            node_c[2] = '-' + node_c[2]         #reflect node around x axis
                        #node_c[0] = int(node_c[0]) #node label has to be an integer
                        line = "    " + ",    ".join(node_c)
                    elif flags['cload_flag']:
                        cload_dir = int(line.split(',')[1])
                        load_mag = line.split(',')[2].strip()
                        if load_mag[0] != '-':
                            load_mag = cload_m[cload_dir-1] + load_mag #reverse the cload direction (assuming reflection around global x-axis)
                        else:
                            load_mag = cload_m[(cload_dir+1)%6] + load_mag
                        line = line.split(',')[0] + ',' + str(cload_dir) + ',' + load_mag
                ############################################################################
                    if lh_search_re.search(line):
                        line = line.replace('LH', 'RH')
                ############################################################################
                #----------------------------Finite State Machine end ----------------------------
                print line.rstrip()
  
path = dst_dir #reset the path
                
for (path, dirs, files) in os.walk(path):
    for file in files:
        if '.inp' in file:
            newname = file.replace('LH','RH')
            os.rename(os.path.join(path,file),os.path.join(path,newname))
            sys.stderr.write('renaming part/instance names: ' + newname + '\n')            
            
sys.stderr.write('total files processed: ' + str(totalfiles)+ '\n')   
