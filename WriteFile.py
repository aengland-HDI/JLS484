import numpy as np 
import os
import glob


## User Inputs

Path_To_Folder = "C:\\Users\\alex.england\\Documents\\JLS484\\Version10"


## Retrieve Master File
Master = glob.glob(Path_To_Folder+"\\*Master.i")
## Create New Slave File
split_filename = os.path.basename(Master[0]).split("_")
Model, Version = split_filename[0], split_filename[1]
Master = open(Master[0], 'r')
slave  = open(Path_To_Folder+"\\"+ Model+"_"+Version+"_"+"Slave.i", 'w')
Master = Master.readlines()
############## Retrieving Cards ##############

def retrieve_section(Master, section):
    Flags = [False, False]
    Lines = ['', ""]
    i= 0
    for line in Master:
        if Flags[0] != True and section in line:
            Flags[0] = True
            Lines[0] = i+5
        if line.isspace():
            Flags[1] = True
            Lines[1] = i-1
            break
        i = i+1
    # print("Section Starts and Ends on lines:  %i and %i"%(Lines[0], Lines[1]))
    return Lines

############## Changing Densities ############## 

def new_density(Master, slave):
    for line in Master[:cell_lines[0]]:
        slave.write(line)

    for line in Master[cell_lines[0]+1:cell_lines[1]-1]:
        l = line.split()
        ## Check if line is commented or numbered
        if l[0].isdigit():
            l[2] = str(float(l[2])*multiplier)
            slave.write(' '.join(l)+'\n')
        else:
            slave.write(line)

############## Weight Windows ##############




section = "c                       Cell Cards"
cell_lines = retrieve_section(Master,section)
multiplier = 0.5
new_density(Master, slave)

section = "c                       Surface Cards"
surface_lines = retrieve_section(Master,section)

data_lines = 'c                       Data Cards'
data_lines = retrieve_section(Master, section);

for line in Master[cell_lines[1]:]:
    slave.write(line)





############## Weight Window Generator ############## 




############## Batch Scipt ############## 



