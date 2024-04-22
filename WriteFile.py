import numpy as np 
import os
import glob


## User Inputs


Path_To_Folder = "C:\\Users\\alex.england\\Documents\\JLS484\\Version10"
Master_Batch = "C:\\Users\\alex.england\\Documents\\JLS484\\Master.batch"

n_red = 4

new_seam = 10


M_batch  = open(Master_Batch, 'r')
S_Batch = open(Path_To_Folder+"\\"+"Slave.bat", 'w')

## Retrieve Master File
Master = glob.glob(Path_To_Folder+"\\*Master.i")
## Create New Slave File
split_filename = os.path.basename(Master[0]).split("_")
Model, Version = split_filename[0], split_filename[1]
Master = open(Master[0], 'r')
Master = Master.readlines()

############## Retrieving Cards ##############

def retrieve_section(Master, section, kill, step):
    Flags = [False, False]
    Lines = ['', ""]
    i= 0
    for line in Master:
        if Flags[0] != True and section in line:
            Flags[0] = True
            Lines[0] = i+step
        if Flags[0] and kill[0] and line.isspace():
            Flags[1] = True
            Lines[1] = i-1
        if Flags[0] and kill[0] != True and kill[1] in line :
            Flags[1] = True
            Lines[1] = i-1
        if Flags[0] and Flags[1]:
            break
        i = i+1
    # print("Section Starts and Ends on lines:  %i and %i"%(Lines[0], Lines[1]))
    return Lines

############## Changing Densities ############## 

def new_density(Master, slave, multiplier):
    for line in Master[:cell_lines[0]]:
        slave.write(line)

    for line in Master[cell_lines[0]:cell_lines[1]]:
               ## Check if line is commented or numbered
        l = line.split()
        if line[0].isdigit() != True:
            slave.write(line)
        elif l[0].isdigit(): 
            l[2] = '%.6f'%(float(l[2])*multiplier)
            slave.write(' '.join(l)+'\n')
    for line in Master[cell_lines[1]:cell_lines[1]+3]:
        slave.write(line)
            

############## Changing main seam allowance ##############

def change_seam(dist):
    for line in Master[surface_lines[0]-6:seam_lines[0]]:
        slave.write(line)

    for line in Master[seam_lines[0]:seam_lines[1]-2]:
        line = line.split()
        
        if line[0].isdigit():
            type = line[1]
            cm = 0.00254 # convert thou inch to cm
            if type == "P":
                line[2] = str(round(float(line[2]) + dist*cm - 0.1, 3))
                line[5] = str(round(float(line[5]) + dist*cm - 0.1, 3))
                line[8] = str(round(float(line[8]) + dist*cm - 0.1,3))
            elif type == "RPP":
                line[2] = str(round(float(line[2]) + dist*cm - 0.1,3))
                line[3] = str(round(float(line[3]) + dist*cm - 0.1,3))
            elif type == "RCC":
                line[2] = str(round(float(line[2]) + dist*cm - 0.1,3))
            else:
                print('The Surface is none of these')
        
        slave.write(' '.join(line)+'\n')

############## Weight Windows/Tallies ##############
def tallies_WWG(Variance_Reduction, Tallies, variance_lines, tally_lines, first):
    for line in Master[seam_lines[1]:variance_lines[0]]:
            slave.write(line)
    if Variance_Reduction and first:
        for line in Master[variance_lines[0]:tally_lines[0]-5]:
            slave.write(line[2:])
    else: # Variance_Reduction:
        for line in Master[variance_lines[0]:tally_lines[0]-3]:
            slave.write(line[2:])
    if Tallies:
        for line in Master[tally_lines[0]-1:]:
            slave.write(line[2:])   

kill = [True, 'c']
section = "c                       Cell Cards"
cell_lines = retrieve_section(Master,section, kill, step=5)

section = "c                       Surface Cards"
surface_lines = retrieve_section(Master,section, kill, step=5)

section = 'c                       Data Cards'
variance= 'c                   Variance Reduction'
tally= 'c                   Tallies'
data_lines = retrieve_section(Master, section, kill,step=3)
variance_lines = retrieve_section(Master, variance, ["",tally], 2)
tally_lines = retrieve_section(Master, tally, kill ,2)


############## Changing main seam allowance ##############
Master_Seam = 1 #mm

seam_flag = "c       Secondary Shield"
seam_lines = retrieve_section(Master[surface_lines[0]:], seam_flag, kill, step=2)
seam_lines[0], seam_lines[1] = seam_lines[0] + surface_lines[0],  seam_lines[1] + surface_lines[0]


############## Action ##############

# number of reductions
n_red = 4
M_batch  = open(Master_Batch, 'r')
S_Batch = open(Path_To_Folder+"\\"+"Slave.bat", 'w')

for i in range(n_red):
    slave  = open(Path_To_Folder+"\\"+ Model+"_"+Version+"_"+"Slave"+str(i)+".i", 'w')
    new_density(Master, slave, (i+1)/n_red)
    change_seam(new_seam)
    if i+1 == n_red:
        tallies_WWG(True, True, variance_lines, tally_lines, False)
    elif i == 0:
        tallies_WWG(True, False, variance_lines, tally_lines, True)
    else:
        tallies_WWG(True, False, variance_lines, tally_lines, False)
    for line in M_batch.readlines():
        S_Batch.write(line)
    if i == 0:
        S_Batch.write('\nmcnp6 name = %s tasks 26'%(Model+"_"+Version+"_"+"Slave"+str(i)+".i"))
    elif i>0:
        S_Batch.write('\nmcnp6 name = %s wwinp=%s tasks 26'%(Model+"_"+Version+"_"+"Slave"+str(i)+".i",Model+"_"+Version+"_"+"Slave"+str(i-1)+".ie" ))

