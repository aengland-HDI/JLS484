###############################################################################
##
##    Goals: Retirve data from all the mesh tallies and plot dy, dz
##              as a function of smoothness
##
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from tabulate import tabulate
import pandas as pd
import pickle
from prettytable import PrettyTable
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

def read_data_to_table(path, files, region):
    if region == "Full Model": 
        RE = 1
    elif region == "Beam Chamber":
        RE = 0 
    tally_number = []
    line_number = []
    mesh_tally_whole = []
    indexes_start = [5,6,7,11]
    with open(path+files) as file:
        for num, line in enumerate(file, 1):
            ## determine location of mesh tallies
            if "Mesh Tally Number" in line:
                line_number.append(num)
                tally_number.append(float(re.split(" ", line)[-1][0:-1]))
            mesh_tally_whole.append(line)
        ## determine which tally to read
        list_x, list_y, list_z, list_tally, list_error = [], [], [], [], []
        if region == "Full Model":

            X, Y, Z = [],  [] , []
            for i in mesh_tally_whole[line_number[1]+indexes_start[3]::]:
                nums = [float(j) for j in i.split()]
                list_x.append(nums[1])
                list_y.append(nums[2])
                list_z.append(nums[3])
                list_tally.append(nums[4])
                list_error.append(nums[5])
                if X == [] or X[-1] != nums[1]:
                    X.append(nums[1])
                elif Y == [] or Y[-1] != nums[2]:
                    Y.append(nums[2])
                elif Z == [] or Z[-1] != nums[3]:
                    Z.append(nums[3])
            df_x, df_y, df_z = pd.DataFrame(X), pd.DataFrame(Y), pd.DataFrame(Z)
            df = pd.DataFrame(columns = ["X-Vals", "Y-Vals", "Z-Vals", "X", "Y", "Z", "Tally", "Error"])
            df["X"], df["Y"], df["Z"], df["Tally"], df["Error"] = list_x, list_y, list_z, list_tally, list_error
            total = pd.concat([df_x, df_y, df_z, df], axis=1)
            total.to_csv(path+region+".csv")


        elif region == "Beam Chamber":

            X, Y, Z = [],  [] , []
            for i in mesh_tally_whole[line_number[0]+indexes_start[3]:line_number[1]-2]:
                nums = [float(j) for j in i.split()]
                list_x.append(nums[1])
                list_y.append(nums[2])
                list_z.append(nums[3])
                list_tally.append(nums[4])
                list_error.append(nums[5])
                if X == [] or X[-1] != nums[1]:
                    X.append(nums[1])
                if Y == [] or Y[-1] != nums[2]:
                    Y.append(nums[2])
                if Z == [] or Z[-1] != nums[3]:
                    Z.append(nums[3])

            ylocs = np.argmin(np.array(Y[1::])-Y[0])+1
            zlocs = np.argmin(np.array(Z[1::])-Z[0])+1

            Y,Z = Y[0:ylocs+1], Z[0:zlocs+1]


            df_x, df_y, df_z = pd.DataFrame(X), pd.DataFrame(Y), pd.DataFrame(Z)
            df = pd.DataFrame(columns = ["X-Vals", "Y-Vals", "Z-Vals", "X", "Y", "Z", "Tally", "Error"])
            df["X"], df["Y"], df["Z"], df["Tally"], df["Error"] = list_x, list_y, list_z, list_tally, list_error
            total = pd.concat([df_x, df_y, df_z, df], axis=1)
            total.to_csv(path+region+".csv")

def X_CUTS(filepath, out):
    df = pd.read_csv(filepath, dtype=float).to_numpy()
    I_X, I_Y, I_Z = df[~np.isnan(df[:,1]), 1], df[~np.isnan(df[:,2]), 2], df[~np.isnan(df[:,3]), 3]
    
    Tallies_X, Error_X = np.zeros((len(I_X), len(I_Z), len(I_Y))), np.zeros((len(I_X), len(I_Z), len(I_Y)))
    Tallies_X[:], Error_X[:] = np.nan, np.nan
    for i in range(len(df[:,11])):
        X_index = np.where(np.isclose(float(df[i,7]), I_X, rtol = 0.0001))[0][0]
        Y_index = np.where(np.isclose(float(df[i,8]), I_Y, rtol = 0.0001))[0][0]
        Z_index = np.where(np.isclose(float(df[i,9]), I_Z, rtol = 0.0001))[0][0]
        Tallies_X[X_index, Z_index, Y_index] = df[i, 10]
        Error_X[X_index, Z_index, Y_index] = df[i, 11]
    

    
    

path = "X:\\Operations\\ProjectsEng\\HDI\\SO 3887 JLS 484 Replacement\\Radiation Physics\\MCNP\\Version 10\\"
file = "JLS484_V10_Slave3.imsht"
# version = 10.1
# read_data_to_table(path, file, "Beam Chamber")

X_CUTS(path+"Beam Chamber.csv", 1)