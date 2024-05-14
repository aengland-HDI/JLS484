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
from pathlib import Path
import glob

X, Y, Z = 102, 26, 28

indices_X = np.around(np.linspace(0.5, 101.5, 102), decimals=3)
indices_Y = np.around(np.linspace(-12.822, 12.822, 26), decimals=3)
indices_Z = np.around(np.linspace(-14.083, 14.083, 28), decimals=3)

def calc_DUR(min, max):
    return max/min

def convert_beam(data, mult, material):
    ## From Johns and Cunningham
    ## 1 Roentgen = 0.869 rads (air)
    ## D' = D_Air x A_eq x f_med
    ## Not going to calc. A_eq and f_med is ratio of mass-energy atten of media
    mass_atten_air = 0.9988*np.interp(1.1732, [1, 1.25, 1.5], [2.789E-02, 2.666E-02, 2.547E-02])+np.interp(1.3325, [1, 1.25, 1.5], [2.789E-02, 2.666E-02, 2.547E-02])
    mass_atten_silicon = 0.9988*np.interp(1.1732, [1, 1.022, 1.25, 1.5], [6.361E-02, 6.293E-02, 5.688E-02, 5.183E-02])+np.interp(1.3325, [1, 1.022, 1.25, 1.5], [6.361E-02, 6.293E-02, 5.688E-02, 5.183E-02])
    mass_atten_water = 0.9988*np.interp(1.1732,  [1, 1.022, 1.25, 1.5], [7.072E-02, 6.997E-02, 6.323E-02, 5.754E-02])+np.interp(1.3325,  [1, 1.022, 1.25, 1.5], [7.072E-02, 6.997E-02, 6.323E-02, 5.754E-02])
    if material == "water":
        ratio = mass_atten_water/mass_atten_air
    elif material == "silicon":
        ratio = mass_atten_silicon/mass_atten_air
    result_rad = list(np.array(data)*ratio*0.869*mult)
    result_Gray = list(np.array(data)*ratio*mult*0.869*1E-2)
    return result_rad, result_Gray

def read_data_to_table(path):
    ## initialization variables
    for file in glob.glob(path+"*.imsht"):
        fname = os.path.splitext(os.path.basename(file))[0]

        ## Setting up Data in Np.Array 
        TALLY_VALUES = np.zeros((X, Z, Y))
        REL_ERROR = np.zeros((X, Z, Y))
        MT_LINE=100000

        ## Looping over lines in the mesh file
        for i,line in enumerate(open(file, 'r').readlines()):
            if line.find('Mesh Tally Number') != -1:
                print("*****  READING MESH TALLY  *****")
                MT_LINE = i
            if i > MT_LINE+12:
                DAT = line.split()
                X_index = np.where(np.isclose(float(DAT[1]), indices_X, rtol = 0.01))[0][0]
                Y_index = np.where(np.isclose(float(DAT[2]), indices_Y, rtol = 0.01))[0][0]
                Z_index = np.where(np.isclose(float(DAT[3]), indices_Z, rtol = 0.01))[0][0]
                Tally_Value = float(DAT[4])
                REL_ERROR_VALUE = float(DAT[5])

                TALLY_VALUES[X_index, Z_index, Y_index] = Tally_Value
                REL_ERROR[X_index, Z_index, Y_index] = REL_ERROR_VALUE

        np.save(path+"%s_TallyValues"%(fname), TALLY_VALUES )
        np.save(path+"%s_TallyError"%(fname), REL_ERROR )

def plot_beam_YZ_ofX(data_set, X, save_directory):
    # fname = os.path.splitext(os.path.basename(data_set))[0]
    # y_dist = fname.split("-")[1].split("_")[0]
    y_dist = "5.334"
    fname = "RUSS"
    X_index = np.where(np.isclose(float(X), indices_X, rtol = 0.01))[0][0]
    beam = np.load(data_set)
    beam_rad, beam_Gy = convert_beam(beam, 1E-3, "silicon")
    
    print("------------------------------------------------------------------------\n")
    print("     MAX EXPOSURE is: %.3f R/hr"%np.max(beam[X_index]*1E-3))
    print("     MAX DOSE RATE is: %.3f rad(Si)/hr"%np.max(beam_rad[X_index]))
    print("     MAX DOSE RATE is: %.3f Gy(Si)/hr\n"%np.max(beam_Gy[X_index]))
    print("     DUR: %.3f"%(np.nanmax(beam[X_index])/np.nanmin(beam[X_index])))
    print("------------------------------------------------------------------------")


    fig, ax = plt.subplots()
    im = ax.imshow(beam[X_index], origin='lower')
    
    plt.colorbar(im, label="Exposure Rate (mR/hr)")
    ax.set_xticks(np.linspace(0, 25, 6),np.linspace(-12.5, 12.5, 6))
    ax.set_yticks(np.linspace(0, 27, 6),np.linspace(-15, 15, 6))
    ax.vlines(12.5, 0, 27, color="r", linestyles="dashed", linewidths=1.0)
    ax.hlines(13.5, 0, 25, color="r", linestyles="dashed", linewidths=1.0)

    ax.set_xlabel("Distance from Y-Centerline (cm)")
    ax.set_ylabel("Distance from Z-Centerline (cm)")
    ax.set_title("Beam Profile at %.1f from Sources\n Y-Spacing:%s cm"%(X, y_dist))
    fig.savefig(save_directory+"BeamProfileExposure_%.1f_%s.jpg"%(X, fname))


    fig, ax = plt.subplots()
    im = ax.imshow(beam_rad[X_index], origin='lower')
    
    plt.colorbar(im, label="Dose Rate [rads(Si)/hr]")
    ax.set_xticks(np.linspace(0, 25, 6),np.linspace(-12.5, 12.5, 6))
    ax.set_yticks(np.linspace(0, 27, 6),np.linspace(-15, 15, 6))
    ax.vlines(12.5, 0, 27, color="r", linestyles="dashed", linewidths=1.0)
    ax.hlines(13.5, 0, 25, color="r", linestyles="dashed", linewidths=1.0)

    ax.set_xlabel("Distance from Y-Centerline (cm)")
    ax.set_ylabel("Distance from Z-Centerline (cm)")
    ax.set_title("Beam Profile at %.1f from Sources\n Y-Spacing:%s cm"%(X, y_dist))
    fig.savefig(save_directory+"BeamProfileRADS_%.1f_%s.jpg"%(X, fname))

    fig, ax = plt.subplots()
    im = ax.imshow(beam_Gy[X_index], origin='lower')
    
    plt.colorbar(im, label="Dose Rate [Gy(Si)/hr]")
    ax.set_xticks(np.linspace(0, 25, 6),np.linspace(-12.5, 12.5, 6))
    ax.set_yticks(np.linspace(0, 27, 6),np.linspace(-15, 15, 6))
    ax.vlines(12.5, 0, 27, color="r", linestyles="dashed", linewidths=1.0)
    ax.hlines(13.5, 0, 25, color="r", linestyles="dashed", linewidths=1.0)

    ax.set_xlabel("Distance from Y-Centerline (cm)")
    ax.set_ylabel("Distance from Z-Centerline (cm)")
    ax.set_title("Beam Profile at %.1f from Sources\n Y-Spacing:%s cm"%(X, y_dist))
    fig.savefig(save_directory+"BeamProfileGRAYS_%.1f_%s.jpg"%(X, fname))


    ### Chop down to size of film: 15.25 x 20.32 cm
    Z_1, Z_2 = np.where(np.isclose(-7.5, indices_Z, rtol = 0.06))[0][0], np.where(np.isclose(7.5, indices_Z, rtol = 0.06))[0][0]
    Y_1, Y_2 = np.where(np.isclose(-10, indices_Z, rtol = 0.06))[0][0], np.where(np.isclose(10, indices_Z, rtol = 0.06))[0][0]
    
    fig, ax = plt.subplots()
    im = ax.imshow(beam_rad[X_index][Z_1:Z_2][Y_1:Y_2], origin='lower')
    plt.colorbar(im, label="Dose Rate [Gy(Si)/hr]")
    ax.set_xticks(np.linspace(0, 25, 6),np.linspace(-12.5, 12.5, 6))
    ax.set_yticks(np.linspace(0, 10, 6),np.linspace(-15, 15, 6))
    ax.vlines(12.5, -0.5, 10.5, color="r", linestyles="dashed", linewidths=1.0)
    ax.hlines(5, -0.5, 25.5, color="r", linestyles="dashed", linewidths=1.0)

    ax.set_xlabel("Distance from Y-Centerline (cm)")
    ax.set_ylabel("Distance from Z-Centerline (cm)")
    ax.set_title("Film Profile at %.1f from Sources\n Y-Spacing:%s cm"%(X, y_dist))
    fig.savefig(save_directory+"FILMProfileGRAYS_%.1f_%s.jpg"%(X, fname))

path_L4 = "X:\\Operations\\ProjectsEng\\HDI\\SO 3887 JLS 484 Replacement\\Radiation Physics\\MCNP\\Beam_Profile-L4\\"
path_L6 = "X:\\Operations\\ProjectsEng\\HDI\\SO 3887 JLS 484 Replacement\\Radiation Physics\\MCNP\\Beam_Profile-L6\\"
path_RUSS = "X:\\Operations\\ProjectsEng\\HDI\\SO 3887 JLS 484 Replacement\\Radiation Physics\\MCNP\\Beam_Profile-RUSS\\"
path_4x4 = "X:\\Operations\\ProjectsEng\\HDI\\SO 3887 JLS 484 Replacement\\Radiation Physics\\MCNP\\Beam_Profile-4x4\\"

# read_data_to_table(path_L4)
# read_data_to_table(path_L6)
# read_data_to_table(path_RUSS)
# read_data_to_table(path_4x4)

Processed_L4 = "X:\\Operations\\ProjectsEng\\HDI\\SO 3887 JLS 484 Replacement\\Radiation Physics\\MCNP\\Beam_Profile-L4\\Processed\\"
Processed_L6 = "X:\\Operations\\ProjectsEng\\HDI\\SO 3887 JLS 484 Replacement\\Radiation Physics\\MCNP\\Beam_Profile-L6\\Processed\\"
Processed_RUSS = "X:\\Operations\\ProjectsEng\\HDI\\SO 3887 JLS 484 Replacement\\Radiation Physics\\MCNP\\Beam_Profile-RUSS\\Processed\\"
Processed_4x4 = "X:\\Operations\\ProjectsEng\\HDI\\SO 3887 JLS 484 Replacement\\Radiation Physics\\MCNP\\Beam_Profile-4x4\\Processed\\"

# for i in [10.5, 17.5, 30.5,50.5,  70.5, 90.5]:
#     print("Starting %.1f"%i)
#     plot_beam_YZ_ofX(path_L4+"L4-17_TallyValues.npy", i, Processed_L4+"17 cm\\")

# for i in [10.5, 17.5, 30.5,50.5,  70.5, 90.5]:
#     print("Starting %.1f"%i)
#     plot_beam_YZ_ofX(path_L6+"L6-7.0_TallyValues.npy", i, Processed_L6+"7.0 cm\\")

## For RUSS change lines 69-72 
# for i in [10.5, 17.5, 30.5,50.5,  70.5, 90.5]:
#     print("Starting %.1f"%i)
#     plot_beam_YZ_ofX(path_RUSS+"RUSS_TallyValues.npy", i, Processed_RUSS)

# for i in [10.5, 17.5, 30.5,50.5,  70.5, 90.5]:
#     print("Starting %.1f"%i)
#     plot_beam_YZ_ofX(path_4x4+"4x4_TallyValues.npy", i, Processed_4x4)
