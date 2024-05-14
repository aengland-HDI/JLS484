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

# file = "C:\\Users\\alex.england\\Documents\\Project-3887\\Beam_Profile\\Beam_Profile_4.50_1.50.imsht"
# path = "C:\\Users\\alex.england\\Documents\\Project-3887\\Beam_Profile"
directory = "C:\\Users\\alex.england\\Documents\\Project-3887\\Server_Runs\\Beam_Profile\\Beam_Profile"

Save_Tallies = "C:\\Users\\alex.england\\Documents\\Project-3887\\Server_Runs\\Beam_Profile\\Tally_Values\\"
Save_Error = "C:\\Users\\alex.england\\Documents\\Project-3887\\Server_Runs\\Beam_Profile\\Error_Values\\"
Processed_Data = "C:\\Users\\alex.england\\Documents\\Project-3887\\Server_Runs\\Beam_Profile\\Processed_Data\\"

## Uncomment the following if you need to load the data

X, Y, Z = 102, 26, 28

indices_X = np.around(np.linspace(0.5, 101.5, 102), decimals=3)
indices_Y = np.around(np.linspace(-12.822, 12.822, 26), decimals=3)
indices_Z = np.around(np.linspace(-14.083, 14.083, 28), decimals=3)


## Want to plot the data such that it does the following:
## Animate how the spacing changes the dose dy dz
## Animate going down the beam tunnel


def calc_ratio(min, max):
    return max/min


## animate the beam line
def animate_beam(beam, filename):
    fig, ax = plt.subplots()

    def animate(i):
        im = ax.imshow(beam[i])
        ax.set_title("Sources spereated by x and y\n%.2f cm from center of sources"%indices_X[i])

    ani = animation.FuncAnimation(fig, animate, interval=100, frames=len(indices_X))
    ani.save(filename)
    plt.show()

def convert_exposure_dose(data, mult, material):
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
    dy, dz = [],[]
    filename_pattern = re.compile(r'[-+]?\d*\.\d+|\d+')
    for filename in os.listdir(directory):
        ## Pull information from filename, index 0 is dy and index 1 is dz
        floats = filename_pattern.findall(filename)
        dy = float(floats[0])
        dz = float(floats[1])

        ## Setting up Data in Np.Array 
        TALLY_VALUES = np.zeros((X, Z, Y))
        REL_ERROR = np.zeros((X, Z, Y))
        MT_LINE=100000

        ## Looping over lines in the mesh file
        for i,line in enumerate(open(directory+"\\"+ filename, 'r').readlines()):
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

                np.save(Save_Tallies+"%.2f-%.2f_TallyValues"%(dy, dz), TALLY_VALUES )
                np.save(Save_Error+"%.2f-%.2f_TallyError"%(dy, dz), REL_ERROR )




## Need to to determine DUR for each file and create a animated plot for that
## Required to run read_data_to_table prior to this function
def beam_uniformity_multiFiles(directory, j_dist, save):

    indices_dy = np.arange(4.5, 24.5, 1)
    indices_dz = np.arange(1.5, 23.5, 1)
    DUR_mean = np.empty((X, len(indices_dz), len(indices_dy)))
    DUR_mean[:]=np.nan
    Delta = np.empty((X, len(indices_dz), len(indices_dy)))
    Delta[:]=np.nan

    for filename in os.listdir(Save_Tallies):
        split = re.findall(r'[0-9]+', filename)
        dy = split[0]+"."+split[1]
        dz = split[2]+"."+split[3]
        DY_index = np.where(np.isclose(float(dy), indices_dy, rtol = 0.01))[0][0]
        DZ_index = np.where(np.isclose(float(dz), indices_dz, rtol = 0.01))[0][0]
        PATH = Save_Tallies+filename
        try:
            beam = np.load(PATH)
        except EOFError as e:
            print(e,PATH)
        ## This is worse uniformity
        for i in range(X):
            DUR = np.nanmax(np.nanmax(beam[i])/(beam[i]))
            DUR_mean[i, DZ_index, DY_index] = DUR


    ## determine the best beam uniformity ratios for 
    print("------------------------------------------------------------------------\n")
    print("     MIN DUR:%.3f"%np.nanmin(DUR_mean[j_dist]))
    print("     MAX DUR:%.3f"%np.nanmax(DUR_mean[j_dist]))
    print("\n------------------------------------------------------------------------\n")


    fig, ax = plt.subplots()

    im = ax.imshow(DUR_mean[0], origin="lower", )
    ax.set_xlabel("Distance between Center of Sources (cm)")
    ax.set_ylabel("Distance between Top and Bottom of Sources (cm)")

    def animate(i):
        ax.clear()
        im = ax.imshow(DUR_mean[i], origin="lower", )
        max, min = round(np.nanmax(DUR_mean[i]),2), round(np.nanmin(DUR_mean[i]),2)
        ax.set_title("Dose Uniformity Ratios(DUR) at %.2f from Source\nMIN DUR: %.2f   MAX DUR:%.2f"%(indices_X[i], min, max))
        ax.set_xlabel("Distance between Center of Sources (cm)")
        ax.set_ylabel("Distance between Top and Bottom of Sources (cm)")

    ani = animation.FuncAnimation(fig, animate, interval=250, frames=50)
    ani.save(save+"DUR.gif")
    plt.show()
    plt.close()



def plot_beam_YZ_ofX(data_set, X, save_directory):
    split = re.findall(r'[0-9]+', data_set)[1::]
    dy = float(split[0]+"."+split[1])
    dz = float(split[2]+"."+split[3])
    X_index = np.where(np.isclose(float(X), indices_X, rtol = 0.01))[0][0]
    beam = np.load(data_set)
    beam_rad, beam_Gy = convert_exposure_dose(beam, 1E-3, "silicon")
    
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
    ax.set_title("Beam Profile at %.1f from Sources\n Y-Spacing:%.1f cm  Z-Spacing:%.1f cm"%(X, dy, dz))
    fig.savefig(save_directory+"BeamProfileExposure_%.1f_%.1f_%.1f.jpg"%(X, dy, dz))


    fig, ax = plt.subplots()
    im = ax.imshow(beam_rad[X_index], origin='lower')
    
    plt.colorbar(im, label="Dose Rate [rads(Si)/hr]")
    ax.set_xticks(np.linspace(0, 25, 6),np.linspace(-12.5, 12.5, 6))
    ax.set_yticks(np.linspace(0, 27, 6),np.linspace(-15, 15, 6))
    ax.vlines(12.5, 0, 27, color="r", linestyles="dashed", linewidths=1.0)
    ax.hlines(13.5, 0, 25, color="r", linestyles="dashed", linewidths=1.0)

    ax.set_xlabel("Distance from Y-Centerline (cm)")
    ax.set_ylabel("Distance from Z-Centerline (cm)")
    ax.set_title("Beam Profile at %.1f from Sources\n Y-Spacing:%.1f cm  Z-Spacing:%.1f cm"%(X, dy, dz))
    fig.savefig(save_directory+"BeamProfileRADS_%.1f_%.1f_%.1f.jpg"%(X, dy, dz))

    fig, ax = plt.subplots()
    im = ax.imshow(beam_Gy[X_index], origin='lower')
    
    plt.colorbar(im, label="Dose Rate [Gy(Si)/hr]")
    ax.set_xticks(np.linspace(0, 25, 6),np.linspace(-12.5, 12.5, 6))
    ax.set_yticks(np.linspace(0, 27, 6),np.linspace(-15, 15, 6))
    ax.vlines(12.5, 0, 27, color="r", linestyles="dashed", linewidths=1.0)
    ax.hlines(13.5, 0, 25, color="r", linestyles="dashed", linewidths=1.0)

    ax.set_xlabel("Distance from Y-Centerline (cm)")
    ax.set_ylabel("Distance from Z-Centerline (cm)")
    ax.set_title("Beam Profile at %.1f from Sources\n Y-Spacing:%.1f cm  Z-Spacing:%.1f cm"%(X, dy, dz))
    fig.savefig(save_directory+"BeamProfileGRAYS_%.1f_%.1f_%.1f.jpg"%(X, dy, dz))


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
    ax.set_title("Film Profile at %.1f from Sources\n Y-Spacing:%.1f cm  Z-Spacing:%.1f cm"%(X, dy, dz))
    fig.savefig(save_directory+"FILMProfileGRAYS_%.1f_%.1f_%.1f.jpg"%(X, dy, dz))

def max_dose_configs(directory, X_dist, Processed_Data):
    indices_dy = np.arange(4.5, 24.5, 1)
    indices_dz = np.arange(1.5, 23.5, 1)
    X_index = np.where(np.isclose(float(X_dist), indices_X, rtol = 0.01))[0][0]
    DOSE = np.zeros((3, len(indices_dz), len(indices_dy)))
    DOSE[:] = np.nan
    for filename in os.listdir(Save_Tallies):
        split = re.findall(r'[0-9]+', filename)
        dy = split[0]+"."+split[1]
        dz = split[2]+"."+split[3]
        DY_index = np.where(np.isclose(float(dy), indices_dy, rtol = 0.01))[0][0]
        DZ_index = np.where(np.isclose(float(dz), indices_dz, rtol = 0.01))[0][0]
        PATH = Save_Tallies+filename
        try:
            beam = np.load(PATH)
        except EOFError as e:
            print(e,PATH)

        DOSE[0, DZ_index, DY_index] = np.max(beam[X_index])*1E-3
        rad_max, gray_max = convert_exposure_dose(beam[X_index], 1E-3, "water")
        DOSE[1, DZ_index, DY_index], DOSE[2, DZ_index, DY_index] = np.max(rad_max), np.max(gray_max)


    print("------------------------------------------------------------------------\n")
    print("     Measurement taken at %.1f cm from sources"%X_dist)
    print("     MAX EXPOSURE is: %.3f R/hr"%np.nanmax(DOSE[0]*1E-3))
    print("     MAX DOSE RATE is: %.3f rad(Si)/hr"%np.nanmax(DOSE[1]))
    print("     MAX DOSE RATE is: %.3f Gy(Si)/hr"%np.nanmax(DOSE[2]))
    print("     MIN DOSE RATE is: %.3f Gy(Si)/hr"%np.nanmin(DOSE[2]))
    print("\n------------------------------------------------------------------------")


    plt.figure()
    im = plt.imshow(DOSE[0], origin='lower')
    plt.colorbar(im, label="Exposure Rate (R/hr)")
    plt.xlabel("Distance between Center of Sources (cm)")
    plt.ylabel("Distance between Top and Bottom of Sources (cm)")
    plt.title("Maximum Exposure Rate at %.1f cm"%X_dist)
    plt.savefig(Processed_Data+"MaxExposure_%.1fcm.jpg"%X_dist)

    plt.figure()
    im = plt.imshow(DOSE[1], origin='lower')
    plt.colorbar(im, label="Absorbed Dose Rate (rads(Si)/hr)")
    plt.xlabel("Horizontal Distance between Center of Sources (cm)")
    plt.ylabel("Vertical Distance between Top and Bottom of Sources (cm)")
    plt.title("Maximum Absorbed Dose Rate at %.1f cm"%X_dist)
    plt.savefig(Processed_Data+"MaxRADS_%.1fcm.jpg"%X_dist)
    plt.show()

    plt.figure()
    im = plt.imshow(DOSE[2], origin='lower')
    plt.colorbar(im, label="Absorbed Dose Rate (Gy(Si)/hr)")
    plt.xlabel("Distance between Center of Sources (cm)")
    plt.ylabel("Distance between Top and Bottom of Sources (cm)")
    plt.title("Maximum Absorbed Dose Rate at %.1f cm"%X_dist)
    plt.savefig(Processed_Data+"MaxGREYS_%.1fcm.jpg"%X_dist)


def radiochromic_film(path_10, path_175, save):
    data_10 = np.loadtxt(path_10,dtype="float", skiprows=1)    
    data_175 = np.loadtxt(path_175,dtype="float", skiprows=1)  

    ## define method of returning the DUR
    def DUR(data):
        args_max, args_min = np.unravel_index(np.argmax(data), data.shape), np.unravel_index(np.argmin(data), data.shape)
        max, min = np.max(data), np.min(data)
        DUR = max/data
        max_DUR = np.max(DUR)
        return DUR, max_DUR
    
    DUR_10, max_DUR_10 = DUR(data_10)
    DUR_175, max_DUR_175 = DUR(data_175)

    def PRINT(dist, max_DUR):
        print("------------------------------------------------------------------------\n")
        print("     Measurement taken at %.1f cm from sources"%dist)
        print("     MIM DUR: %.3f"%max_DUR)
        print("\n------------------------------------------------------------------------")
    PRINT(10, max_DUR_10)
    PRINT(17.5, max_DUR_175)

    def plotter(dist, data):
        shape = data.shape
        center_x, center_y = shape[1]/2, shape[0]/2

        fig, ax = plt.subplots()
        im = ax.imshow(data, origin="lower")
        plt.colorbar(im, label="Dose Uniformity Ratio")
        ax.vlines(center_x, -0.5, shape[0]-0.5, color="r", linestyles="dashed", linewidths=1.0)
        ax.hlines(center_y-0.5, -0.5, shape[1]-0.5, color="r", linestyles="dashed", linewidths=1.0)
        ax.set_xticks(np.arange(0, 19, 2), np.arange(-10, 10, 2))
        ax.set_yticks(np.linspace(0,14, 8), np.linspace(-7, 7, 8))
        ax.set_xlabel("Distance from Center (cm)")
        ax.set_ylabel("Distance from Center (cm)")
        ax.set_title("Measured DUR at %.1f cm"%dist)
        plt.savefig(save+"Radiochromic_Film_2UP_%.1fcm.jpg"%dist)
        plt.show()

    plotter(10, DUR_10)
    plotter(17.5, DUR_175)
        

# read_data_to_table(directory)
    
# beam_uniformity_multiFiles(Save_Tallies, 17, Processed_Data)    

for i in [10.5, 17.5, 30.5,50.5,  70.5, 90.5]:
    print("Starting %.1f"%i)
    plot_beam_YZ_ofX(Save_Tallies+"4.50-4.50_TallyValues.npy", i, Processed_Data)

# max_dose_configs(Save_Tallies, 10.5, Processed_Data)

# film_10 = "X:\\Operations\\ProjectsEng\\HDI\\SO 3887 JLS 484 Replacement\\Radiation Physics\\FWT - Radiochromic Film Readouts\\HDR Cal Data_100mm.txt"
# film_175 = "X:\\Operations\\ProjectsEng\\HDI\\SO 3887 JLS 484 Replacement\\Radiation Physics\\FWT - Radiochromic Film Readouts\\HDR Cal Data_175mm.txt"
# save_film = "X:\\Operations\\ProjectsEng\\HDI\\SO 3887 JLS 484 Replacement\\Radiation Physics\\FWT - Radiochromic Film Readouts\\"
# radiochromic_film(film_10, film_175, save_film )

# print(mass_atten_water/mass_atten_air)    