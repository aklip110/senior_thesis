#z is vertical
#for use with any dataset
#Goal: grab the desired data. rotate each row (both the standard rotations and then the corrections (future)). output rotated data
#no averaging.
# trolley centering value not known exactly--will have to estimate

import numpy as np
import matplotlib.pyplot as plt

#given a data file name from umit (XX-XX_XX-XX) date and time
#we import the corresponding Z and XY files. I will stick to manually separating the data because it's simpler

def grab_data(filePath, dataName, OnOff, fixed=True):
    """
    INPUT: filePath (string, / at end) is the path to the directory right above the folder where both Z and XY data are stored. dataName (string) is the date/time signature that specifies the specific directory and data. OnOff (int 0 or 1) specifies if we want to grab on or off data.
    FUNCTION: grabs and outputs the untransformed XY and Z datasets for either OFF or ON data in a specific data/time folder.
    OUTPUT: XYdat: XY data, Zdat: Z data.
    """
    print("   ")
    print("Importing " + str(OnOff) + " data...")
    print("  File: " + str(dataName))
    #import XY data ------------------------------
    if fixed==True:
        XYfile = filePath + "data_" + dataName + "/" + "data_" + dataName + "-3D_scan-FIXED.txt"
    elif fixed==False:
                XYfile = filePath + "data_" + dataName + "/" + "data_" + dataName + "-3D_scan.txt"
    XYdata = np.loadtxt(XYfile)
    print("  XY Data sweep Shape: ", XYdata.shape)
    length = XYdata.shape[0]
    
    #import z data
    Zfile = filePath + "data_" + dataName + "/" + "data_" + dataName + "-vertical_sweep-FIXED.txt"
    Zdata = np.loadtxt(Zfile)
    print("  Z Data sweep Shape: ", Zdata.shape)
    zlength = Zdata.shape[0]

    #grab desired data --------------------
    dat = np.zeros(13)
    zdat = np.zeros(13)

    for i in range(length):
        if (XYdata[i, 9] == OnOff):
            dat = np.vstack([dat, XYdata[i,0:13]])
    for i in range(zlength):
        if (Zdata[i, 9] == OnOff):
            zdat = np.vstack([zdat, Zdata[i,0:13]])

    #drop zero row
    XYdat = dat[1:, :]
    Zdat = zdat[1:, :]
    print("  XY " +str(OnOff) + " data shape: ", XYdat.shape)
    print("  Z " +str(OnOff) + " data shape: ", Zdat.shape)
    print("   ")
    return XYdat, Zdat

def trans_row(row, pX, mX, pY, mY):
    """
    INPUT: row (1 x 13 dimensional vector). p/m X/Y (integers) are plus and minus X and Y rotate "R" values.
    FUNCTION: given a row, this function will transform the Bx, By, Bz probe frame values into the lab frame UNDER THE ASSUMPTION that the given plusX==pX etc rotation values (in steps not radians) are exactly the true axes. currently not implementing rotations to fix the misalignments.
    OUTPUT: newRow (1 x 13 dimensional vector) that's been correctly rotated.
    """
    newRow = np.copy(row)
    R = row[1]
    if R == pX:
        newRow[3] = row[7]
        newRow[5] = -row[3]
    if R == mX:
        newRow[3] = -row[7]
        newRow[5] = row[3]
    if R == pY:
        newRow[3] = row[3]
        newRow[5] = row[7]
    if R == mY:
        newRow[3] = -row[3]
        newRow[5] = -row[7]
    newRow[7] = -row[5]
    return newRow

def rotate_XY(dat, dataName, onORoff, plusX, minX, plusY, minY, filePath):
    """
    INPUT: dat (n x 13 array) is XY data (either on or off), dataName (string) is the date/time signature for the file. onORoff (string) either "on" or "off".
    FUNCTION: rotates each row in the XY data set, sorts data by T, R, V values. also saves the rotated data to a file that signifies on/off status and data name.
    OUTPUT: rot_dat (n x 13 array) of rotated data. saves a file "on/off_rot_XX-XX_XX-XX.txt"
    """
    print("   ")
    print("Rotating "+ dataName + " " + onORoff + " XY data...")
    
    length = dat.shape[0]
    T = dat[:, 0] #1st column trolley vals
    R = dat[:, 1] #2nd column rotate vals
    V = dat[:, 2] #3rd column vertical vals

    print("  T values: ", np.unique(T))
    print("  R values: ",np.unique(R))
    print("  V values: ",np.unique(V))
    numT = len(np.unique(T))
    numR = len(np.unique(R))
    numV = len(np.unique(V))

    numPoints = numT * numR * numV
    print("  unique pts in xy sweep: ", numPoints)
    
    rot_dat = np.zeros(13)
    
    for t in np.unique(T):
        for r in np.unique(R):
            for v in np.unique(V):
                for i in range(length):
                    if (dat[i, 0] == t) and (dat[i, 1] == r) and (dat[i, 2] == v):
                        #rotate/transform the row
                        newRow = trans_row(dat[i, :], plusX, minX, plusY, minY)
                        rot_dat = np.vstack([rot_dat, newRow])

    rot_dat = rot_dat[1:, :]
    print("  length of rot_dat: ", rot_dat.shape[0])
    savedName = onORoff + "_rot_" + dataName + ".txt"
    np.savetxt(filePath + "general_analysis/txt_files/" + savedName, rot_dat, delimiter=" ")
    print("  number of points to average over: ", int( rot_dat.shape[0] / numPoints))
    print("  Saved to: " + savedName)
    print("   ")

    return rot_dat

def rotate_Z(Zdat, dataName, onORoff, plusX, minX, plusY, minY, filePath):
    """
    INPUT: dat (n x 13 array) is Z data (either on or off), dataName (string) is the date/time signature for the file. onORoff (string) either "on" or "off".
    FUNCTION: rotates each row in the Z data set, sorts data by V values. also saves the rotated data to a file that signifies on/off status and data name.
    OUTPUT: rot_Zdat (n x 13 array) of rotated data. saves a file "on/off_rot_Z_XX-XX_XX-XX.txt"
    """
    print("   ")
    print("Rotating "+ dataName + " " + onORoff + " Z data...")
    
    Zlength = Zdat.shape[0]
    Z = Zdat[:, 2] #3rd column
    numZ = np.unique(Z)
    print("  Z values: ", numZ)
    
    rot_Zdat = np.zeros(13)
    
    for zval in numZ:
        for i in range(Zlength):
            if Zdat[i, 2] == zval:
                newZRow = trans_row(Zdat[i, :], plusX, minX, plusY, minY)
                rot_Zdat = np.vstack([rot_Zdat, newZRow])

    rot_Zdat = rot_Zdat[1:, :]
    print("  length of rot_Zdat: ", rot_Zdat.shape[0])
    savedName = onORoff + "_rot_Z_" + dataName + ".txt"
    np.savetxt(filePath + "general_analysis/txt_files/" + savedName, rot_Zdat, delimiter=" ")
    print("  number of points to average over: ", int( rot_Zdat.shape[0] / len(numZ)))
    print("  Saved to: " + savedName)
    print("   ")

    return rot_Zdat
