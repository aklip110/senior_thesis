import numpy as np
import matplotlib.pyplot as plt
from get_data import *

#define functions----------------------------------------------

def subtract(ONdata, OFFdata):
    """
    INPUT: this function takes two same-shape arrays (Ondata and OFFdata) as input.
    FUNCTION: it subtracts Bx, By, Bz (on - off) and computes the new errors by adding in quadrature.
    RETURNS: newDat array that's the same shape as both initial arrays. output is non-avgd, subtracted array that has correctly propagated errors.
    """
    numRows = ONdata.shape[0]
    numCols = ONdata.shape[1]
    newDat = ONdata.copy()
    for i in range(numRows):
        #Bx, By, Bz
        newDat[i, 3] = ONdata[i, 3] - OFFdata[i, 3]
        newDat[i, 5] = ONdata[i, 5] - OFFdata[i, 5]
        newDat[i, 7] = ONdata[i, 7] - OFFdata[i, 7]
        #errors
        newDat[i, 4] = np.sqrt(ONdata[i, 4]**2 + OFFdata[i, 4]**2)
        newDat[i, 6] = np.sqrt(ONdata[i, 6]**2 + OFFdata[i, 6]**2)
        newDat[i, 8] = np.sqrt(ONdata[i, 8]**2 + OFFdata[i, 8]**2)
        #SA probe
        newDat[i, 10] = ONdata[i, 10] - OFFdata[i, 10]
        newDat[i, 11] = np.sqrt(ONdata[i, 11]**2 + OFFdata[i, 11]**2)
    return newDat
    
def WAV(data, errors):
    """
    INPUT: data (1 x n vector of data), errors (1 x n vector of corresponding uncertainties)
    FUNCTION: computes a weighted average of a vector of values (data) with corresponding uncertainties (errors) via two methods. 1st method is weighted average second is regular average.
    RETURNS: x_wav: a number, RMS: a number, sd_wav: a number, x_av: number
    I am curious to compare the sd_wav and the RMS, and x_wav and x_av
    """
    weights = np.reciprocal(np.multiply(errors, errors))
    x_wav = (1 / np.sum(weights)) * np.sum(np.multiply(data, weights))
    x_av = np.mean(data)
    sd_wav = 1 / np.sqrt(np.sum(weights))
    RMS = np.std(data, ddof=1) / np.sqrt(len(data))
    return np.array([x_wav, sd_wav, x_av, RMS])
    
def allAvgs(array):
    """
    INPUT: array (nx15 element array)
    FUNCTION: applies the WAV function to each of Bx, By, Bz, SA.
    OUTPUT: a 1x19 element array (droppping the on/off row and the time row bc meaningless) that will get stacked onto the avgd_rot_dat array.
    output array = [T, R, V, Bx_wav, Bx_wav_sd, Bx_av, Bx_sd,...]
    """
    outputArr = np.zeros(19)
    outputArr[0:3] = array[0, 0:3]
    outputArr[3:7] = WAV(array[:, 3], array[:, 4])
    outputArr[7:11] = WAV(array[:, 5], array[:, 6])
    outputArr[11:15] = WAV(array[:, 7], array[:, 8])
    outputArr[15:19] = WAV(array[:, 10], array[:, 11])
    return outputArr
    
def getMags(data):
    """
    INPUT: data is an n x 13 array of subtracted data--from the subtract function.
    FUNCTION: this function computes |B| from Bx, By, Bz. also computes the new errors via the formula for errors of a multivariate function.
    OUTPUT: n x 5 array: [T, R, V, |B|, |B|error]
    """
    numPts = data.shape[0]
    outDat = np.zeros(5)
    for i in range(numPts):
        # compute |B|
        Bx = data[i, 3]
        By = data[i, 5]
        Bz = data[i, 7]
        dBx = data[i, 4]
        dBy = data[i, 6]
        dBz = data[i, 8]
        Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)
        Berr = (1 / Bmag) * np.sqrt((Bx * dBx)**2 + (By * dBy)**2 + (Bz * dBz)**2)
        newRow = np.array([data[i, 0], data[i, 1], data[i, 2], Bmag, Berr])
        outDat = np.vstack([outDat, newRow])
    outDat = outDat[1:, :]
    return outDat

def finalize_data(filePath, dataName, plusX, minX, plusY, minY):
    """
    INPUT: filepath (string) specifies up to the senior thesis directory. dataName (string) is the date/time signifier for a particular dataset.
    FUNCTION: gets on and off data, rotates XY and Z data, subtracts out background and computes errors, computes magnitudes and errors
    OUTPUT: xyDat (nx15 array) and zDat (nx15 array) are ready to be averaged and that have two new columns: |B| and Berrs
    """
    # get on and off data
    XYon, Zon = grab_data(filePath, dataName, 1)
    XYoff, Zoff = grab_data(filePath, dataName, 0)
    
    # rotate all four files
    XYonRot = rotate_XY(XYon, dataName, "on", plusX, minX, plusY, minY, filePath)
    XYoffRot = rotate_XY(XYoff, dataName, "off", plusX, minX, plusY, minY, filePath)
    ZonRot = rotate_Z(Zon, dataName, "on", plusX, minX, plusY, minY, filePath)
    ZoffRot = rotate_Z(Zoff, dataName, "off", plusX, minX, plusY, minY, filePath)
    print(" ")
    print("Subtracting and computing uncertainties for "+ dataName + " data...")
    #get subtracted "delta" data------------------------------------
    xyDat = subtract(XYonRot, XYoffRot)
    zDat = subtract(ZonRot, ZoffRot)
    #length = len(xyDat)
    #Zlength = len(zDat)
    print("  xyDat shape: ", xyDat.shape)
    print("  zDat shape: ", zDat.shape)
    
    #compute magnitudes |B|
    xyMags = getMags(xyDat)
    zMags = getMags(zDat)
    print("  xyMags shape: ", xyMags.shape)
    print("  zMags shape: ", zMags.shape)
    
    #stack the |B| and |B|error columns onto the xyDat and zDat arrays
    newcol1 = np.array([xyMags[:, 3]])
    newcol1 = newcol1.T
    newcol2 = np.array([xyMags[:, 4]])
    newcol2 = newcol2.T
    xyDat = np.hstack((xyDat, newcol1, newcol2))
    print("  xyDat shape after: ", xyDat.shape)
    
    newcol3 = np.array([zMags[:, 3]])
    newcol3 = newcol3.T
    newcol4 = np.array([zMags[:, 4]])
    newcol4 = newcol4.T
    zDat = np.hstack((zDat, newcol3, newcol4))
    print("  zDat shape after: ", zDat.shape)
    #now the xyDat and zDat arrasy have 15 columns
    print(" ")
    
    return xyDat, zDat
    
def XY_averaging(xyDat, filePath, dataName):
    """
    INPUT: xyDat (nx15 array) of rotated, subtracted xy data;
    FUNCTION: averages the XY data to get <Bx>, <By>, <Bz> vs. x and y.
    OUTPUT: avgd_rot_dat (mx23 array) of averaged data. also gets saved to a file
    """
    print("    ")
    print("Averaging XY " + dataName + " data...")
    length = xyDat.shape[0]
    T = xyDat[:, 0] #1st column
    R = xyDat[:, 1] #2nd column
    V = xyDat[:, 2] #3rd column

    print("  T values: ", np.unique(T))
    print("  R values: ",np.unique(R))
    print("  V values: ",np.unique(V))
    numT = len(np.unique(T))
    numR = len(np.unique(R))
    numV = len(np.unique(V))

    numPoints = numT * numR * numV
    print("  unique points in xy sweep: ", numPoints)
    #data arrays
    avgd_rot_dat = np.zeros(13 + 6 + 4)
    rot_dat = np.zeros(13 + 2)

    for t in np.unique(T):
        for r in np.unique(R):
            for v in np.unique(V):
                vals_to_avg = np.zeros(13 + 2)
                for i in range(length):
                    if (xyDat[i, 0] == t) and (xyDat[i, 1] == r) and (xyDat[i, 2] == v):
                        newRow = xyDat[i, :]
                        vals_to_avg = np.vstack([vals_to_avg, newRow])
                        rot_dat = np.vstack([rot_dat, newRow])
                vals_to_avg = vals_to_avg[1:, :]
                magAvgs = WAV(vals_to_avg[:, 13], vals_to_avg[:, 14])
                avgd_rot_dat = np.vstack([avgd_rot_dat, np.hstack((allAvgs(vals_to_avg), magAvgs))])
    
    avgd_rot_dat = avgd_rot_dat[1:, :]
    rot_dat = rot_dat[1:, :]
    
    #double check the shapes of the arrays
    print("  avgd_rot_dat array shape: ", avgd_rot_dat.shape)
    print("  rot_dat array shape: ", rot_dat.shape)

    savedName = "averaged_ep_" + dataName +".txt"
    np.savetxt(filePath + "general_analysis/txt_files/" + savedName, avgd_rot_dat, delimiter=" ")
    np.savetxt(filePath + "general_analysis/txt_files/" + "ep_" + dataName +".txt", rot_dat, delimiter=" ")

    print("  Saved to: " + savedName)
    print("   ")

    return avgd_rot_dat

def Z_averaging(zDat, filePath, dataName):
    """
    INPUT: zDat (nx15 array) of rotated, subtracted z data;
    FUNCTION: averages the Z data to get <Bx>, <By>, <Bz> vs. z.
    OUTPUT: avgd_rot_Zdat (mx23 array) of averaged data. also gets saved to a file
    """
    print("    ")
    print("Averaging Z " + dataName + " data...")
    Zlength = zDat.shape[0]
    
    Z = zDat[:, 2] #3rd column
    numZ = np.unique(Z)
    print("  unique points in z sweep: ", numZ)

    avgd_rot_Zdat = np.zeros(13 + 6 + 4)
    rot_Zdat = np.zeros(13 + 2)
    
    for zval in numZ:
        zvals_to_avg = np.zeros(13 + 2)
        for i in range(Zlength):
            if zDat[i, 2] == zval:
                newZRow = zDat[i, :]
                zvals_to_avg = np.vstack([zvals_to_avg, newZRow])
                rot_Zdat = np.vstack([rot_Zdat, newZRow])
        zvals_to_avg = zvals_to_avg[1:, :]
        zmagAvgs = WAV(zvals_to_avg[:, 13], zvals_to_avg[:, 14])
        avgd_rot_Zdat = np.vstack([avgd_rot_Zdat, np.hstack((allAvgs(zvals_to_avg), zmagAvgs))])
        
    avgd_rot_Zdat = avgd_rot_Zdat[1:, :]
    rot_Zdat = rot_Zdat[1:, :]
    
    #double check the shapes of the arrays
    print("  avgd_rot_Zdat array shape: ", avgd_rot_Zdat.shape)
    print("  rot_Zdat array shape: ", rot_Zdat.shape)

    savedName = "averaged_ep-Z_" + dataName +".txt"
    np.savetxt(filePath + "general_analysis/txt_files/" + savedName, avgd_rot_Zdat, delimiter=" ")
    np.savetxt(filePath + "general_analysis/txt_files/" + "ep-Z_" + dataName + ".txt", rot_Zdat, delimiter=" ")

    print("  Saved to: " + savedName)
    print("   ")

    return avgd_rot_Zdat
    
def get_LeftRight(avgd_rot_dat, Vval, PLUSval, MINval, T0):
    """
    INPUT: avgd_rot_dat (nx23 array) is output from the XY_averaging function, Vval (int) is the vertical value where the xy slice is taken, PLUSval/MINval (ints) are the plus and minus either x or y rotation "R" values. T) is trolley centering value.
    FUNCTION: this function grabs the non-weighted average and errors for either x or y at a certain height. the output is perfect to immediately plot.
    OUTPUT: Right and Left (nx9 arrays). [x, <Bx>, err, ..., <|B|>, err]. to plot, just grab desired columns.
    """
    
    print("    ")
    print("Getting left (" + str(MINval) +") and right (" + str(PLUSval) +") data...")
    Right = np.zeros(9)
    Left = np.zeros(9)
    
    for i in range(len(avgd_rot_dat)):
        if (avgd_rot_dat[i, 2] == Vval) and (avgd_rot_dat[i, 1] == PLUSval):
            dist = (22.7 / 30000) * (avgd_rot_dat[i,0] - T0)
            addRow = [dist, avgd_rot_dat[i, 5], avgd_rot_dat[i, 6],avgd_rot_dat[i, 9],avgd_rot_dat[i, 10],avgd_rot_dat[i, 13],avgd_rot_dat[i, 14],avgd_rot_dat[i, -2],avgd_rot_dat[i, -1]]
            #print(addRow)
            Right = np.vstack([Right, addRow])

        if (avgd_rot_dat[i, 2] == Vval) and (avgd_rot_dat[i, 1] == MINval):
            dist = (22.7 / 30000) * (-(avgd_rot_dat[i,0] - T0))
            addRow = [dist, avgd_rot_dat[i, 5], avgd_rot_dat[i, 6],avgd_rot_dat[i, 9],avgd_rot_dat[i, 10],avgd_rot_dat[i, 13],avgd_rot_dat[i, 14],avgd_rot_dat[i, -2],avgd_rot_dat[i, -1]]
            #print(addRow)
            Left = np.vstack([Left, addRow])
        
    Right = Right[1:, :]
    Left = Left[1:, :]
    print("    ")
    return Right, Left

def get_AllZ(avgd_rot_dat, Vval):
    """
    INPUT: avgd_rot_dat (nx23 array) is output from the Z_averaging functions.  Vval is the vertical value where the z dat is centered.
    FUNCTION: this function doesnt discriminate between left or right like the above, can be used for z data.
    OUTPUT: allVals (nx9 array)
    """
    print("    ")
    print("Getting all z data...")
    allVals = np.zeros(9)
    
    for i in range(len(avgd_rot_dat)):
        dist = (15.2 / 50000) * (avgd_rot_dat[i,2] - Vval)
        addRow = [dist, avgd_rot_dat[i, 5], avgd_rot_dat[i, 6],avgd_rot_dat[i, 9],avgd_rot_dat[i, 10],avgd_rot_dat[i, 13],avgd_rot_dat[i, 14],avgd_rot_dat[i, -2],avgd_rot_dat[i, -1]]
        #print(addRow)
        allVals = np.vstack([allVals, addRow])
        
    allVals = allVals[1:, :]
    print("    ")
    return allVals
    
def complete_data(filePath, dataName, pX, mX, pY, mY, V1, T0):
    """
    INPUT: standard from previous functions.
    FUNCTION: this is the most general/broad function that calls all subsequent functions in this script and the get_data.py script. it generates the Bx, By, Bz, |B| data and their errors necessary for plotting given a single dataset name.
    OUTPUT: +/- x data, +/- y data, z data. (all nx9 arrays) [dist, Bx, err, By, err, Bz, err, |B|, err]
    """
    xyDat, zDat = finalize_data(filePath, dataName, pX, mX, pY, mY)
    
    xyAvgd = XY_averaging(xyDat, filePath, dataName)
    zAvgd = Z_averaging(zDat, filePath, dataName)
    
    xRight, xLeft = get_LeftRight(xyAvgd, V1, pX, mX, T0)
    yRight, yLeft = get_LeftRight(xyAvgd, V1, pY, mY, T0)
    zAll = get_AllZ(zAvgd, V1)
    print(" ")
    print("Assembling complete plottable datasets for " + dataName + "...")
    print("  xLeft shape: ", xLeft.shape)
    print("  xRight shape: ", xRight.shape)
    print("  yLeft shape: ", yLeft.shape)
    print("  yRight shape: ", yRight.shape)
    print("  zAll shape: ", zAll.shape)
    print(" ")
    
    return xRight, xLeft, yRight, yLeft, zAll


#checked all functions


