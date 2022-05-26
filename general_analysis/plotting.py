import numpy as np
import matplotlib.pyplot as plt
from get_data import *
from deltaB_general import *
from numpy.polynomial import polynomial as P

#plotting functions
def quadratic(x, params):
    """
    for use with fits and plots. params is a vector of three parameters ax^2 + bx + c
    """
    y = params[2] + params[1] * x + params[0] * x**2
    return y
    
def vertex(params):
    """
    computes vertex of parabola given three parameters
    """
    vert = -params[1] / (2 * params[0])
    return vert
    
def T0Err(params, cov):
    a = params[0]
    aErr = np.sqrt(cov[0,0])
    b = params[1]
    bErr = np.sqrt(cov[1,1])
    tErr = (1/2) * np.sqrt((bErr / b)**2 + (aErr / a)**2)
    return tErr
    
def ComparisonPlots(Right, Left, Right2, Left2, rColor, lColor, dataName, filePath,  xyz, T0, plus=0, minus=0, legend=True, fixed=True):
    """
    INPUT: Right/Right2 (nx9 array) of data for RHS, Left/Left2 (nx9 array) of data for LHS--set both same if dont want to distinguish; r/lCOlor is list of two right/left colors e.g. ["red", "blue"]; dataName (string) is data/time stamp; filePath (string) is path to senior_thesis directory; plus/minus are the plus and minus R values (only used with x or y data); xyz (string: "x", "y", or "z") specifies which axis we're sweeping; legend (Bool) is False for z.
    FUNCTION: makes 4 plots given the input data: <Bx>, <By>, <Bz>, <|B|>.
    OUTPUT: nothing, just saves plots to the dataName directory.
    """
    if xyz == "z":
        T0 = 0
        
    #if fixed == True:
        #saveLoc = "/FIXEDplots/"
        #titleAdd = "FIXED"
        #fileDelim = "-FIXED.png"
    #elif fixed==False:
    saveLoc = "/compareplots/"
    titleAdd = " comparison"
    fileDelim = "-comparison.png"
    # <Bx> vs. x
    print("-------")
    #print("T): ", T0)
    #print(Right[:, 0])
    #print(Right[:, 0] - T0)
    fig1 = plt.figure()
    plt.ion()
    plt.scatter(Right2[:, 0] - T0, Right2[:, 1],  color=rColor[1], label="R = "+str(plus))
    plt.scatter(Left2[:, 0] + T0, Left2[:, 1], color=lColor[1], label="R = "+str(minus))
    plt.errorbar(Right2[:, 0] - T0, Right2[:, 1],  yerr=Right[:, 2],fmt="o", color=rColor[1])
    plt.errorbar(Left2[:, 0] + T0, Left2[:, 1],  yerr=Left[:, 2], fmt="o",color=lColor[1])
    
    plt.scatter(Right[:, 0] - T0, Right[:, 1],  color=rColor[0], label="R = "+str(plus))
    plt.scatter(Left[:, 0] + T0, Left[:, 1], color=lColor[0], label="R = "+str(minus))
    plt.errorbar(Right[:, 0] - T0, Right[:, 1],  yerr=Right[:, 2],fmt="o", color=rColor[0])
    plt.errorbar(Left[:, 0] + T0, Left[:, 1],  yerr=Left[:, 2], fmt="o",color=lColor[0])


    plt.xlabel(xyz + " (cm)")
    plt.ylabel("<Bx> (mG)")
    #plt.xlim((0, 5))
    plt.title("<Bx> (along B0) vs "+ xyz +": " + dataName + titleAdd)
    if legend == True:
        plt.legend(loc="upper center")
    plt.ioff()
    plt.savefig(filePath + "data_" + dataName + saveLoc + "Bx_" + xyz + "-deltaB_" + dataName + fileDelim, dpi=500)
    #plt.show()
    
    # <By> vs. x
    fig2 = plt.figure()
    plt.ion()
    
    plt.scatter(Right2[:, 0] - T0, Right2[:, 3],  color=rColor[1], label="R = "+str(plus))
    plt.scatter(Left2[:, 0] + T0, Left2[:, 3], color=lColor[1], label="R = "+str(minus))
    plt.errorbar(Right2[:, 0] - T0, Right2[:, 3], yerr=Right[:, 4],fmt="o", color=rColor[1])
    plt.errorbar(Left2[:, 0] + T0, Left2[:, 3], yerr=Left[:, 4], fmt="o",color=lColor[1])
    
    plt.scatter(Right[:, 0] - T0, Right[:, 3],  color=rColor[0], label="R = "+str(plus))
    plt.scatter(Left[:, 0] + T0, Left[:, 3], color=lColor[0], label="R = "+str(minus))
    plt.errorbar(Right[:, 0] - T0, Right[:, 3], yerr=Right[:, 4],fmt="o", color=rColor[0])
    plt.errorbar(Left[:, 0] + T0, Left[:, 3], yerr=Left[:, 4], fmt="o",color=lColor[0])

    
    #limy = np.max(np.hstack((Right[:, 4], Left[:, 4]))) + np.max(np.hstack((np.abs(Right[:, 3]), np.abs(Left[:, 3]))))
    
    plt.xlabel(xyz + " (cm)")
    plt.ylabel("<By> (mG)")
    #if xyz != "z":
        #plt.ylim((-2 * limy, 2 * limy))
    plt.title("<By> vs "+ xyz +": " + dataName + titleAdd)
    if legend == True:
        plt.legend(loc="lower right")
    plt.ioff()
    plt.savefig(filePath + "data_" + dataName + saveLoc + "By_" + xyz + "-deltaB_" + dataName + fileDelim, dpi=500)
    #plt.show()
    
    # <Bz> vs. x
    fig3 = plt.figure()
    plt.ion()
    plt.scatter(Right2[:, 0] - T0, Right2[:, 5],  color=rColor[1], label="R = "+str(plus))
    plt.scatter(Left2[:, 0] + T0, Left2[:, 5], color=lColor[1], label="R = "+str(minus))
    plt.errorbar(Right2[:, 0] - T0, Right2[:, 5], yerr=Right[:, 6],fmt="o", color=rColor[1])
    plt.errorbar(Left2[:, 0] + T0, Left2[:, 5], yerr=Left[:, 6], fmt="o", color=lColor[1])
    
    plt.scatter(Right[:, 0] - T0, Right[:, 5],  color=rColor[0], label="R = "+str(plus))
    plt.scatter(Left[:, 0] + T0, Left[:, 5], color=lColor[0], label="R = "+str(minus))
    plt.errorbar(Right[:, 0] - T0, Right[:, 5], yerr=Right[:, 6],fmt="o", color=rColor[0])
    plt.errorbar(Left[:, 0] + T0, Left[:, 5], yerr=Left[:, 6], fmt="o", color=lColor[0])


    #limz = np.max(np.hstack((Right[:, 6], Left[:, 6]))) + np.max(np.hstack((np.abs(Right[:, 5]), np.abs(Left[:, 5]))))

    plt.xlabel(xyz + " (cm)")
    plt.ylabel("<Bz> (mG)")
    #if xyz != "z":
        #plt.ylim((-2 * limz, 2 * limz))
    plt.title("<Bz> (vertical) vs "+ xyz +": " + dataName + titleAdd)
    if legend == True:
        plt.legend(loc="upper center")
    plt.ioff()
    plt.savefig(filePath + "data_" + dataName + saveLoc + "Bz_" + xyz + "-deltaB_" + dataName + fileDelim, dpi=500)
    #plt.show()
    
    #<|B|> vs. x
    fig4 = plt.figure()
    plt.ion()
    plt.scatter(Right[:, 0] - T0, Right[:, 7],  color=rColor[0], label="R = "+str(plus))
    plt.scatter(Left[:, 0] + T0, Left[:, 7], color=lColor[0], label="R = "+str(minus))
    plt.errorbar(Right[:, 0] - T0, Right[:, 7], yerr=Right[:, 8],fmt="o", color=rColor[0])
    plt.errorbar(Left[:, 0] + T0, Left[:, 7], yerr=Left[:, 8], fmt="o",color=lColor[0])
    
    plt.scatter(Right2[:, 0] - T0, Right2[:, 7],  color=rColor[1], label="R = "+str(plus))
    plt.scatter(Left2[:, 0] + T0, Left2[:, 7], color=lColor[1], label="R = "+str(minus))
    plt.errorbar(Right2[:, 0] - T0, Right2[:, 7], yerr=Right[:, 8],fmt="o", color=rColor[1])
    plt.errorbar(Left2[:, 0] + T0, Left2[:, 7], yerr=Left[:, 8], fmt="o",color=lColor[1])

    plt.xlabel(xyz + " (cm)")
    plt.ylabel("<|B|> (mG)")
    plt.title("<|B|> vs "+ xyz +": " + dataName + titleAdd)
    if legend == True:
        plt.legend(loc="upper center")
    plt.ioff()
    plt.savefig(filePath + "data_" + dataName + saveLoc + "B_" + xyz + "-deltaB_" + dataName + fileDelim, dpi=500)
    #plt.show()
    
    return
    
##################################################

def Plots(Right, Left, rColor, lColor, dataName, filePath,  xyz, T0, plus=0, minus=0, legend=True, fixed=True):
    """
    INPUT: Right (nx9 array) of data for RHS, Left (nx9 array) of data for LHS--set both same if dont want to distinguish; r/lCOlor is right/left color; dataName (string) is data/time stamp; filePath (string) is path to senior_thesis directory; plus/minus are the plus and minus R values (only used with x or y data); xyz (string: "x", "y", or "z") specifies which axis we're sweeping; legend (Bool) is False for z.
    FUNCTION: makes 4 plots given the input data: <Bx>, <By>, <Bz>, <|B|>.
    OUTPUT: nothing, just saves plots to the dataName directory.
    """
    if xyz == "z":
        T0 = 0
        
    if fixed == True:
        saveLoc = "/FIXEDplots/"
        titleAdd = "FIXED"
        fileDelim = "-FIXED.png"
    elif fixed==False:
        saveLoc = "/plots/"
        titleAdd = " "
        fileDelim = ".png"
    # <Bx> vs. x
    print("-------")
    print("T): ", T0)
    print(Right[:, 0])
    print(Right[:, 0] - T0)
    fig1 = plt.figure()
    plt.ion()
    plt.scatter(Right[:, 0] - T0, Right[:, 1],  color=rColor, label="R = "+str(plus))
    plt.scatter(Left[:, 0] + T0, Left[:, 1], color=lColor, label="R = "+str(minus))

    plt.errorbar(Right[:, 0] - T0, Right[:, 1],  yerr=Right[:, 2],fmt="o", color=rColor)
    plt.errorbar(Left[:, 0] + T0, Left[:, 1],  yerr=Left[:, 2], fmt="o",color=lColor)

    plt.xlabel(xyz + " (cm)")
    plt.ylabel("<Bx> (mG)")
    #plt.xlim((0, 5))
    plt.title("<Bx> (along B0) vs "+ xyz +": " + dataName + titleAdd)
    if legend == True:
        plt.legend(loc="upper center")
    plt.ioff()
    plt.savefig(filePath + "data_" + dataName + saveLoc + "Bx_" + xyz + "-deltaB_" + dataName + fileDelim, dpi=500)
    #plt.show()
    
    # <By> vs. x
    fig2 = plt.figure()
    plt.ion()
    plt.scatter(Right[:, 0] - T0, Right[:, 3],  color=rColor, label="R = "+str(plus))
    plt.scatter(Left[:, 0] + T0, Left[:, 3], color=lColor, label="R = "+str(minus))

    plt.errorbar(Right[:, 0] - T0, Right[:, 3], yerr=Right[:, 4],fmt="o", color=rColor)
    plt.errorbar(Left[:, 0] + T0, Left[:, 3], yerr=Left[:, 4], fmt="o",color=lColor)
    
    limy = np.max(np.hstack((Right[:, 4], Left[:, 4]))) + np.max(np.hstack((np.abs(Right[:, 3]), np.abs(Left[:, 3]))))
    
    plt.xlabel(xyz + " (cm)")
    plt.ylabel("<By> (mG)")
    if xyz != "z":
        plt.ylim((-2 * limy, 2 * limy))
    plt.title("<By> vs "+ xyz +": " + dataName + titleAdd)
    if legend == True:
        plt.legend(loc="upper left")
    plt.ioff()
    plt.savefig(filePath + "data_" + dataName + saveLoc + "By_" + xyz + "-deltaB_" + dataName + fileDelim, dpi=500)
    #plt.show()
    
    # <Bz> vs. x
    fig3 = plt.figure()
    plt.ion()
    plt.scatter(Right[:, 0] - T0, Right[:, 5],  color=rColor, label="R = "+str(plus))
    plt.scatter(Left[:, 0] + T0, Left[:, 5], color=lColor, label="R = "+str(minus))

    plt.errorbar(Right[:, 0] - T0, Right[:, 5], yerr=Right[:, 6],fmt="o", color=rColor)
    plt.errorbar(Left[:, 0] + T0, Left[:, 5], yerr=Left[:, 6], fmt="o", color=lColor)

    limz = np.max(np.hstack((Right[:, 6], Left[:, 6]))) + np.max(np.hstack((np.abs(Right[:, 5]), np.abs(Left[:, 5]))))

    plt.xlabel(xyz + " (cm)")
    plt.ylabel("<Bz> (mG)")
    if xyz != "z":
        plt.ylim((-2 * limz, 2 * limz))
    plt.title("<Bz> (vertical) vs "+ xyz +": " + dataName + titleAdd)
    if legend == True:
        plt.legend(loc="upper center")
    plt.ioff()
    plt.savefig(filePath + "data_" + dataName + saveLoc + "Bz_" + xyz + "-deltaB_" + dataName + fileDelim, dpi=500)
    #plt.show()
    
    #<|B|> vs. x
    fig4 = plt.figure()
    plt.ion()
    plt.scatter(Right[:, 0] - T0, Right[:, 7],  color=rColor, label="R = "+str(plus))
    plt.scatter(Left[:, 0] + T0, Left[:, 7], color=lColor, label="R = "+str(minus))

    plt.errorbar(Right[:, 0] - T0, Right[:, 7], yerr=Right[:, 8],fmt="o", color=rColor)
    plt.errorbar(Left[:, 0] + T0, Left[:, 7], yerr=Left[:, 8], fmt="o",color=lColor)

    plt.xlabel(xyz + " (cm)")
    plt.ylabel("<|B|> (mG)")
    plt.title("<|B|> vs "+ xyz +": " + dataName + titleAdd)
    if legend == True:
        plt.legend(loc="upper center")
    plt.ioff()
    plt.savefig(filePath + "data_" + dataName + saveLoc + "B_" + xyz + "-deltaB_" + dataName + fileDelim, dpi=500)
    #plt.show()
    
    return
    
def make_all_plots(filepath, paramArray, Fixed=True):
    print("Parameter array: ")
    print(paramArray)
    print("Parameter array shape: ", paramArray.shape)
    for n in range(4, 5):
        dataname = paramArray[n][0]
        plusX = paramArray[n][1]
        minX = paramArray[n][2]
        plusY = paramArray[n][3]
        minY = paramArray[n][4]
        V1 = paramArray[n][5]
        T0 = paramArray[n][10]

        xRight, xLeft, yRight, yLeft, zAll = complete_data(filepath, dataname, plusX, minX, plusY, minY, V1, T0, fixed=Fixed)
        xRight2, xLeft2, yRight2, yLeft2, zAll2 = complete_data(filepath, dataname, plusX, minX, plusY, minY, V1, T0, fixed=False)

        ################
        #this is where i will have to compute X0 and Y0, if T0 term has been removed from deltaB_general.py"
        ###############
        #right data
        BvalsR = xRight[:, 7]
        distvalsR = xRight[:, 0]
        weightsRight = np.reciprocal(xRight[:, 8]) # 1/error
        valsRight, covRight = np.polyfit(distvalsR, BvalsR, 2, full=False, w=weightsRight, cov = True)
        valsRightP, statsRight = P.polyfit(distvalsR, BvalsR, 2, full=True, w=weightsRight)
        print("Right fit vals: ", valsRight)
        print("Right stats: ", statsRight)
        print("Right cov: ", covRight)
        
        minRight = vertex(valsRight)
        
        fig1 = plt.figure()
        plt.ion()
        plt.scatter(distvalsR, BvalsR, color = "red")
        plt.scatter([minRight], [quadratic(minRight, valsRight)], color="black")
        plt.plot(distvalsR, quadratic(distvalsR, valsRight))
        plt.title("<|B|> vs. x Center-Estimation Fit: " + dataname)
        plt.xlabel("x (cm)")
        plt.ylabel("<|B|>")
        plt.ioff()
        plt.savefig(filepath + "data_" + dataname + "/fitPlots/right-cent_est_fit" + dataname + ".png", dpi=500)
        #plt.show()
        plt.close(fig1)
        
        #left data
        BvalsL = xLeft[:, 7]
        distvalsL = xLeft[:, 0]
        weightsLeft = np.reciprocal(xLeft[:, 8])
        valsLeft,  covLeft = np.polyfit(distvalsL, BvalsL, 2, full=False, w=weightsLeft, cov=True)
        valsLeftP,  statsLeft = P.polyfit(distvalsL, BvalsL, 2, full=True, w=weightsLeft)
        print("Left fit vals: ", valsLeft)
        print("Left stats: ", statsLeft)
        print("Left cov: ", covLeft)
        
        
        minLeft = vertex(valsLeft)
        print("right, left mins: ", [minRight, minLeft])
        
        fig2 = plt.figure()
        plt.ion()
        plt.scatter(distvalsL, BvalsL, color = "blue")
        plt.scatter([minLeft], [quadratic(minLeft, valsLeft)], color="black")
        plt.plot(distvalsL, quadratic(distvalsL, valsLeft))
        plt.title("<|B|> vs. x Center-Estimation Fit: " + dataname)
        plt.xlabel("x (cm)")
        plt.ylabel("<|B|>")
        plt.ioff()
        plt.savefig(filepath + "data_" + dataname + "/fitPlots/left-cent_est_fit" + dataname + ".png", dpi=500)
        #plt.show()
        plt.close(fig2)
        
        #both datasets combined
        Bvals = np.hstack((BvalsR, BvalsL))
        distvals = np.hstack((distvalsR, distvalsR))
        weights = np.hstack((weightsRight, weightsLeft))
        vals,  cov = np.polyfit(distvals, Bvals, 2, full=False, w=weights, cov=True)
        valsP,  stats = P.polyfit(distvals, Bvals, 2, full=True, w=weights)
        print("Both fit vals: ", vals)
        print("Joint stats: ", stats)
        print("Joint cov: ", cov)
        
        minBoth = vertex(vals)
        print("both min: ", minBoth)
        
        fig3 = plt.figure()
        plt.ion()
        plt.scatter(distvalsR, BvalsR, color="red")
        plt.scatter(distvalsR, BvalsL, color="blue")
        plt.scatter([minBoth], [quadratic(minBoth, vals)], color="black")
        plt.plot(distvalsR, quadratic(distvalsR, vals))
        plt.title("<|B|> vs. x Center-Estimation Fit: " + dataname)
        plt.xlabel("x (cm)")
        plt.ylabel("<|B|>")
        plt.ioff()
        plt.savefig(filepath + "data_" + dataname + "/fitPlots/both-cent_est_fit" + dataname + ".png", dpi=500)
        #plt.show()
        plt.close(fig3)
        
        plt.close('all')
        
        T0 = T0 * (22.7 /30000)
        #lets have the shiftvalue be the mean of the two...idk
        T0est = np.mean([np.abs(minRight), np.abs(minLeft)])
        print("T0 estimate from average: ", T0est)
        print("T0 from joint estimate: ", minBoth)
        print("T0 umit: ", T0)
        
        #uncertainties in T0 estimates
        print("T0 error Right: ", T0Err(valsRight, covRight))
        print("T0 error Left: ", T0Err(valsLeft, covLeft))
        print("T0 error Joint: ", T0Err(vals, cov))
        
        ###################
        
        #the T0 correction and errors are made in the Plots function

        #Plots(xRight, xLeft, "red", "maroon", dataname, filepath, "x", T0, plus=plusX, minus=minX, fixed=Fixed)

        #Plots(yRight, yLeft, "red", "maroon", dataname, filepath, "y",  T0, plus=plusY, minus=minY, fixed=Fixed)

        #Plots(zAll, zAll, "maroon", "maroon", dataname, filepath, "z", T0, legend=False, fixed=Fixed)
        
        ComparisonPlots(xRight, xLeft, xRight2, xLeft2, ["red", "blue"], ["maroon", "darkblue"], dataname, filepath, "x", T0, plus=plusX, minus=minX, fixed=Fixed)

        ComparisonPlots(yRight, yLeft, yRight2, yLeft2, ["red", "blue"], ["maroon", "darkblue"], dataname, filepath, "y",  T0, plus=plusY, minus=minY, fixed=Fixed)

        ComparisonPlots(zAll, zAll, zAll2, zAll2, ["maroon", "darkblue"], ["maroon", "darkblue"], dataname, filepath, "z", T0, legend=False, fixed=Fixed)
    return
    
#####################
def overlay_all_plots(filepath, paramArray, Fixed=True):
    filepath = "/Users/alexandraklipfel/Desktop/senior_thesis/"
    print("Parameter array: ")
    print(paramArray)
    print("Parameter array shape: ", paramArray.shape)

    #dataname = paramArray[n][0]
    #plusX = paramArray[n][1]
    #minX = paramArray[n][2]
    #plusY = paramArray[n][3]
    #minY = paramArray[n][4]
    #V1 = paramArray[n][5]
    #T0 = paramArray[n][10]

    n = 0
    xRight0, xLeft0, yRight0, yLeft0, zAll0 = complete_data(filepath, paramArray[n][0], paramArray[n][1], paramArray[n][2], paramArray[n][3], paramArray[n][4], paramArray[n][5], paramArray[n][10], fixed=Fixed)
    n = 1
    xRight1, xLeft1, yRight1, yLeft1, zAll1 = complete_data(filepath, paramArray[n][0], paramArray[n][1], paramArray[n][2], paramArray[n][3], paramArray[n][4], paramArray[n][5], paramArray[n][10], fixed=Fixed)
    n = 2
    xRight2, xLeft2, yRight2, yLeft2, zAll2 = complete_data(filepath, paramArray[n][0], paramArray[n][1], paramArray[n][2], paramArray[n][3], paramArray[n][4], paramArray[n][5], paramArray[n][10], fixed=Fixed)
    
    n = 3
    xRight3, xLeft3, yRight3, yLeft3, zAll3 = complete_data(filepath, paramArray[n][0], paramArray[n][1], paramArray[n][2], paramArray[n][3], paramArray[n][4], paramArray[n][5], paramArray[n][10], fixed=Fixed)
    n = 4
    xRight4, xLeft4, yRight4, yLeft4, zAll4 = complete_data(filepath, paramArray[n][0], paramArray[n][1], paramArray[n][2], paramArray[n][3], paramArray[n][4], paramArray[n][5], paramArray[n][10], fixed=Fixed)
    n = 5
    xRight5, xLeft5, yRight5, yLeft5, zAll5 = complete_data(filepath, paramArray[n][0], paramArray[n][1], paramArray[n][2], paramArray[n][3], paramArray[n][4], paramArray[n][5], paramArray[n][10], fixed=Fixed)
    
    n = 6
    xRight6, xLeft6, yRight6, yLeft6, zAll6 = complete_data(filepath, paramArray[n][0], paramArray[n][1], paramArray[n][2], paramArray[n][3], paramArray[n][4], paramArray[n][5], paramArray[n][10], fixed=Fixed)
    n = 7
    xRight7, xLeft7, yRight7, yLeft7, zAll7 = complete_data(filepath, paramArray[n][0], paramArray[n][1], paramArray[n][2], paramArray[n][3], paramArray[n][4], paramArray[n][5], paramArray[n][10], fixed=Fixed)
    n = 8
    xRight8, xLeft8, yRight8, yLeft8, zAll8 = complete_data(filepath, paramArray[n][0], paramArray[n][1], paramArray[n][2], paramArray[n][3], paramArray[n][4], paramArray[n][5], paramArray[n][10], fixed=Fixed)
    
    n = 9
    xRight9, xLeft9, yRight9, yLeft9, zAll9 = complete_data(filepath, paramArray[n][0], paramArray[n][1], paramArray[n][2], paramArray[n][3], paramArray[n][4], paramArray[n][5], paramArray[n][10], fixed=Fixed)
    n = 10
    xRight10, xLeft10, yRight10, yLeft10, zAll10 = complete_data(filepath, paramArray[n][0], paramArray[n][1], paramArray[n][2], paramArray[n][3], paramArray[n][4], paramArray[n][5], paramArray[n][10], fixed=Fixed)
    n = 11
    xRight11, xLeft11, yRight11, yLeft11, zAll11 = complete_data(filepath, paramArray[n][0], paramArray[n][1], paramArray[n][2], paramArray[n][3], paramArray[n][4], paramArray[n][5], paramArray[n][10], fixed=Fixed)
    n = 12
    xRight12, xLeft12, yRight12, yLeft12, zAll12 = complete_data(filepath, paramArray[n][0], paramArray[n][1], paramArray[n][2], paramArray[n][3], paramArray[n][4], paramArray[n][5], paramArray[n][10], fixed=Fixed)

    cs = ["black", "maroon", "red", "orange", "gold", "yellow", "yellowgreen", "green", "blue", "darkblue", "purple", "orchid", "pink"]
    
    #<|B|> vs. x
    fig = plt.figure()
    plt.ion()
    
    n = 0
    T0 = paramArray[n][10] * (22.7 / 30000)
    name = paramArray[n][0]
    name = "Date: " + name[0:4] + ", Time: " + name[5:7] + ":" + name[8:10]
    plt.scatter(xRight0[:, 0] - T0, xRight0[:, 7],  color=cs[n], label=name)
    plt.scatter(xLeft0[:, 0] + T0, xLeft0[:, 7], color=cs[n])
    plt.errorbar(xRight0[:, 0] - T0, xRight0[:, 7], yerr=xRight0[:, 8],fmt="o", color=cs[n])
    plt.errorbar(xLeft0[:, 0] + T0, xLeft0[:, 7], yerr=xLeft0[:, 8], fmt="o",color=cs[n])
    
    n = 1
    T0 = paramArray[n][10] * (22.7 / 30000)
    name = paramArray[n][0]
    name = "Date: " + name[0:4] + ", Time: " + name[5:7] + ":" + name[8:10]
    plt.scatter(xRight1[:, 0] - T0, xRight1[:, 7],  color=cs[n], label=name)
    plt.scatter(xLeft1[:, 0] + T0, xLeft1[:, 7], color=cs[n])
    plt.errorbar(xRight1[:, 0] - T0, xRight1[:, 7], yerr=xRight1[:, 8],fmt="o", color=cs[n])
    plt.errorbar(xLeft1[:, 0] + T0, xLeft1[:, 7], yerr=xLeft1[:, 8], fmt="o",color=cs[n])
    
    n = 2
    T0 = paramArray[n][10] * (22.7 / 30000)
    name = paramArray[n][0]
    name = "Date: " + name[0:4] + ", Time: " + name[5:7] + ":" + name[8:10]
    plt.scatter(xRight2[:, 0] - T0, xRight2[:, 7],  color=cs[n], label=name)
    plt.scatter(xLeft2[:, 0] + T0, xLeft2[:, 7], color=cs[n])
    plt.errorbar(xRight2[:, 0] - T0, xRight2[:, 7], yerr=xRight2[:, 8],fmt="o", color=cs[n])
    plt.errorbar(xLeft2[:, 0] + T0, xLeft2[:, 7], yerr=xLeft2[:, 8], fmt="o",color=cs[n])

    n = 3
    T0 = paramArray[n][10] * (22.7 / 30000)
    name = paramArray[n][0]
    name = "Date: " + name[0:4] + ", Time: " + name[5:7] + ":" + name[8:10]
    plt.scatter(xRight3[:, 0] - T0, xRight3[:, 7],  color=cs[n], label=name)
    plt.scatter(xLeft3[:, 0] + T0, xLeft3[:, 7], color=cs[n])
    plt.errorbar(xRight3[:, 0] - T0, xRight3[:, 7], yerr=xRight3[:, 8],fmt="o", color=cs[n])
    plt.errorbar(xLeft3[:, 0] + T0, xLeft3[:, 7], yerr=xLeft3[:, 8], fmt="o",color=cs[n])
    
    n = 4
    T0 = paramArray[n][10] * (22.7 / 30000)
    name = paramArray[n][0]
    name = "Date: " + name[0:4] + ", Time: " + name[5:7] + ":" + name[8:10]
    plt.scatter(xRight4[:, 0] - T0, xRight4[:, 7],  color=cs[n], label=name)
    plt.scatter(xLeft4[:, 0] + T0, xLeft4[:, 7], color=cs[n])
    plt.errorbar(xRight4[:, 0] - T0, xRight4[:, 7], yerr=xRight4[:, 8],fmt="o", color=cs[n])
    plt.errorbar(xLeft4[:, 0] + T0, xLeft4[:, 7], yerr=xLeft4[:, 8], fmt="o",color=cs[n])
    
    n = 5
    T0 = paramArray[n][10] * (22.7 / 30000)
    name = paramArray[n][0]
    name = "Date: " + name[0:4] + ", Time: " + name[5:7] + ":" + name[8:10]
    plt.scatter(xRight5[:, 0] - T0, xRight5[:, 7],  color=cs[n], label=name)
    plt.scatter(xLeft5[:, 0] + T0, xLeft5[:, 7], color=cs[n])
    plt.errorbar(xRight5[:, 0] - T0, xRight5[:, 7], yerr=xRight5[:, 8],fmt="o", color=cs[n])
    plt.errorbar(xLeft5[:, 0] + T0, xLeft5[:, 7], yerr=xLeft5[:, 8], fmt="o",color=cs[n])
    
    n = 6
    T0 = paramArray[n][10] * (22.7 / 30000)
    name = paramArray[n][0]
    name = "Date: " + name[0:4] + ", Time: " + name[5:7] + ":" + name[8:10]
    plt.scatter(xRight6[:, 0] - T0, xRight6[:, 7],  color=cs[n], label=name)
    plt.scatter(xLeft6[:, 0] + T0, xLeft6[:, 7], color=cs[n])
    plt.errorbar(xRight6[:, 0] - T0, xRight6[:, 7], yerr=xRight6[:, 8],fmt="o", color=cs[n])
    plt.errorbar(xLeft6[:, 0] + T0, xLeft6[:, 7], yerr=xLeft6[:, 8], fmt="o",color=cs[n])
    
    n = 7
    T0 = paramArray[n][10] * (22.7 / 30000)
    name = paramArray[n][0]
    name = "Date: " + name[0:4] + ", Time: " + name[5:7] + ":" + name[8:10]
    plt.scatter(xRight7[:, 0] - T0, xRight7[:, 7],  color=cs[n], label=name)
    plt.scatter(xLeft7[:, 0] + T0, xLeft7[:, 7], color=cs[n])
    plt.errorbar(xRight7[:, 0] - T0, xRight7[:, 7], yerr=xRight7[:, 8],fmt="o", color=cs[n])
    plt.errorbar(xLeft7[:, 0] + T0, xLeft7[:, 7], yerr=xLeft7[:, 8], fmt="o",color=cs[n])
    
    n = 8
    T0 = paramArray[n][10] * (22.7 / 30000)
    name = paramArray[n][0]
    name = "Date: " + name[0:4] + ", Time: " + name[5:7] + ":" + name[8:10]
    plt.scatter(xRight8[:, 0] - T0, xRight8[:, 7],  color=cs[n], label=name)
    plt.scatter(xLeft8[:, 0] + T0, xLeft8[:, 7], color=cs[n])
    plt.errorbar(xRight8[:, 0] - T0, xRight8[:, 7], yerr=xRight8[:, 8],fmt="o", color=cs[n])
    plt.errorbar(xLeft8[:, 0] + T0, xLeft8[:, 7], yerr=xLeft8[:, 8], fmt="o",color=cs[n])
    
    n = 10
    T0 = paramArray[n][10] * (22.7 / 30000)
    name = paramArray[n][0]
    name = "Date: " + name[0:4] + ", Time: " + name[5:7] + ":" + name[8:10]
    plt.scatter(xRight10[:, 0] - T0, xRight10[:, 7],  color=cs[n], label=name)
    plt.scatter(xLeft10[:, 0] + T0, xLeft10[:, 7], color=cs[n])
    plt.errorbar(xRight10[:, 0] - T0, xRight10[:, 7], yerr=xRight10[:, 8],fmt="o", color=cs[n])
    plt.errorbar(xLeft10[:, 0] + T0, xLeft10[:, 7], yerr=xLeft10[:, 8], fmt="o",color=cs[n])
    
    n = 11
    T0 = paramArray[n][10] * (22.7 / 30000)
    name = paramArray[n][0]
    name = "Date: " + name[0:4] + ", Time: " + name[5:7] + ":" + name[8:10]
    plt.scatter(xRight11[:, 0] - T0, xRight11[:, 7],  color=cs[n], label=name)
    plt.scatter(xLeft11[:, 0] + T0, xLeft11[:, 7], color=cs[n])
    plt.errorbar(xRight11[:, 0] - T0, xRight11[:, 7], yerr=xRight11[:, 8],fmt="o", color=cs[n])
    plt.errorbar(xLeft11[:, 0] + T0, xLeft11[:, 7], yerr=xLeft11[:, 8], fmt="o",color=cs[n])
    
    n = 12
    T0 = paramArray[n][10] * (22.7 / 30000)
    name = paramArray[n][0]
    name = "Date: " + name[0:4] + ", Time: " + name[5:7] + ":" + name[8:10]
    plt.scatter(xRight12[:, 0] - T0, xRight12[:, 7],  color=cs[n], label=name)
    plt.scatter(xLeft12[:, 0] + T0, xLeft12[:, 7], color=cs[n])
    plt.errorbar(xRight12[:, 0] - T0, xRight12[:, 7], yerr=xRight12[:, 8],fmt="o", color=cs[n])
    plt.errorbar(xLeft12[:, 0] + T0, xLeft12[:, 7], yerr=xLeft12[:, 8], fmt="o",color=cs[n])

    plt.xlabel("x (cm)")
    plt.ylabel("<|B|> (mG)")
    plt.title("<|B|> vs x: all overlayed")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.ioff()
    plt.savefig(filepath + "/all_overlayed.png", dpi=500)
    plt.show()
    
    
    return
    
def import_comsol(filePath, fileName):
    """
    INPUT: filepath (string) is path to senior_thesis directory; fileName (string: "Bi(j)" | i,j = x, y, or z)
    FUNCTION: imports the specified comsol file
    OUTPUT: CSarr (nx2 array) of x and y values to plot comsol data.
    """
    file = filePath + "comsol_dat/B017fieldprofile_Metglas_noPB_" + fileName + ".txt"
    CSarr = np.loadtxt(file)
    return CSarr
    
def plot_comsol(i, j, filePath, cutoff, fileName, pArr, symmetric=False, Fixed=True):
    """
    INPUT: i, j (strings: "x", "y" or "z", or "mag" for x) that specify Bi(j); filePath to senior_thesis directory; cutoff (float \in [0, 40 cm]) is the x/y/z value we want to plot up to, symmetric=True will plot left and right data, fileName (string) is the name of the data we will overlay. Note: this comparison is really only meaningful for the Bx data. Perhaps i'll also add functionality to plot |B|...
    FUNCTION: plot all 9 comsol datasets.
    OUTPUT: none
    """
    if Fixed == True:
        titleAdd = "FIXED"
        fileDelim = "corrected-FIXED.png"
    elif Fixed==False:
        titleAdd = " "
        fileDelim = "corrected.png"
    #import comsol data
    if i=="mag":
        dat = import_comsol(filePath, "Bx(" + j + ")")
    else:
        dat = import_comsol(filePath, "B" + i + "(" + j + ")")
    if symmetric == True:
        dat2 = np.hstack((np.array([-dat[:, 0]]).T, np.array([dat[:, 1]]).T))
        #print(dat2)
        dat = np.vstack([dat, dat2])
    #print(dat)
    counter = 0
    
    plottingDat = np.zeros(2)
    for k in range(dat.shape[0]):
        #print("k: ", k)
        if (np.abs(dat[k, 0]) <= cutoff / 100):
            #print("yes")
            #print(dat[k, 1])
            plottingDat = np.vstack([plottingDat, dat[k, :]])
            
    plottingDat = plottingDat[1:, :]
    plottingDat = plottingDat[plottingDat[:, 0].argsort(), :]
    #a = a[a[:, 0].argsort()]
    
    #import actual data for comparison
    colVal = 0
    if i=="x":
        colVal = 1
    if i=="y":
        colVal = 3
    if i=="z":
        colVal = 5
    if i=="mag":
        colVal = 7
    #first grab the parameter values from the parameter file

    print(j)
    
    for k in range(pArr.shape[0]):
        if pArr[k][0] == fileName:
            T0 = pArr[k][10] * (22.7 /30000)
            #complete_data(filepath, dataname, plusX, minX, plusY, minY, V1, T0, fixed=Fixed)
            xRight, xLeft, yRight, yLeft, zAll = complete_data(filePath, fileName, pArr[k][1], pArr[k][2], pArr[k][3], pArr[k][4], pArr[k][5], T0, fixed=Fixed)
            if j=="x":
                print("yes")
                #if i!= "mag":
                xdat = np.hstack([xRight[:, 0] - T0, xLeft[:, 0] + T0])
                ydat = np.hstack([xRight[:, colVal], xLeft[:, colVal]])
                errs = np.hstack([xRight[:, colVal + 1], xLeft[:, colVal + 1]])
                #if i== "mag":
                    #X = np.hstack([xRight[:, 1], xLeft[:, 1]])
                    #Y = np.hstack([xRight[:, 3], xLeft[:, 3]])
                    #Z = np.hstack([xRight[:, 5], xLeft[:, 5]])
                    #ydat = np.sqrt(np.square(X) + np.square(Y) + np.square(Z))
                    #ydat = np.hstack([xRight[:, 7], xLeft[:, 7]])
                    #xdat = np.hstack([xRight[:, 0] - T0, xLeft[:, 0] + T0])
                    #errs =
                    
                    
                Bvals = ydat
                distvals = xdat
                weights = np.reciprocal(errs) # 1/error
                vals, cov = np.polyfit(distvals, Bvals, 2, full=False, w=weights, cov = True)
                valsP, stats = P.polyfit(distvals, Bvals, 2, full=True, w=weights)
                print("Data Fit vals: ", vals)
                print("Data stats: ", stats)
                print("Data cov: ", cov)
                
                Terr = T0Err(vals, cov)
                min = vertex(vals)
                fval = quadratic(min, vals)
                    
                    
                #rescale the comsol data using minima
                minDat = np.min(ydat)
                minCS = np.min(plottingDat[:, 1])
                ### FIX
                ydatCS = plottingDat[:, 1] * (fval / minCS)
                xdatCS = plottingDat[:, 0] * 100
            elif j=="y":
                xdat = np.hstack([yRight[:, 0] - T0, yLeft[:, 0] + T0])
                ydat = np.hstack([yRight[:, colVal], yLeft[:, colVal]])
                errs = np.hstack([yRight[:, colVal + 1], yLeft[:, colVal + 1]])
                
                Bvals = ydat
                distvals = xdat
                weights = np.reciprocal(errs) # 1/error
                vals, cov = np.polyfit(distvals, Bvals, 2, full=False, w=weights, cov = True)
                valsP, stats = P.polyfit(distvals, Bvals, 2, full=True, w=weights)
                print("Data Fit vals: ", vals)
                print("Data stats: ", stats)
                print("Data cov: ", cov)
                
                Terr = T0Err(vals, cov)
                min = vertex(vals)
                fval = quadratic(min, vals)
                
                
                #rescale the comsol data using minima
                minDat = np.min(ydat)
                minCS = np.min(plottingDat[:, 1])
                ####### FIX
                ydatCS = plottingDat[:, 1] * (fval / minCS)
                xdatCS = plottingDat[:, 0] * 100
            elif j=="z":
                xdat = zAll[:, 0]
                ydat = zAll[:, colVal]
                errs = zAll[:, colVal + 1]
                
                Bvals = ydat
                distvals = xdat
                
                weights = np.reciprocal(errs) # 1/error
                vals, cov = np.polyfit(distvals, Bvals, 2, full=False, w=weights, cov = True)
                valsP, stats = P.polyfit(distvals, Bvals, 2, full=True, w=weights)
                print("Data Fit vals: ", vals)
                print("Data stats: ", stats)
                print("Data cov: ", cov)
                
                Terr = T0Err(vals, cov)
                min = vertex(vals)
                fval = quadratic(min, vals)
                
                #rescale the comsol data using minima
                maxDat = np.max(ydat)
                maxCS = np.max(plottingDat[:, 1])
                ydatCS = plottingDat[:, 1] * (fval / maxCS)
                xdatCS = plottingDat[:, 0] * 100
                
    # CS fit
    CSvals, CScov = np.polyfit(xdatCS, ydatCS, 2, full=False, cov = True)
    CSvalsP, CSstats = P.polyfit(xdatCS, ydatCS, 2, full=True)
    print("Comsol Fit vals: ", CSvals)
    print("Comsol stats: ", CSstats)
    print("Comsol cov: ", CScov)
    
    
    
    fig = plt.figure()
    plt.ion()
    plt.scatter(xdat - min, ydat,  color="maroon")
    plt.errorbar(xdat - min, ydat, xerr=Terr, yerr=errs, fmt="o", color="maroon")
    
    plt.scatter(xdatCS, ydatCS, color="gray")
    plt.plot(xdatCS, quadratic(xdatCS, CSvals), color="gray")
    plt.plot(np.sort(xdat) - min, quadratic(np.sort(xdat), vals), color="red")
    plt.xlabel(j + " (cm)")
    plt.ylabel("B" + i)
    plt.title("B" + i + "(" + j + ") comsol comparison: " + fileName + titleAdd)
    plt.ioff()
    plt.savefig(filePath + "/comsol_dat/"+ fileName + "/B_" + i + "(" + j + ")_" + dataName + fileDelim, dpi=500)
    #plt.show()
    plt.close(fig)
    print("min: ", min)
    print("T0 uncertainty: ", Terr)
    
    print([np.sqrt(cov[0,0]), np.sqrt(cov[1,1]), np.sqrt(cov[2,2])])
    print(vals)
    """
    #compare fit values
    fig2 = plt.figure()
    plt.ion()
    plt.scatter([1], vals[0], color="red")
    plt.errorbar([1], vals[0], yerr=[np.sqrt(cov[0,0])], fmt = "o", color="red")
    plt.scatter([1], CSvals[0], color="gray")
    plt.errorbar([1], CSvals[0], yerr=[np.sqrt(CScov[0,0])], fmt = "o", color="gray")
    plt.ioff()
    plt.show()
    
    fig3 = plt.figure()
    plt.ion()
    plt.scatter([1], vals[1], color="red")
    plt.errorbar([1], vals[1], yerr=[np.sqrt(cov[1,1])], fmt = "o", color="red")
    plt.scatter([1], CSvals[1], color="gray")
    plt.errorbar([1], CSvals[1], yerr=[np.sqrt(CScov[1,1])], fmt = "o", color="gray")
    plt.ioff()
    plt.show()
    
    fig3 = plt.figure()
    plt.ion()
    plt.scatter([1], vals[2], color="red")
    plt.errorbar([1], vals[2], yerr=[np.sqrt(cov[2,2])], fmt = "o", color="red")
    plt.scatter([1], CSvals[2], color="gray")
    plt.errorbar([1], CSvals[2], yerr=[np.sqrt(CScov[2,2])], fmt = "o", color="gray")
    plt.ioff()
    plt.show()
    """
    return
    
#----------------------------

paramFile = "/Users/alexandraklipfel/Desktop/senior_thesis/general_analysis/parameters.txt"
paramArray = np.genfromtxt(paramFile, dtype=['U11', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8'])

filepath = "/Users/alexandraklipfel/Desktop/senior_thesis/"

#plot_comsol("mag", "x", filepath, 20, "4-05_21-36", paramArray, symmetric=True)
#plot_comsol("mag", "y", filepath, 20,  "4-05_21-36", paramArray, symmetric=True)
#plot_comsol("mag", "z", filepath, 20, "4-05_21-36", paramArray)


#make_all_plots(filepath, paramArray, Fixed=False)
#make_all_plots(filepath, paramArray, Fixed=True)


#for i in range(1, 13):
    #name = paramArray[i][0]
    #print(name)
    #plot_comsol("mag", "z", filepath, 15, name, paramArray, symmetric=True, Fixed=True)

overlay_all_plots(filepath, paramArray, Fixed=True)
