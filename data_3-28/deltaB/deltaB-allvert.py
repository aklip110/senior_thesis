#NOTE: z is vertical

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

#data
fileOff = "/Users/alexandraklipfel/Desktop/senior_thesis/data_3-29/off/V-5000/averaged_off--5000.txt"
datOff = np.loadtxt(fileOff)
print("Data Off Shape: ", datOff.shape)
lengthOff = datOff.shape[0]
print("Number of Off Data Points: ", lengthOff)

fileOn = "/Users/alexandraklipfel/Desktop/senior_thesis/data_3-29/on/V-5000/averaged_on--5000.txt"
datOn = np.loadtxt(fileOn)
print("Data On Shape: ", datOn.shape)
lengthOn = datOn.shape[0]
print("Number of On Data Points: ", lengthOn)



T0 = 100

#subtract the off field values from the on field values
#TO DO: propagate errors

####################################################
#V=-5000
#final data:
# y axis values
dat = datOn.copy()
y = []
bx = []
by = []
bz = []

yleft = []
bxleft = []
byleft = []
bzleft = []
for i in range(lengthOn):
    if (dat[i, 2] == -5000) and (dat[i, 1] == 3900) :
        y += [(22.7 / 30000) * (dat[i,0] - T0)]
        dat[i, 3] = datOn[i, 3] - datOff[i, 3]
        bx += [dat[i, 3]]
        dat[i, 5] = datOn[i, 7] - datOff[i, 7]
        by += [dat[i, 5]]
        dat[i, 7] = -datOn[i, 5] + datOff[i, 5]
        bz += [dat[i, 7]]
    if (dat[i, 2] == -5000) and (dat[i, 1] == -4300) :
        yleft += [(22.7 / 30000) * (-(dat[i,0] - T0))]
        dat[i, 3] = -datOn[i, 3] + datOff[i, 3]
        bxleft += [dat[i, 3]]
        dat[i, 5] = -datOn[i, 7] + datOff[i, 7]
        byleft += [dat[i, 5]]
        dat[i, 7] = -datOn[i, 5] + datOff[i, 5]
        bzleft += [dat[i, 7]]

####################################################
#V=12000
#final data:
# y axis values
dat12 = datOn.copy()
y12 = []
bx12 = []
by12 = []
bz12 = []

yleft12 = []
bxleft12 = []
byleft12 = []
bzleft12 = []
for i in range(lengthOn):
    if (dat12[i, 2] == 12000) and (dat12[i, 1] == 3900) :
        y12 += [(22.7 / 30000) * (dat12[i,0] - T0)]
        dat12[i, 3] = datOn[i, 3] - datOff[i, 3]
        bx12 += [dat12[i, 3]]
        dat12[i, 5] = datOn[i, 7] - datOff[i, 7]
        by12 += [dat12[i, 5]]
        dat12[i, 7] = -datOn[i, 5] + datOff[i, 5]
        bz12 += [dat12[i, 7]]
    if (dat12[i, 2] == 12000) and (dat12[i, 1] == -4300) :
        yleft12 += [(22.7 / 30000) * (-(dat12[i,0] - T0))]
        dat12[i, 3] = -datOn[i, 3] + datOff[i, 3]
        bxleft12 += [dat12[i, 3]]
        dat12[i, 5] = -datOn[i, 7] + datOff[i, 7]
        byleft12 += [dat12[i, 5]]
        dat12[i, 7] = -datOn[i, 5] + datOff[i, 5]
        bzleft12 += [dat12[i, 7]]
        
####################################################
#V=29000
#final data:
# y axis values
dat29 = datOn.copy()
y29 = []
bx29 = []
by29 = []
bz29 = []

yleft29 = []
bxleft29 = []
byleft29 = []
bzleft29 = []
for i in range(lengthOn):
    if (dat29[i, 2] == 29000) and (dat29[i, 1] == 3900) :
        y29 += [(22.7 / 30000) * (dat29[i,0] - T0)]
        dat29[i, 3] = datOn[i, 3] - datOff[i, 3]
        bx29 += [dat29[i, 3]]
        dat29[i, 5] = datOn[i, 7] - datOff[i, 7]
        by29 += [dat29[i, 5]]
        dat29[i, 7] = -datOn[i, 5] + datOff[i, 5]
        bz29 += [dat29[i, 7]]
    if (dat29[i, 2] == 29000) and (dat29[i, 1] == -4300) :
        yleft29 += [(22.7 / 30000) * (-(dat29[i,0] - T0))]
        dat29[i, 3] = -datOn[i, 3] + datOff[i, 3]
        print("on: ", -datOn[i, 3])
        print("off: ", -datOff[i, 3])
        bxleft29 += [dat29[i, 3]]
        dat29[i, 5] = -datOn[i, 7] + datOff[i, 7]
        byleft29 += [dat29[i, 5]]
        dat29[i, 7] = -datOn[i, 5] + datOff[i, 5]
        bzleft29 += [dat29[i, 7]]
        

# make y plots---these are actually labeled "z" because i'm switching to the lab coordinates that Umit's using
#along B0
plt.scatter(y, bx, color="red")
plt.scatter(yleft, bxleft, color="maroon")
plt.scatter(y12, bx12, color="cyan")
plt.scatter(yleft12, bxleft12, color="darkblue")
plt.scatter(y29, bx29, color="yellowgreen")
plt.scatter(yleft29, bxleft29, color="darkgreen")

plt.xlabel("y (cm)")
plt.ylabel("Bx (mG)")
plt.title("delta Bx (along B0) vs y")
plt.savefig("Bx_y-deltaB.png", dpi=1000)
plt.show()

#orthogonal
plt.scatter(y, by, color="red")
plt.scatter(yleft, byleft, color="maroon")
plt.scatter(y12, by12, color="cyan")
plt.scatter(yleft12, byleft12, color="darkblue")
plt.scatter(y29, by29, color="yellowgreen")
plt.scatter(yleft29, byleft29, color="darkgreen")

plt.xlabel("y (cm)")
plt.ylabel("By (mG)")
plt.title("delta By vs y")
plt.savefig("By_y-deltaB.png", dpi=1000)
plt.show()
        
#vertical
plt.scatter(y, bz, color="red")
plt.scatter(yleft, bzleft, color="maroon")
plt.scatter(y12, bz12, color="cyan")
plt.scatter(yleft12, bzleft12, color="darkblue")
plt.scatter(y29, bz29, color="yellowgreen")
plt.scatter(yleft29, bzleft29, color="darkgreen")

plt.xlabel("y (cm)")
plt.ylabel("Bz (mG)")
plt.title("delta Bz (vertical) vs y")
plt.savefig("Bz_y-deltaB.png", dpi=1000)
plt.show()

############################################
# y axis values
#V=-5000
datX = datOn.copy()
x = []
bxX = []
byX = []
bzX = []

xleft = []
bxXleft = []
byXleft = []
bzXleft = []
for i in range(lengthOn):
    if (datX[i, 2] == -5000) and (datX[i, 1] == -200) :
        x += [(22.7 / 30000) * (datX[i,0] - T0)]
        datX[i, 3] = datOn[i, 7] - datOff[i, 7]
        bxX += [datX[i, 3]]
        datX[i, 5] = -datOn[i, 3] - -datOff[i, 3]
        byX += [datX[i, 5]]
        datX[i, 7] = -datOn[i, 5] + datOff[i, 5]
        bzX += [datX[i, 7]]
    if (datX[i, 2] == -5000) and (datX[i, 1] == 7900) :
        xleft += [(22.7 / 30000) * (-(datX[i,0] - T0))]
        datX[i, 3] = -datOn[i, 7] + datOff[i, 7]
        bxXleft += [datX[i, 3]]
        datX[i, 5] = datOn[i, 3] - datOff[i, 3]
        byXleft += [datX[i, 5]]
        datX[i, 7] = -datOn[i, 5] + datOff[i, 5]
        bzXleft += [datX[i, 7]]
        
############################################
# y axis values
#V=12000
datX12 = datOn.copy()
x12 = []
bxX12 = []
byX12 = []
bzX12 = []

xleft12 = []
bxXleft12 = []
byXleft12 = []
bzXleft12 = []
for i in range(lengthOn):
    if (datX12[i, 2] == 12000) and (datX12[i, 1] == -200) :
        x12 += [(22.7 / 30000) * (datX12[i,0] - T0)]
        datX12[i, 3] = datOn[i, 7] - datOff[i, 7]
        bxX12 += [datX12[i, 3]]
        datX12[i, 5] = -datOn[i, 3] - -datOff[i, 3]
        byX12 += [datX12[i, 5]]
        datX12[i, 7] = -datOn[i, 5] + datOff[i, 5]
        bzX12 += [datX12[i, 7]]
    if (datX12[i, 2] == 12000) and (datX12[i, 1] == 7900) :
        xleft12 += [(22.7 / 30000) * (-(datX12[i,0] - T0))]
        datX12[i, 3] = -datOn[i, 7] + datOff[i, 7]
        bxXleft12 += [datX12[i, 3]]
        datX12[i, 5] = datOn[i, 3] - datOff[i, 3]
        byXleft12 += [datX12[i, 5]]
        datX12[i, 7] = -datOn[i, 5] + datOff[i, 5]
        bzXleft12 += [datX12[i, 7]]
        
############################################
# y axis values
#V=29000
datX29 = datOn.copy()
x29 = []
bxX29 = []
byX29 = []
bzX29 = []

xleft29 = []
bxXleft29 = []
byXleft29 = []
bzXleft29 = []
for i in range(lengthOn):
    if (datX29[i, 2] == 29000) and (datX29[i, 1] == -200) :
        x29 += [(22.7 / 30000) * (datX29[i,0] - T0)]
        datX29[i, 3] = datOn[i, 7] - datOff[i, 7]
        bxX29 += [datX29[i, 3]]
        datX29[i, 5] = -datOn[i, 3] - -datOff[i, 3]
        byX29 += [datX29[i, 5]]
        datX29[i, 7] = -datOn[i, 5] + datOff[i, 5]
        bzX29 += [datX29[i, 7]]
    if (datX29[i, 2] == 29000) and (datX29[i, 1] == 7900) :
        xleft29 += [(22.7 / 30000) * (-(datX29[i,0] - T0))]
        datX29[i, 3] = -datOn[i, 7] + datOff[i, 7]
        bxXleft29 += [datX29[i, 3]]
        datX29[i, 5] = datOn[i, 3] - datOff[i, 3]
        byXleft29 += [datX29[i, 5]]
        datX29[i, 7] = -datOn[i, 5] + datOff[i, 5]
        bzXleft29 += [datX29[i, 7]]
        
#along B0
plt.scatter(x, bxX, color="red")
plt.scatter(xleft, bxXleft, color="maroon")
plt.scatter(x12, bxX12, color="cyan")
plt.scatter(xleft12, bxXleft12, color="darkblue")
plt.scatter(x29, bxX29, color="yellowgreen")
plt.scatter(xleft29, bxXleft29, color="darkgreen")

plt.xlabel("x (cm)")
plt.ylabel("Bx (mG)")
plt.title("delta Bx (along B0) vs x")
plt.savefig("Bx_x-deltaB.png", dpi=1000)
plt.show()

#orthogonal
plt.scatter(x, byX, color="red")
plt.scatter(xleft, byXleft, color="maroon")
plt.scatter(x12, byX12, color="cyan")
plt.scatter(xleft12, byXleft12, color="darkblue")
plt.scatter(x29, byX29, color="yellowgreen")
plt.scatter(xleft29, byXleft29, color="darkgreen")

plt.xlabel("x (cm)")
plt.ylabel("By (mG)")
plt.title("delta By vs x")
plt.savefig("By_x-deltaB.png", dpi=1000)
plt.show()
        
#vertical
plt.scatter(x, bzX, color="red")
plt.scatter(xleft, bzXleft, color="maroon")
plt.scatter(x12, bzX12, color="cyan")
plt.scatter(xleft12, bzXleft12, color="darkblue")
plt.scatter(x29, bzX29, color="yellowgreen")
plt.scatter(xleft29, bzXleft29, color="darkgreen")

plt.xlabel("x (cm)")
plt.ylabel("Bz (mG)")
plt.title("delta Bz (vertical) vs x")
plt.savefig("Bz_x-deltaB.png", dpi=1000)
plt.show()

# now the new file should be ready to be plotted--note that the errors are currently meaningless and need to be fixed
#np.savetxt("averaged_deltaB.txt", data, delimiter=" ")
