# imports rotation sweep data file and computes offset

# to do: propagate and calculate new errors on field values
# create general script to take in a file and perform coord & field transformation

# sweep_out_2-8_good.txt:: four height values, four angle values, 5 trolley values
# plot horizontal slices of field

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
#plt.rcParams["figure.figsize"] = 6,4

file = "/Users/alexandraklipfel/Desktop/senior_thesis/data_3-29/on/output_b0_on.txt"
dat = np.loadtxt(file)
print("Data Shape: ", dat.shape)
length = dat.shape[0]
print("Number of Data Points: ", length)

# set reference values
T0 = 100
R0 = -200
V0 = max(dat[:, 2])

# grab columns
#position
T = dat[:, 0] #1st column
R = dat[:, 1] #2nd column
V = dat[:, 2] #3rd column
#field
BxProbe = dat[:, 3] #4th column
#BxProbe = BxProbe - offsetX
BxProbeErr = dat[:, 4] #5th column
ByProbe = dat[:, 5] #6th column
ByProbeErr = dat[:, 6] #7th column
BzProbe = dat[:, 7] #4th column
#BzProbe = BzProbe - offset_Z
BzProbeErr = dat[:, 8] #5th column

rvals = (22.7 / 30000) * (T - T0)
#phivals = (np.pi / 8000) * (R - R0)
phivals = (np.pi / 8000) * (R - R0) - (np.pi/2)
zvals = -(38.61 / 50000) * (V - V0)
thetavals = (np.pi / 8000) * (R - R0)

# create transfromed array in lab frame
lab_dat = np.zeros(6)

for i in range(length):
    # transform to probe position to cartesian coords
    x = rvals[i] * np.cos(thetavals[i])
    y = rvals[i] * np.sin(thetavals[i])
    z = zvals[i]
    # transform B field with a rotation in probe xz plane
    #Bx = BzProbe[i] * np.cos(phivals[i]) + BxProbe[i] * np.sin(phivals[i])
    #By = BzProbe[i] * np.sin(phivals[i]) - BxProbe[i] * np.cos(phivals[i])
    Bx = BxProbe[i] * np.cos(phivals[i]) - BzProbe[i] * np.sin(phivals[i])
    By = BxProbe[i] * np.sin(phivals[i]) + BzProbe[i] * np.cos(phivals[i])
    Bz = -ByProbe[i]
    lab_dat = np.vstack([lab_dat, [x, y, z, Bx, By, Bz]])
    
# drop the first fow of zeros
lab_dat = lab_dat[1:, :]



# sort array by the z value (third column)
#lab_dat = lab_dat[np.argsort(lab_dat[:, 2])]

pd.DataFrame(lab_dat).to_csv("/Users/alexandraklipfel/Desktop/transformed_on.csv")


#plt.quiver(lab_dat[0:55, 0], lab_dat[0:55, 1], lab_dat[0:55, 3], lab_dat[0:55, 4])
#plt.savefig("horizontal_cut_" + str(0) + ".png")
#plt.show()

#plt.quiver(lab_dat[0:55, 0], lab_dat[0:55, 1], BxProbe[0:55], ByProbe[0:55])
#plt.savefig("horizontal_cut_" + str(0) + ".png")
#plt.show()

#fig = plt.figure()
#ax = fig.gca(projection='3d')

#ax.quiver(lab_dat[:,0], lab_dat[:,1], lab_dat[:,2], lab_dat[:,3], lab_dat[:,4], lab_dat[:,5])
#plt.savefig("3D_vf.png")
#plt.show()

#fig = plt.figure()
#ax = fig.gca(projection='3d')

#ax.quiver(lab_dat[0:19,0], lab_dat[0:19,1], lab_dat[0,2], lab_dat[0:19,3], lab_dat[0:19,4], lab_dat[0:19,5], color="blue")
#ax.quiver(lab_dat[20:39,0], lab_dat[20:39,1], lab_dat[20,2], lab_dat[20:39,3], lab_dat[20:39,4], lab_dat[20:39,5], color="green")
#ax.quiver(lab_dat[40:59,0], lab_dat[40:59,1], lab_dat[40,2], lab_dat[40:59,3], lab_dat[40:59,4], lab_dat[40:59,5], color="orange")
#ax.quiver(lab_dat[60:79,0], lab_dat[60:79,1], lab_dat[60,2], lab_dat[60:79,3], lab_dat[60:79,4], lab_dat[60:79,5], color="red")
#plt.savefig("3D_vf.png")
#plt.show()

heights = np.unique(V)
print(heights)
num = len(heights)

def remove_dat(data, Rval, Vval, spread):
    #data: array of transformed data (2dim)
    keep = np.array([])
    for i in range(len(R)):
        if (((R[i] == Rval) or  (R[i] == Rval+8100) or (R[i] == Rval-8200)) and (V[i] > (Vval - spread)) and (V[i] < (Vval + spread))):
            #print(R[i])
            keep = np.append(keep, i)
    #if len(keep) > 0 :
        #print("successfully got indices!")
    #print(keep)
    return keep
    
def get_rad_dist(index_list):
    dist_list = np.zeros(len(index_list))
    for i in range(len(index_list)):
        x = lab_dat[index_list[i], 0]
        y = lab_dat[index_list[i], 1]
        dist = np.sqrt(x**2 + y**2)
        #if x <= 0:
            #dist = -dist
        j = index_list[i]
        #print(R[j])
        if ((R[j] == -4300) or (R[j] == 7900)):
            dist = -dist
        dist_list[i] = dist
    return dist_list
    
def get_vals(data, Baxis, angle, height, range):
    indices = remove_dat(data, angle, height, range)
    indices = indices.astype(int)
    x = get_rad_dist(indices)
    #print(x)
    if Baxis == "x":
        y = lab_dat[indices, 3]
    if Baxis == "y":
        y = lab_dat[indices, 4]
    if Baxis == "z":
        y = lab_dat[indices, 5]
    #print(y)
    return x, y
    
def get_mags(data, angle, height, spread):
    indices = remove_dat(data, angle, height, spread)
    indices = indices.astype(int)
    x = get_rad_dist(indices)
    y = np.zeros(len(indices))
    for i in range(len(indices)):
        y[i] = np.sqrt(data[indices[i], 3]**2 + data[indices[i], 4]**2 + data[indices[i], 5])
    return x, y
    
#section is fixed
x1 = get_mags(lab_dat, -200, heights[0], 0.01)[0]
y1 = get_mags(lab_dat, -200, heights[0], 0.01)[1]

x2 = get_mags(lab_dat, -200, heights[1], 0.01)[0]
y2 = get_mags(lab_dat, -200, heights[1], 0.01)[1]

x3 = get_mags(lab_dat, -200, heights[2], 0.01)[0]
y3 = get_mags(lab_dat, -200, heights[2], 0.01)[1]


plt.scatter(x1, y1, color="red", label="z=" + str((-38.61 / 50000) * (heights[0] - V0)))
plt.scatter(x2, y2,  color="orange", label="z=" + str((-38.61 / 50000) * (heights[1] - V0)))
plt.scatter(x3, y3,  color="yellow", label="z=" + str((38.61 / 50000) * (heights[2] - V0)))

plt.xlabel("x")
plt.ylabel("|B|")
plt.title("|B| vs x: on")
plt.legend(loc="upper left")
plt.savefig("Bmag_x.png", dpi=1000)
plt.show()
    
################################
#fixed section
x1 = get_vals(lab_dat, "y", -200, heights[0], 0.01)[0]
y1 = get_vals(lab_dat, "y", -200, heights[0], 0.01)[1]

x2 = get_vals(lab_dat, "y", -200, heights[1], 0.01)[0]
y2 = get_vals(lab_dat, "y", -200, heights[1], 0.01)[1]

x3 = get_vals(lab_dat, "y", -200, heights[2], 0.01)[0]
y3 = get_vals(lab_dat, "y", -200, heights[2], 0.01)[1]


plt.scatter(x1, y1, color="red", label="z=" + str((-38.61 / 50000) * (heights[0] - V0)))
plt.scatter(x2, y2,  color="orange", label="z=" + str((-38.61 / 50000) * (heights[1] - V0)))
plt.scatter(x3, y3,  color="yellow", label="z=" + str((38.61 / 50000) * (heights[2] - V0)))

plt.xlabel("radial distance")
plt.ylabel("By")
plt.title("By vs x: on")
plt.legend(loc="lower left")
plt.savefig("By_x.png", dpi=1000)
plt.show()

################################
#section fixed
x1 = get_vals(lab_dat, "x", -200, heights[0], 0.01)[0]
y1 = get_vals(lab_dat, "x", -200, heights[0], 0.01)[1]

x2 = get_vals(lab_dat, "x", -200, heights[1], 0.01)[0]
y2 = get_vals(lab_dat, "x", -200, heights[1], 0.01)[1]

x3 = get_vals(lab_dat, "x", -200, heights[2], 0.01)[0]
y3 = get_vals(lab_dat, "x", -200, heights[2], 0.01)[1]


plt.scatter(x1, y1, color="red", label="z=" + str((-38.61 / 50000) * (heights[0] - V0)))
plt.scatter(x2, y2,  color="orange", label="z=" + str((-38.61 / 50000) * (heights[1] - V0)))
plt.scatter(x3, y3,  color="yellow", label="z=" + str((38.61 / 50000) * (heights[2] - V0)))

plt.xlabel("x")
plt.ylabel("Bx")
plt.title("Bx vs x: on")
plt.legend(loc="upper left")
plt.savefig("Bx_x.png", dpi=1000)
plt.show()

################################
#section fixed
x1 = get_vals(lab_dat, "z", -200, heights[0], 0.01)[0]
y1 = get_vals(lab_dat, "z", -200, heights[0], 0.01)[1]

x2 = get_vals(lab_dat, "z", -200, heights[1], 0.01)[0]
y2 = get_vals(lab_dat, "z", -200, heights[1], 0.01)[1]

x3 = get_vals(lab_dat, "z", -200, heights[2], 0.01)[0]
y3 = get_vals(lab_dat, "z", -200, heights[2], 0.01)[1]


plt.scatter(x1, y1, color="red", label="z=" + str((-38.61 / 50000) * (heights[0] - V0)))
plt.scatter(x2, y2,  color="orange", label="z=" + str((-38.61 / 50000) * (heights[1] - V0)))
plt.scatter(x3, y3,  color="yellow", label="z=" + str((38.61 / 50000) * (heights[2] - V0)))

plt.xlabel("x")
plt.ylabel("Bz")
plt.title("Bz vs x: on")
plt.legend(loc="upper left")
plt.savefig("Bz_x.png", dpi=1000)
plt.show()

#################################
#################################
#fixed
x1 = get_vals(lab_dat, "y", 3900, heights[0], 1)[0]
y1 = get_vals(lab_dat, "y", 3900, heights[0], 1)[1]

x2 = get_vals(lab_dat, "y", 3900, heights[1], 1)[0]
y2 = get_vals(lab_dat, "y", 3900, heights[1], 1)[1]

x3 = get_vals(lab_dat, "y", 3900, heights[2], 1)[0]
y3 = get_vals(lab_dat, "y", 3900, heights[2], 1)[1]

plt.scatter(x1, y1, color="red", label="z=" + str((-38.61 / 50000) * (heights[0] - V0)))
plt.scatter(x2, y2,  color="orange", label="z=" + str((-38.61 / 50000) * (heights[1] - V0)))
plt.scatter(x3, y3,  color="yellow", label="z=" + str((38.61 / 50000) * (heights[2] - V0)))

plt.xlabel("y")
plt.ylabel("By")
plt.title("By vs y: on")
plt.legend(loc="upper left")
plt.savefig("By_y.png", dpi=1000)
plt.show()

################################
#fixed
x1 = get_vals(lab_dat, "x", 3900, heights[0], 0.01)[0]
y1 = get_vals(lab_dat, "x", 3900, heights[0], 0.01)[1]

x2 = get_vals(lab_dat, "x", 3900, heights[1], 0.01)[0]
y2 = get_vals(lab_dat, "x", 3900, heights[1], 0.01)[1]

x3 = get_vals(lab_dat, "x", 3900, heights[2], 0.01)[0]
y3 = get_vals(lab_dat, "x", 3900, heights[2], 0.01)[1]

plt.scatter(x1, y1, color="red", label="z=" + str((-38.61 / 50000) * (heights[0] - V0)))
plt.scatter(x2, y2,  color="orange", label="z=" + str((-38.61 / 50000) * (heights[1] - V0)))
plt.scatter(x3, y3,  color="yellow", label="z=" + str((38.61 / 50000) * (heights[2] - V0)))

plt.xlabel("y")
plt.ylabel("Bx")
plt.title("Bx vs y: on")
plt.legend(loc="upper left")
plt.savefig("Bx_y.png", dpi=1000)
plt.show()

################################
#fixed
x1 = get_vals(lab_dat, "z", 3900, heights[0], 0.01)[0]
y1 = get_vals(lab_dat, "z", 3900, heights[0], 0.01)[1]

x2 = get_vals(lab_dat, "z", 3900, heights[1], 0.01)[0]
y2 = get_vals(lab_dat, "z", 3900, heights[1], 0.01)[1]

x3 = get_vals(lab_dat, "z", 3900, heights[2], 0.01)[0]
y3 = get_vals(lab_dat, "z", 3900, heights[2], 0.01)[1]

plt.scatter(x1, y1, color="red", label="z=" + str((-38.61 / 50000) * (heights[0] - V0)))
plt.scatter(x2, y2,  color="orange", label="z=" + str((-38.61 / 50000) * (heights[1] - V0)))
plt.scatter(x3, y3,  color="yellow", label="z=" + str((38.61 / 50000) * (heights[2] - V0)))

plt.xlabel("y")
plt.ylabel("Bz")
plt.title("Bz vs y: on")
plt.legend(loc="upper right")
plt.savefig("Bz_y.png", dpi=1000)
plt.show()

########################
#fixed
x1 = get_mags(lab_dat, 3900, heights[0], 0.01)[0]
y1 = get_mags(lab_dat, 3900, heights[0], 0.01)[1]

x2 = get_mags(lab_dat, 3900, heights[1], 0.01)[0]
y2 = get_mags(lab_dat, 3900, heights[1], 0.01)[1]

x3 = get_mags(lab_dat, 3900, heights[2], 0.01)[0]
y3 = get_mags(lab_dat, 3900, heights[2], 0.01)[1]

plt.scatter(x1, y1, color="red", label="z=" + str((-38.61 / 50000) * (heights[0] - V0)))
plt.scatter(x2, y2,  color="orange", label="z=" + str((-38.61 / 50000) * (heights[1] - V0)))
plt.scatter(x3, y3,  color="yellow", label="z=" + str((38.61 / 50000) * (heights[2] - V0)))

plt.xlabel("y")
plt.ylabel("|B|")
plt.title("|B| vs y: on")
plt.legend(loc="upper right")
plt.savefig("Bmag_y.png", dpi=1000)
plt.show()
