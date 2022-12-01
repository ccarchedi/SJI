""" Pipeline for running the inversion
"""

import numpy as np
from util import define_models
from util import mineos
from util import inversion
from util import partial_derivatives
from util import weights
from util import constraints
import shutil
import sys
import matplotlib.pyplot as plt
import multiprocess as mp
import os

class FailCheckLimit(Exception):
    pass

def run_inversion_loc(name:str, location:tuple, mdep:int, attempts:int):

    name = name + "." + str(location[0]) + "." + str(location[1])

    #check if this was already run for this set
    if os.path.isfile('./output/models/' + name + '.csv'):

        print(name + " already run for this project name")
        return 1

    model_params = define_models.ModelParams(depth_limits=np.array([0., mdep]),
        id=name, randomize_model=False, randomize_data=False,
        method='mineos',min_layer_thickness=6, head_wave=True, breadth=1,
            roughness = (1,1,4,4,0),
            to_0_grad = (0,0,0,0,0), q_model = './data/earth_models/qmod_highQ')

    depth = np.linspace(0, mdep, mdep)

    i = 0
    #Mineos often fails randomly with file IO, but you can repeat easily without slowing the code down
    failcheck = 0

    #(model, obs, misfit) = inversion.run_inversion_TAarray(model_params,location,25)
    sm = define_models.custom_starting_model(model_params, np.array([ 1, 3.5, 4.2, 4.5 ]), np.array([ 0, 4, 12, 21 ]), np.array(model_params.boundaries[1]))

    while failcheck < attempts:

        try:
            (model, obs, misfit) = inversion.run_inversion_ArrayData(model_params,location,sm,25)

            if model is None:
                return

            break
        except:
            failcheck += 1

            if failcheck >= attempts:
                raise FailCheckLimit(name + ' failed ' + str(attempts) + ' times in a row, aborting.')
                return 0

            continue

        i += 1
        failcheck = 0 #restart count

    define_models.save_model(model, name)
    shutil.rmtree("./output/" + name)

    return model

def main(mdep:int=400, attempts:int=5):

    name = "Data8322"
    #lat = np.arange(25,51,0.5)
    #lon = np.arange(-125.,-65,0.5)
    #lat = np.arange(33.,45.5,0.5)
    #lon = np.arange(-119.,-100.5,0.5)

    #LT, LN = np.meshgrid(lat, lon)

    #LT = LT.flatten()
    #LN = LN.flatten()

    depth = np.linspace(0, mdep, num=20*mdep)

    #hardwire a single location here. Don't need to return model if doing a loop

    model = run_inversion_loc(name, (LAT,LON), mdep, attempts)

    mz     = np.cumsum(model.thickness)
    vseven = np.interp(depth, mz, model.vsv.flatten())

    t = np.cumsum(model.thickness)
    interface_z = t[model.boundary_inds]

    plt.figure()
    plt.plot(vseven,depth)
    plt.ylim(mdep, 0)
    plt.grid()
    plt.hlines(interface_z[0], min(model.vsv),max(model.vsv),linestyles='--',color='black')
    plt.hlines(interface_z[1], min(model.vsv),max(model.vsv), linestyles='--',color='black')
    plt.hlines(interface_z[2], min(model.vsv),max(model.vsv), linestyles='--',color='black')
    plt.text(model.vsv[model.boundary_inds[0]+1] - 0.1,interface_z[0] - 5,'B1 depth=' + str(round(interface_z[0])) + ' km')
    plt.text(model.vsv[model.boundary_inds[1]+1] + 0.1,interface_z[1] + 15,'B2 depth=' + str(round(interface_z[1])) + ' km')
    plt.text(model.vsv[model.boundary_inds[2]+1] + 0.1,interface_z[2] + 15,'B3 depth=' + str(round(interface_z[3])) + ' km')
    plt.xlabel('Vs, km/s (s)')
    plt.ylabel('Depth, km')
    plt.savefig(name +'.pdf')

    #for k in np.arange(0, len(LT)):
    #for k in np.arange(0, 1):
    #for k in np.arange(0, len(LT)):
    #    run_inversion_loc(name, (LT[k],LN[k]), mdep, attempts)
    #pool = mp.Pool(mp.cpu_count())
    #list(pool.map(lambda k:run_inversion_loc(name, (LT[k],LN[k]), mdep, attempts), range(len(LT))))

if __name__ == '__main__':

    if len(sys.argv)==1:
        print('Doing a whole run')
        main()
    elif len(sys.argv)==2:
        main(int(sys.argv[1]))
    else:
        print('Check input arugment list')
