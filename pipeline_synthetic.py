""" Pipeline for running the inversion
"""
import numpy as np
from util import define_models
from util import mineos
from util import inversion
from util import partial_derivatives
from util import weights
from util import constraints
from util import synthetics
import shutil
import sys
import matplotlib.pyplot as plt
import multiprocess as mp
import os
from random import uniform,choices,seed,gauss
import string
from copy import deepcopy

class FailCheckLimit(Exception):
    pass

def run_inversion_syn(name:str, mdep:int, attempts:int, lithorder:int):

    #give it a random id (extremely low collision probability between runs)
    name = name + "." + ''.join(choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=5))

    #check if this was already run for this set
    if os.path.isfile('./output/models/' + name + '-models.csv'):

        print(name + " already run for this project name")
        return 1

    model_params = define_models.ModelParams( depth_limits=np.array([0., mdep]),
        id=name, randomize_model=False, randomize_data=False,
        method='mineos',min_layer_thickness=6,
            roughness = (2,1,1000,2,0),
            to_0_grad = (1,0,0,0,0), head_wave=True, breadth=1)

    #generate a randomized synthetic model subject to conditions
    accept = False

    while not accept:

        """
        synm = synthetics.SynModel(
            Moho_depth = uniform(15, 60),
            Moho_width = 1,
            NVG_depth  = uniform(40, 120),
            NVG_width  = uniform(5, 50),
            VpVs       = gauss(1.76, 0.03),
            Crust_parameters = np.array([ uniform(2,3.5), uniform(3.25, 4.25),
                uniform(0,1e-2), uniform(0,1e-2)]),
            Lithosphere_parameters = np.array([ uniform(4.25,5.0), uniform(4.0,4.5),
                uniform(-5e-2,1e-2), uniform(-2e-2,2e-2)]),
            Asthenosphere_parameters = np.array([ uniform(3.6,4.4), 4.7699,#anchored to PREM at base
                uniform(-5e-3,1e-2), 7e-4]),#anchored to PREM at base
            Ps_tt_sig  = 0.67,#these four values are roughly the mean from Emily's datasets
            Sp_tt_sig  = 0.47,
            Ps_amp_sig = 0.044/4,#note the issue from shen and ritz for the factor of 4
            Sp_amp_sig = 0.0075,
            Sn_sig     = 0.2,#km/s
            periods = np.array([  5.,   6.,   8.,  10.,  12.,  15.,  20.,  25.,  32.,  40.,  50.,
            60.,  80., 100., 120., 140., 180.]),
            dispersion_sig = np.array([0.05,0.05,0.05,0.025,0.025,0.025,0.025,0.025,0.025,0.025,
                                0.028,0.034,0.045,0.056,0.066,0.077,0.098]),
            lithosphere_order = lithorder)
        """
        synm = synthetics.SynModel(
            Moho_depth=40,
            Moho_width=1,
            NVG_depth=75,
            NVG_width=10,
            VpVs=1.76,
            Crust_parameters = np.array([ 3.3, 3.6,
                0.0075, 0.0075]),
            Lithosphere_parameters = np.array([ 4.3, 4.3,
                -0.5, 0]),
            Asthenosphere_parameters = np.array([ 4.0, 4.7699,#anchored to PREM at base
                0.003, 7e-4]),#anchored to PREM at base
            Ps_tt_sig  = 0.67,#these four values are roughly the mean from Emily's datasets
            Sp_tt_sig  = 0.47,
            Ps_amp_sig = 0.044/4,#should be amp or dv????
            Sp_amp_sig = 0.0075,
            Sn_sig     = 0.05,#km/s
            periods = np.array([  5.,   6.,   8.,  10.,  12.,  15.,  20.,  25.,  32.,  40.,  50.,
            60.,  80., 100., 120., 140., 180.]),
            dispersion_sig=np.array([0.05,0.05,0.05,0.025,0.025,0.025,0.025,0.025,0.025,0.025,
                                0.028,0.034,0.045,0.056,0.066,0.077,0.098]),
            lithosphere_order = lithorder, melt_layer = np.array([ 50, 0.5, 0.5 ]))

        #do some tests to see if the model is ok (some of the random values overlap)
        #remove models with 1) lithosphere thinner than 20 km, moho jumps largers that 25%, or positive LAB jumps
        if (100*(synm.Lithosphere_parameters[0] - synm.Crust_parameters[1])/synm.Crust_parameters[1]) > 1 \
            and ((synm.Moho_depth) < synm.NVG_depth) \
            and (100*(synm.Lithosphere_parameters[0] - synm.Crust_parameters[1])/synm.Crust_parameters[1]) < 25 \
            and (100*(synm.Asthenosphere_parameters[0] - synm.Lithosphere_parameters[1])/synm.Lithosphere_parameters[1]) < -1 \
            and (100*(synm.Asthenosphere_parameters[0] - synm.Lithosphere_parameters[1])/synm.Lithosphere_parameters[1]) > -20:#and (100*abs(synm.Lithosphere_parameters[1] - synm.Lithosphere_parameters[0])) < 10 \

            accept = True

        if synm.lithosphere_order == 0: #special cases
            if (100*(synm.Asthenosphere_parameters[0] - synm.Lithosphere_parameters[0])/synm.Lithosphere_parameters[0]) < -2:
                accept = True

    synthetics.write_out_syn_model(synm, name)

    depth = np.linspace(0, mdep, num=20*mdep)

    models = []
    vs     = []

    print("Making the synthetic data for " + name)

    model_params2 = define_models.ModelParams( depth_limits=model_params.depth_limits,
        id=model_params.id, randomize_model=model_params.randomize_model, randomize_data=model_params.randomize_data,
        method=model_params.method,min_layer_thickness=model_params.min_layer_thickness,
            roughness = model_params.roughness,
            to_0_grad = model_params.to_0_grad, head_wave=model_params.head_wave, breadth=model_params.breadth,
            vpv_vsv_ratio = synm.VpVs) #to do vpvs variability

    try:
        synobs = synthetics.make_syn_obs(synm, model_params2)
    except:
        print(name + ' failed to make data, aborting.')
        shutil.rmtree("./output/" + name)
        return 0

    if model_params.randomize_data:
        for k in np.arange(0,len(synobs[0])):
            synobs[0][k] = synobs[0][k] + gauss(0, synobs[1][k])

    if model_params.breadth:

        if model_params.randomize_data:
            sm = define_models.setup_starting_model(model_params, (40, -111), np.array( (synm.Moho_width,max(10, gauss(synm.NVG_width, 15)))))
            #sm = define_models.setup_starting_model(model_params, (40, -111), np.array( [synm.Moho_width] ))
        else:
            sm = define_models.setup_starting_model(model_params, (40, -111), np.array( (synm.Moho_width,synm.NVG_width)))

    else:
        sm = define_models.setup_starting_model(model_params, (40, -111), np.array( (synm.Moho_width,model_params.boundaries[1][1]) ))

    #(model, obs, misfit) = inversion.run_inversion(model_params, synobs, sm, 25)

    #Mineos often fails randomly with file IO, but you can repeat easily without slowing the code down (much)
    failcheck = 0
    while failcheck < attempts:

        try:
            (model, obs, misfit) = inversion.run_inversion(model_params, synobs, sm, 10)
            break
        except:
            failcheck += 1

            if failcheck > attempts:
                print(name + ' failed ' + str(attempts) + ' times in a row, aborting.')
                shutil.rmtree("./output/" + name)
                return 0

            continue

    mz     = np.cumsum(model.thickness)
    vseven = np.interp(depth, mz, model.vsv.flatten())

    synmodel = synthetics.make_synthetic_model(synm)
    synmodelz = np.cumsum(synmodel.thickness)
    synmodelvseven = np.interp(depth, synmodelz, synmodel.vsv.flatten())

    t = np.cumsum(model.thickness)
    interface_z = t[model.boundary_inds]

    plt.figure()
    plt.plot(vseven,depth)
    plt.plot(synmodelvseven,depth,color='gray')
    plt.ylim(200, 0)
    plt.grid()
    plt.hlines(interface_z[0], min(model.vsv),max(model.vsv),linestyles='--',color='black')
    plt.hlines(interface_z[1], min(model.vsv),max(model.vsv), linestyles='--',color='black')
    plt.text(model.vsv[model.boundary_inds[0]+1] - 0.1,interface_z[0] - 5,'Moho depth=' + str(round(interface_z[0])) + ' km')
    plt.text(model.vsv[model.boundary_inds[1]+1] + 0.1,interface_z[1] + 15,'NVG depth=' + str(round(interface_z[1])) + ' km')
    plt.xlabel('Vs, km/s (s)')
    plt.ylabel('Depth, km')
    plt.savefig(name +'.pdf')
    #plt.savefig(name +'.svg')

    if misfit < 4: #rare w/ data
        define_models.save_model(model, name)
    shutil.rmtree("./output/" + name)

    return 1

def main(name:str, nsyns:int=1, mdep:int=400, attempts:int=50, lithorder:int=2):

    for k in np.arange(0, nsyns):
        tmp = run_inversion_syn(name, mdep, attempts,lithorder)
    #pool = mp.Pool(mp.cpu_count())
    #list(pool.map(lambda k:run_inversion_syn(name, mdep, attempts, lithorder), range(0,nsyns)))

if __name__ == '__main__':
    if len(sys.argv)==1:
        print("You need at least a name for the run!")
    elif len(sys.argv)==2:
        main(sys.argv[1]) #argument is the number of randomized synthetics you want to run
    elif len(sys.argv)==3:
        main(sys.argv[1], int(sys.argv[2])) #argument is the number of randomized synthetics you want to run
    elif len(sys.argv)==6:
        main(sys.argv[1], int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]))
