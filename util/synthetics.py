""" Tools for generating synthetic models and inverting them with the Joint Inversion routine

File written by JSB

Classes:
    1. SynModel
        - Contains the values used to construct synthetic data (including the Vs model)
        - Fields:
            Moho_depth               - depth to the moho in km. Float.
            NVG_depth                - depth to the NVG in km (absolute depth, not depth beneath the moho). Float.
            Moho_width               - width of the moho, km
            NVG_width                - width of the NVG, km
            Crust_parameters         - length 4 vector. Format is
                                       [ (Vs, at the top in km/s) (Vs, at the bottom in km/s)
                                       (dVsdz, at the top) (dVsdz, at the bottom) ]
                                       np.array See comments in _make_synthetic_model
            Lithosphere_parameters   - See crust definition. Between moho and NVG (not necessarily literally lithosphere)
            Asthenosphere_parameters - See crust definition, not necessarily literally asthenosphere. Continues to 400 km
            disperion_sig (etc)      - error term on data. Used for bootstrapping.
            periods                  - periods at which you want to evaluate disperions
"""

import typing
import matplotlib.pyplot as plt
from util import define_models
from util import mineos
from util import inversion
import numpy as np
from disba import PhaseDispersion
import csv

class SynModel(typing.NamedTuple):
    Moho_depth:               float
    Moho_width:               float
    NVG_depth:                float
    NVG_width:                float
    VpVs:                     float
    Crust_parameters:         np.array
    Lithosphere_parameters:   np.array
    Asthenosphere_parameters: np.array
    dispersion_sig:           float
    Ps_tt_sig:                float
    Sp_tt_sig:                float
    Ps_amp_sig:               float
    Sp_amp_sig:               float
    Sn_sig:                   float
    periods:                  np.array
    lithosphere_order:        int = 2
    melt_layer:               np.array = np.array([])

def make_synthetic_model(synm:SynModel):
    """This takes a SynModel, builds it, gets the observations, and returns the
    observations and the model (if you want it). Defined as seperate function in
    case I Want to make examples elsewhere

    How this works: The Vs in each layer is the sum of the first four chebyshev
        polynomials of the first kind. These are
        T0(x) = 1
        T1(x) = x
        T2(x) = 2x^2 - 1
        T3(x) = 4x^3 - 3x

    which are scaled by coefficients (a0, a1, a2, a3)

    The four boundary conditions you give for a layers are Vs at top,bottom and
    the derivative at the top and bottom. This script fits the four chebyshev polynomials
    to give a smooth Vs function between the top and bottom by solving for (a0-3)

    How to solve for the coefficients:
        First nondimensionalize so that the layer has depths ẑ from (-1 - 1) with a V̂ from (-1 - 1)
        ie

        ẑ = 2(z-z0)/Δz - 1; V̂ = 2(V - V0)/ΔV - 1

        Apply this to the depths of the layers and the velocity at the top and bottom.
        Then nondimensionalize the derviatives at the top and bottom (δ̂0, δ̂1) by
        δ̂0 = δ0*(Δz/ΔV)
        δ̂1 = δ1*(Δz/ΔV)

        Now we can use chebyshev polynomials (orthongonal from [-1 1; -1 1])

        V̂(ẑ) = a0 + a1ẑ + a2(2ẑ^2 - 1) + a3(4ẑ^3 - 3ẑ)

        so that
        δV̂/δẑ = a1 + 4a2ẑ + 12a3ẑ^2 - 3a3

        and the four boundary conditions give

        V̂(-1)     => -1  = a0      - a2
        V̂(1)      =>  1  = a0 + a1 + a2  +  a3
        δV̂/δẑ(-1) => δ̂0  =      a1       - 3a3
        δV̂/δẑ(1)  => δ̂1  =      a1 + 4a2 + 9a3

        yada yada yada, page or so of algebra, you get

        a0 =      + 1/8  δ̂0 - 1/8  δ̂1
        a1 = 9/8  - 1/16 δ̂0 - 1/16 δ̂1
        a2 =      - 1/8  δ̂0 - 1/8  δ̂1
        a3 = -1/8 + 1/16 δ̂0 + 1/16 δ̂1

        Use these coefficients to builds V̂ for ẑ from [-1 1], then redimensionalize to V and z,
            do for each layers and stack it, adjust for thickness of discontinituies. Viola.

        Build it, and get the disperison curve. You can get the RF values directly from the input.

        Future work - could add a n=4 polynomial and define middle velocity or middle gradient or something
            (or both with an n=5 term)

    """

    ΔVsM = np.diff(synm.Crust_parameters[0:2]).item()
    ΔVsL = np.diff(synm.Lithosphere_parameters[0:2]).item()
    ΔVsA = np.diff(synm.Asthenosphere_parameters[0:2]).item()

    #non-dimensionalize
    zHat = np.arange(-1,1.01,0.01) #exactly 200 elements, important later

    Mδ0 = (synm.Crust_parameters[2]*synm.Moho_depth/ΔVsM).item()
    Mδ1 = (synm.Crust_parameters[3]*synm.Moho_depth/ΔVsM).item()

    if synm.lithosphere_order == 2:
        #treat it normally, though you need to handle a special case
        if ΔVsL > 0:
            Lδ0 = (synm.Lithosphere_parameters[2]*(synm.NVG_depth - synm.Moho_depth - synm.Moho_width)/ΔVsL).item()
            Lδ1 = (synm.Lithosphere_parameters[3]*(synm.NVG_depth - synm.Moho_depth - synm.Moho_width)/ΔVsL).item()
        else:
            Lδ0 = 0
            Lδ1 = 0

    elif synm.lithosphere_order == 1:

        #make a line by resetting Lδ0 and Lδ1 to the gradient implied by ΔVsL, then non-dimensionalize. This means its just 1
        Lδ0 = 1
        Lδ1 = 1

    elif synm.lithosphere_order == 0:

        ΔVsL = 0
        Lδ0  = 0
        Lδ1  = 0

    Aδ0 = (synm.Asthenosphere_parameters[2]*(400 - synm.NVG_depth + synm.NVG_width)/ΔVsA).item()
    Aδ1 = (synm.Asthenosphere_parameters[3]*(400 - synm.NVG_depth + synm.NVG_width)/ΔVsA).item()

    Ma = np.array([ (Mδ0/8 - Mδ1/8), (9/8 - Mδ0/16 - Mδ1/16),
        (-Mδ0/8 + Mδ1/8), (-1/8 + Mδ0/16 + Mδ1/16) ])

    La = np.array([ (Lδ0/8 - Lδ1/8), (9/8 - Lδ0/16 - Lδ1/16),
        (-Lδ0/8 + Lδ1/8), (-1/8 + Lδ0/16 + Lδ1/16) ])

    Aa = np.array([ (Aδ0/8 - Aδ1/8), (9/8 - Aδ0/16 - Aδ1/16),
        (-Aδ0/8 + Aδ1/8), (-1/8 + Aδ0/16 + Aδ1/16) ])

    makeVs = lambda a: a[0] + a[1]*zHat + a[2]*(2*zHat**2 - 1) + a[3]*(4*zHat**3 - 3*zHat)

    mVs = makeVs(Ma)
    lVs = makeVs(La)
    aVs = makeVs(Aa)

    #stack into a Vs model while redimensionalizing. Note the funky algebra related to redimensionalizing from [-1 1]
    Vs = np.hstack( [ 0.5*(ΔVsM*(mVs + 1) + 2*synm.Crust_parameters[0]),
            0.5*(ΔVsL*(lVs + 1) + 2*synm.Lithosphere_parameters[0]),
            0.5*(ΔVsA*(aVs + 1) + 2*synm.Asthenosphere_parameters[0])])
    z  = np.hstack([0.5*(synm.Moho_depth*(zHat + 1)),
            0.5*((synm.NVG_depth - (synm.Moho_depth + synm.Moho_width))*(zHat + 1) + 2*(synm.Moho_depth + synm.Moho_width)),
            0.5*((400 - synm.NVG_depth - synm.NVG_width)*(zHat + 1) + 2*(synm.NVG_depth + synm.NVG_width)) ])

    if synm.melt_layer.size > 0:
        ind = np.where(np.logical_and(z>=(synm.NVG_depth + synm.NVG_width), z<(synm.NVG_depth + synm.NVG_width + synm.melt_layer[0])))
        m = (synm.melt_layer[2] - synm.melt_layer[1])/(synm.melt_layer[0])
        for k in ind:
            Vs[k] -= Vs[k]*0.08*(synm.melt_layer[1] + (z[k] - (synm.NVG_depth + synm.NVG_width))*m)

    Δz        = np.diff(z)
    Vs        = Vs[1:]
    #Δz        = np.hstack([ Δz, Δz[-1].item()])
    #Δz        = np.hstack([ 0, Δz ] )
    dinds     = np.ones(Δz.shape)
    dinds[-1] = 0

    modelout = define_models.VsvModel(vsv=Vs, thickness=Δz, boundary_inds=np.array((199, 400)),
                d_inds=np.array( dinds , dtype=bool))

    #now return an evenly spaced version
    thickness, vsv, boundary_inds = define_models._return_evenly_spaced_model(modelout, 6)
    depth_inds = define_models._find_depth_indices(thickness, np.array((0, 400)))

    modelout = define_models.VsvModel(vsv, thickness, boundary_inds, depth_inds)

    #now you need to make it in the format for the actual program
    return modelout

def make_syn_obs(synm:SynModel, model_params:define_models.ModelParams):
    """ Returns predicted disperion and Ps/Sp parameters for your model
        Dispersion comes from a model built with _make_synthetic_model
        Receiver function results are directly calculated from the parameters given.
    """

    model  = make_synthetic_model(synm)

    #get the dispersion
    params = mineos.RunParameters(freq_max = 1000 / min(synm.periods) + 1)
    if model_params.method == 'mineos':

        # Build all of the inputs to the damped least squares
        # Run MINEOS to get phase velocities and kernels
        mineos_model = define_models.convert_vsv_model_to_mineos_model(
            model, model_params
        )

        # Can vary other parameters in MINEOS by putting them as inputs to this call
        # e.g. defaults include l_min, l_max; qmod_path; phase_or_group_velocity
        ph_vel_pred, _ = mineos.run_mineos_and_kernels(
            params, synm.periods, model_params.id
        )

    elif model_params.method == 'disba':

        disba_model = define_models.convert_vsv_model_to_disba_model(
            model, model_params)

        pd = PhaseDispersion(*disba_model.T)
        cpr = pd(synm.periods, mode=0, wave=params.Rayleigh_or_Love.lower())
        ph_vel_pred = cpr.velocity

    else:

        raise Exception("Only two methods are mineos and disba")

    #now receiver functions
    RFdata = inversion.predict_RF_vals(model, model_params)

    if len(model_params.boundaries[1]) == 1:
        RFdata = np.delete(RFdata, [1,3])

    if model_params.head_wave:

        HeadWave = inversion.predict_headwave(model, model_params)

        dvec = np.atleast_2d(np.hstack([ph_vel_pred, RFdata, HeadWave])).T

        if len(model_params.boundaries[1]) == 2:
            σ    = np.atleast_2d(np.hstack([ synm.dispersion_sig, synm.Ps_tt_sig, synm.Sp_tt_sig,
                        synm.Ps_amp_sig, synm.Sp_amp_sig, synm.Sn_sig])).T
        elif len(model_params.boundaries[1]) == 1:
            σ    = np.atleast_2d(np.hstack([ synm.dispersion_sig, synm.Ps_tt_sig,
                        synm.Ps_amp_sig, synm.Sn_sig])).T

    else:

        dvec = np.atleast_2d(np.hstack([ph_vel_pred, RFdata])).T

        if len(model_params.boundaries[1]) == 2:
            σ    = np.atleast_2d(np.hstack([ synm.dispersion_sig, synm.Ps_tt_sig, synm.Sp_tt_sig,
                        synm.Ps_amp_sig, synm.Sp_amp_sig])).T
        elif len(model_params.boundaries[1]) == 1:
            σ    = np.atleast_2d(np.hstack([ synm.dispersion_sig, synm.Ps_tt_sig,
                        synm.Ps_amp_sig])).T

    return dvec, σ, synm.periods, 2

def write_out_syn_model(synm:SynModel, name:str):

    fid = open('./output/synmodels/' + name + ".synmodel", "w")
    fid.write("Order of information is (8 header lines)" + '\n')
    fid.write("Moho depth, km, Moho width, km, NVG depth, km, NVG width, km" + '\n')
    fid.write("Crust Vs top, crust vs bottom, crust gradient top, crust gradient bottom" + '\n')
    fid.write("Lithosphere Vs top, Lithosphere vs bottom, Lithosphere gradient top, Lithosphere gradient bottom" + '\n')
    fid.write("Asthenosphere Vs top, Asthenosphere vs bottom, Asthenosphere gradient top, Asthenosphere gradient bottom" + '\n')
    fid.write("Periods used in s" + '\n')
    fid.write("Phase velocity errors, 1σ, s" + '\n')
    fid.write("Ps tt errror, Sp tt error, Ps amp error, Sp amp error" + '\n')

    fid.write(str(synm.Moho_depth) + "," + str(synm.Moho_width) + "," + str(synm.NVG_depth) + "," + str(synm.NVG_width) + '\n')
    fid.write(','.join(str(e) for e in synm.Crust_parameters) + '\n')
    fid.write(','.join(str(e) for e in synm.Lithosphere_parameters) + '\n')
    fid.write(','.join(str(e) for e in synm.Asthenosphere_parameters) + '\n')
    fid.write(','.join(str(e) for e in synm.periods) + '\n')
    fid.write(','.join(str(e) for e in synm.dispersion_sig) + '\n')
    fid.write(str(synm.Ps_tt_sig) + "," + str(synm.Sp_tt_sig) + "," + str(synm.Ps_amp_sig) + "," + str(synm.Sp_amp_sig))

    fid.close()
