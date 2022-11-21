""" Inversion from phase velocities to velocity model.

This is based on MATLAB code from Josh Russell and Natalie Accardo (2019),
and formulated after Geophysical Data Analysis: Discrete Inverse Theory.
(DOI: 10.1016/B978-0-12-397160-9.00003-5) by Bill Menke (2012).

For simplicity, I am only doing this for Rayleigh waves at the moment,
although there are some bits scattered around for Love waves that are commented
out.  This isn't complete, though, and for now I am not going to test those.
In any comments talking about stacking Love and Rayleigh kernels/data etc,
this indicates how it should be done, not that it is being done here.
"""

#import collections
import typing
import numpy as np
import pandas as pd
import time

from util import define_models
from util import mineos
from util import constraints
from util import partial_derivatives
from util import weights
from util import synthetics
from disba import PhaseDispersion

# =============================================================================
#       Run the Damped Least Squares Inversion
# =============================================================================
def run_with_no_inputs():

    model_params = define_models.ModelParams(
        depth_limits=np.array([0., 200.]),
        #boundary_vsv=np.array([3.5, 4.0, 4.2, 4.1]),
        #boundary_widths=np.array([5., 10.]),
        #boundary_depth_uncertainty=np.array([3., 10.,]),
        #boundary_depths=np.array([35., 90]),
        id='test2')

    location = (35, -104)

    #return run_inversion(model_params, loc)
    return run_inversion(model_params, location)

def run_inversion_TAarray(model_params:define_models.ModelParams,
                  location:tuple, n_iterations:int=10) -> (define_models.VsvModel):
    """ Set the inversion running over some number of iterations.
    """

    obs_constraints = constraints.extract_observations(
        location, model_params.id, model_params.boundaries, model_params.vpv_vsv_ratio,
        model_params.randomize_data, model_params.head_wave, model_params.breadth)#jsb added randomize data

    if obs_constraints is None:
        return None, None, None

    breadth = obs_constraints[-1]
    n_boundaries = obs_constraints[-2]
    obs_constraints = obs_constraints[0:-2]

    if n_boundaries == 1:
        #remove the LAB constraint
        model_params = model_params._replace(boundaries=( \
            (model_params.boundaries[0][0],),model_params.boundaries[1][0], model_params.boundaries[2][0]))

    model = define_models.setup_starting_model(model_params, location, breadth)

    if any(np.isnan(model.vsv)):
        return model, obs_constraints, np.Inf

    prevmisfit = 1e9
    flag       = False

    for i in range(n_iterations):
        # Still need to pass model_params as it has info on e.g. vp/vs ratio
        # needed to convert from VsvModel to MINEOS card

        modeln, misfit, pred = _inversion_iteration(model_params, model, obs_constraints)

        #check if you should terminate inverison. Note the zero divide protection (both numerator terms will be zero)
        max_pert = max(max( [ abs((100*(x - y)/(x+1e-5)).item()) for (x,y) in zip(model.thickness,modeln.thickness) if x!=0],
                    [ abs((100*(x - y)/(x+1e-5)).item()) for (x,y) in zip(model.vsv,modeln.vsv)]))

        mis = 1e-3*round((1e3*sum((misfit/obs_constraints[1])**2/len(obs_constraints[1])).item()))

        print(model_params.id + " generated max perturbation of " +
            str(1e-4*round(max_pert*1e4)) + "% with a χ̄^2 of " + str(mis) \
            + " on iteration #" + str(i+1))

        if max_pert > model_params.perturbation_threshold:
            model = modeln
        else:
            break

    return model, obs_constraints, sum((misfit/obs_constraints[1])**2/len(obs_constraints[1])).item() #jsb earlier iterations not saved

def run_inversion(model_params:define_models.ModelParams,
                  obs_constraints:np.array, model:define_models.VsvModel, n_iterations:int=25) -> (define_models.VsvModel):
    """ JSB added. Modified version of run_inversion to use an externally defined starting model
    """

    n_boundaries = obs_constraints[-1]
    obs_constraints = obs_constraints[0:-1]

    if n_boundaries == 1:
        #remove the lab constraint
        model_params = model_params._replace(boundaries=(('Moho',),[3.]))

    if any(np.isnan(model.vsv)):
        print('Nans in starting model')
        return model

    for i in range(n_iterations):
        # Still need to pass model_params as it has info on e.g. vp/vs ratio
        # needed to convert from VsvModel to MINEOS card

        modeln, misfit, pred = _inversion_iteration(model_params, model, obs_constraints)

        #check if you should terminate inverison. Note the zero divide protection (both terms will be zero if needed)
        max_pert = max(max( [ abs((100*(x - y)/(x+1e-5)).item()) for (x,y) in zip(model.thickness,modeln.thickness) if x!=0],
                    [ abs((100*(x - y)/(x+1e-5)).item()) for (x,y) in zip(model.vsv,modeln.vsv)]))

        print(model_params.id + " generated max perturbation of " +
            str(1e-4*round(max_pert*1e4)) + "% with a χ̄^2 of " + str( 1e-3*round((1e3*sum((misfit/obs_constraints[1])**2/len(obs_constraints[1])).item())) ) \
            + " on iteration #" + str(i+1))

        if max_pert > model_params.perturbation_threshold:
            model = modeln
        else:
            break

    return model, obs_constraints, sum((misfit/obs_constraints[1])**2/len(obs_constraints[1])).item() #jsb earlier iterations not saved

def _inversion_iteration(model_params:define_models.ModelParams,
                         model:define_models.VsvModel,
                         obs_constraints:tuple,
                         ) -> define_models.VsvModel:
    """ Run a single iteration of the least squares
    """

    obs, std_obs, periods = obs_constraints

    params = mineos.RunParameters(freq_max = 1000 / (min(periods)) + 1, qmod_path = model_params.q_model)

    if model_params.method == 'mineos':

        # Build all of the inputs to the damped least squares
        # Run MINEOS to get phase velocities and kernels
        mineos_model = define_models.convert_vsv_model_to_mineos_model(
            model, model_params
        )

        # Can vary other parameters in MINEOS by putting them as inputs to this call
        # e.g. defaults include l_min, l_max; qmod_path; phase_or_group_velocity
        ph_vel_pred, kernels = mineos.run_mineos_and_kernels(
            params, periods, model_params.id
        )

    elif model_params.method == 'disba':

        disba_model = define_models.convert_vsv_model_to_disba_model(
            model, model_params)

        pd = PhaseDispersion(*disba_model.T)
        cpr = pd(periods, mode=0, wave=params.Rayleigh_or_Love.lower())
        ph_vel_pred = cpr.velocity

        kernels = partial_derivatives.kernels_from_disba(disba_model, periods, params)

    else:

        raise Exception("Only two methods are mineos and disba")

    kernels = kernels[kernels['z'] <= model_params.depth_limits[1]]

    # Assemble G, p, and d
    G = partial_derivatives._build_partial_derivatives_matrix(
        kernels, model, model_params
    )

    p0 = _build_model_vector(model, model_params.depth_limits)

    if model_params.head_wave:
        predictions = np.concatenate((ph_vel_pred, predict_RF_vals(model, model_params), predict_headwave(model, model_params)))
    else:
        predictions = np.concatenate((ph_vel_pred, predict_RF_vals(model, model_params)))

    #print('G: {}, p: {}, preds: {}'.format(
    #    G.shape, p.shape, predictions.shape
    #)) jsb
    dvec, misfit = _build_data_misfit_vector(obs, predictions, p0, G) #name of d changed to dvec for python reasons, misfit added jsb

    # Remove constraint on Moho strength
    #G = G[:-2, :]#np.vstack((G[:-2, :], G[-1, :]))
    #dvec = dvec[:-2]#np.vstack((d[:-2], d[-1]))
    #std_obs = std_obs[:-2]#np.vstack((std_obs[:-2], std_obs[-1]))

    # Build all of the weighting functions for damped least squares

    W, H_mat, h_vec = (
        weights.build_weighting_damping(std_obs, p0, model, model_params)
    )

    # Perform inversion
    # print('G: {}, p: {}, W: {}, d: {}, H_mat: {}, h_vec: {}'.format(
    #     G.shape, p.shape, W.shape, d.shape, H_mat.shape, h_vec.shape
    # ))

    p_new = _damped_least_squares(p0, G, dvec, W, H_mat, h_vec)

    model = _build_inversion_model_from_model_vector(p_new, model)

    model.vsv[model.vsv < model_params.min_vs] = model_params.min_vs

    thickness, vsv, bi = define_models._return_evenly_spaced_model(
        model, model_params.min_layer_thickness
    )

    return define_models.VsvModel(
            vsv, thickness, bi,
            define_models._find_depth_indices(thickness, model_params.depth_limits)
           ), misfit, predictions #p, G, d, W, H_mat, h_vec

def predict_RF_vals(model:define_models.VsvModel, model_params:define_models.ModelParams): #JSB took _ off for synthetics.py
    """modified by JSB to account for "extra layers" that are not included in the actual model
    """

    travel_time = np.zeros_like(model.boundary_inds).astype(float)
    dV = np.zeros_like(model.boundary_inds).astype(float)

    n = 0
    for ib in model.boundary_inds:
        dV[n] = model.vsv[ib + 1] / model.vsv[ib] - 1
        for i in range(ib):
            travel_time[n] += model.thickness[i + 1] / np.mean(model.vsv[i:i+2])
        travel_time[n] += (
            0.5 * model.thickness[ib + 1]
            / ((3 * model.vsv[ib] + model.vsv[ib + 1]) / 4)
        )

        n += 1

    #now add up the time in the extra layers
    if model_params.extra_layers.size > 0:
        vs = model_params.extra_layers[1].copy()
        vs[vs==0] = 1.e100 #effectively remove it from consideration
        travel_time = travel_time + np.sum(model_params.extra_layers[0]/vs)

    return np.concatenate((travel_time, dV))

def predict_headwave(model:define_models.VsvModel, model_params:define_models.ModelParams):
    """created by JSB to extract the velocity at the top of mantle, if there are headwaves used as a constraint
    """
    return model.vsv[model.boundary_inds[model_params.boundaries[0].index('Moho')] + 1]

def _build_model_vector(model:define_models.VsvModel,
                        depth_limits:tuple) -> (np.array):
    """ Make model into column vector [s; t].

    Arguments:
        model:
            - define_models.VsvModel
            - Units:    seismological (km/s, km)
            - Input Vs model

    Returns:
        p:
            - (n_depth points + n_boundary_layers, 1) np.array
            - Units:    seismological, so km/s for velocities (s),
                        km for layer thicknesses (t)
            - Inversion model, p, is made up of velocities defined at various
              depths (s) and thicknesses for the layers above boundary layers
              of interest (t), i.e. controlling the depth of e.g. Moho, LAB
            - All other thicknesses are either fixed, or are dependent only
              on the variation of thicknesses, t
            - Note that the deepest velocity point is fixed too.
            - This model, p, ONLY includes the parameters that we are inverting
              for - it is not a complete description of vs(z)!
    """

    return np.vstack((model.vsv[model.d_inds],
                      model.thickness[list(model.boundary_inds)]))

def _build_inversion_model_from_model_vector(p_in:np.array, #changed to p_in jsb
        model:define_models.VsvModel):
    """ Make column vector, [s; t] into VsvModel format.

    Arguments:
        model:
            - define_models.VsvModel
            - Units:    seismological (km/s, km)
            - Input Vs model
        p_in:
            - (n_depth points + n_boundary_layers, 1) np.array
            - Units:    seismological, so km/s for velocities (s),
                        km for layer thicknesses (t)

    Returns:
        model:
            - define_models.VsvModel
            - Units:    seismological (km/s, km)
            - Vs model with values updated from p_in.

    """

    if model.boundary_inds.size == 0:
        return define_models.VsvModel(
            vsv = np.vstack((p_in.copy(), model.vsv[-1])),
            thickness = model.thickness,
            boundary_inds = model.boundary_inds
        )

    new_thickness = model.thickness.copy()
    dt = p_in[-len(model.boundary_inds):] - model.thickness[model.boundary_inds]
    new_thickness[model.boundary_inds] += dt
    new_thickness[model.boundary_inds + 2] -= dt
    new_vsv = np.vstack((p_in[:-len(model.boundary_inds)].copy(), model.vsv[-1]))
    #jsb thickness/vsv length can change and be more than one off, so fix new_thickness here
    new_thickness = new_thickness[:new_vsv.size]

    return define_models.VsvModel(
        vsv = new_vsv,
        thickness = new_thickness,
        boundary_inds = model.boundary_inds,
        d_inds = model.d_inds,
    )

def _build_data_misfit_vector(data:np.array, prediction:np.array,
        m0:np.array, G:np.array):
    """ Calculate data misfit.

    This is the difference between the observed and calculated phase velocities.
    To allow the inversion to invert for the model directly (as opposed to
    the model perturbation), we add this to G * m0.

    Gm = d              Here, G = d(phase vel)/d(model),
                              m = change in model, m - m0
                              d = change in phase vel (misfit from observations)

    To allow us to use m = new model, such that we can later add linear
    constraints based on model parameters rather than model perturbation
    parameters,
        G (m - m0) = d          (as above)
        Gm - G*m0 = d
        Gm = d + G*m0

    d is still the misfit between observed (data.c) and predicted phase
    velocities (predictions.c), so we find the full RHS of this equation
    as (data.c - predictions.c) + G * m0 (aka Gm0).

    See Russell et al., 2019 (10.1029/2018JB016598), eq. 7 to 8.

    Arguments:
        data:
            - (n_periods, 1) np.array
            - Units:    data.surface_waves['Phase_vel'] is in km/s
            - Observed phase velocity data
            - Extracted from a constraints.Observations object,
              all_data.surface_waves['Phase_vel']
        prediction:
            - (n_periods, ) np.array
            - Units:    km/s
            - Previously calculated phase velocities for the input model.
        m0:
            - (n_model_points, 1) np.array
            - Units:    seismology units, i.e. km/s for velocities
            - Earth model in format for calculating the forward problem, G * m0.
        G:
            - (n_periods, n_model_points) np.array
            - Units:    assumes velocities in km/s
            - Matrix of partial derivatives of phase velocity wrt the
              Earth model (i.e. stacked Frechet kernels converted for use
              with m0).

    Returns:
        data_misfit:
            - (n_periods, 1) np.array
            - Units:    km/s
            - The misfit of the data to the predictions, altered to account
              for the inclusion of G * m0.

    """

    Gm0 = np.matmul(G, m0)
    dc = data - prediction[:, np.newaxis]

    data_misfit = Gm0 + dc

    return data_misfit, dc

def _damped_least_squares(m0, G, d, W, H_mat, h_vec):
    """ Calculate the damped least squares, after Menke (2012).

    Least squares (Gauss-Newton solution):
    m = (G'*G)^-1 * G'* d       (i.e. d = G * m)
        - m:    new model (n_model_params*n_depth_points x 1)
        - G:    partial derivative matrix (n_data_points x n_model_points)
        - d:    data misfit (n_data_points x 1)

        where n_model_points = n_model_params * n_depth_points
            (model parameters are Vsv, Vsh, Vpv, Vph, eta (= F/(A-2L), anellipticity))
p, G, d, W, H_mat, h_vec
    Damped least squares in the Menke (2012; eq. 345) notation:
    m = (G' * We * G  + ε^2 * Wm )^-1  (G' * We * d + ε^2 * Wm * <m>)
        - m:    new model
                    (n_model_points x 1) == (n_model_params*n_depth_points x 1)
        - G:    partial derivatives matrix
                    (n_data_points x n_model_points)
        - We:   data error weighting matrix (our input W)
                    (n_data_points x n_data_points)
        - ε:    damping parameters
        - Wm:   smoothing matrix (in terms of our input: D' * D)
                    (n_model_points x n_model_points)
        - d:    data (e.g. phase velocity at each period)
                    (n_data_points x 1)
        - <m>:  old (a priori) model
                    (n_model_points x 1)
                    Note that in our final inversion, we do not use this
                    explicitly as in the notation below, but is included here
                    to work forwards from the Menke (2012) formulation.
    Where
        - D:    roughness matrix
                    (n_smoothing_equations x n_model_points)

    This is formulated in Menke (2012; eq. 3.46) as
        F * m_est  = f
        F  =  [[  sqrt(We) * G  ]       f   =  [[    sqrt(We) * d    ]
               [      εD        ]]              [       εD<m>        ]]


        i.e. m_est = (F' * F)^-1 * F' * f
                   = [(G' * We * G) + (ε^2 * D' * D)]^-1
                     * [(G' * We * d) + (ε^2 * D' * D * <m>) ]

    Note that here the old model, <m>, is just standing in for the constraints
    on the smoothness, here explicitly picked out as εD * m_est = εD * <m>.
    In this formulation, we can just sweep this smoothing constraint in with
    all of the other a priori constraints.  To damp towards a perfectly smooth
    model (as coded in weights.py), D approximates the second derivative of
    the model with D * m_est and is set equal to zero instead of εD * <m>.
        F  =  [[  sqrt(We) * G  ]       f   =  [[    sqrt(We) * d    ]
               [      εD        ]]              [  [[0],[0],...,[0]] ]]

        i.e. m_est = (F' * F)^-1 * F' * f
                   = [(G' * We * G) + (ε^2 * D' * D)]^-1 * [(G' * We * d)]

    Any other a priori constraints, formulated by Menke (2012; eq. 3.55)
    as linear constraint equations:
        H * m = h

    e.g. to damp to the starting model, H would be a diagonal matrix of ones,
    and h would be the original model.

    These are combined as vertical stacks:
        F * m_est = f
        F  =  [[  sqrt(We) * G  ]       f   =  [[    sqrt(We) * d    ]
               [      εD        ]               [  [[0],[0],...,[0]] ]
               [      εH        ]]              [         h          ]]
               ((n_data_points                  ((n_data_points
                + n_smoothing_equations          + n_smoothing_equations
                + n_a_priori_constraints),       + n_a_priori_constraints),
                n_model_params) np.array         1) np.array
        i.e. m_est = (F' * F)^-1 * F' * f
                   = [(G' * We * G) + (ε^2 * D' * D) + (H' * H)]^-1
                     * [(G' * We * d) + (H' * h)]
    """

    F = np.vstack((np.matmul(np.sqrt(W), G), H_mat))
    f = np.vstack((np.matmul(np.sqrt(W), d), h_vec))

    Finv_denominator = np.matmul(F.T, F)
    # x = np.linalg.lstsq(a, b) solves for x: ax = b, i.e. x = a \ b in MATLAB
    Finv = np.linalg.lstsq(Finv_denominator, F.T, rcond=None)[0]

    new_model = np.matmul(Finv, f)

    #
    # H = [D2; H1; H2; H3; H4; H6; H7; H8; H9]; % where H = D
    # h = [d2; h1; h2; h3; h4; h6; h7; h8; h9]; % h = D*mhat
    # epsilon_Gmd_vec = ones(length(Ddobs),1) * epsilon_Gmd;
    # epsilon_Gmd_vec(Ilongper) = epsilon_Gmd_longper; % assign different damping to long period data
    # F = [epsilon_Gmd_vec.*We.^(1/2)*GG; epsilon_HhD*H];
    # % f = [We.^(1/2)*dobs; epsilon2*h];
    # f = [epsilon_Gmd_vec.*We.^(1/2)*Ddobs; epsilon_HhD*h];
    # [MF, NF] = size(F);
    # Finv = (F'*F+epsilon_0norm*eye(NF,NF))\F'; % least squares
    # mest_all = Finv*f;

    return new_model
