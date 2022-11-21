""" Generate Earth models to work with the inversion and MINEOS.

Classes:
    1. ModelParams
        - Fixed input parameters for the velocity model
        - Fields:
            id                  - Unique name for the run, used for saving
            boundaries          - Tuple, (boundary_names, boundary_widths)
                boundary_names      - Tuple of labels for each of the boundaries
                boundary_widths     - List with (fixed) widths of the boundary layers
            depth_limits        - Top and base of the model that we are solving for
            min_layer_thickness - Minimum thickness of the layers in the model
            vsv_vsh_ratio       - Ratio of Vsv to Vsh
            vpv_vsv_ratio       - Ratio of Vpv to Vsv
            vpv_vph_ratio       - Ratio of Vpv to Vph
            ref_card_csv_name   - Path to a reference full MINEOS model card
    2. VsvModel
        - Shear velocity model with
        - Fields:
            vsv                 - Shear velocities, given at model nodes
            thickness           - Layer thicknesses between the model nodes
            boundary_inds       - Indices of boundaries of special interest, e.g. Moho, LAB
            d_inds              - Indices of velocity points that are being solved for
    3. EarthLayerIndices
        - Indices for different layers (e.g. crust, lithosphere, asthenosphere) for which you might want to weight the regularisation differently (see weights.py)
        - Fields:
            sediment            - Depth indices for sedimentary layer.
            crust               - Depth indices for crust.
            lithospheric_mantle - Depth indices for lithospheric mantle.
            asthenosphere       - Depth indices for asthenosphere.
            discontinuities     - Depth indices of discontinuties
            depth               - Depth vector for the whole model space.
            layer_names         - All layer names in order of increasing depth.
Functions:
    1. setup_starting_model(model_params: ModelParams, location: tuple) -> VsvModel:
        - Create VsvModel from ModelParams and a location (used for initial Vsv values)
    3. _list_to_col(l: list) -> np.array:
        - Convert a list into a column vector
    2. _fill_in_base_of_model(thick:list, vs:list, model_params:ModelParams)
        - Append values to lists of Vsv and layer thickness to fit depth limits
    3. _add_BLs_to_starting_model(thick:list, model_params:ModelParams) -> list:
        - Return boundary layer indices for starting model, and update input list of thickness
    4. _find_depth_indices(thick:list, depth_limits:tuple) -> list:
        - Return list of indices in input layer thickness within the inversion depth limits
    5. _return_evenly_spaced_model(model: VsvModel, min_layer_thickness:float
                                   ) -> (np.array, np.array, np.array):
        - Refactor VsvModel so that layers are a more uniform thickness
    6. _mean_val_in_interval(v:list, thick:list , d1:float, d2:float) -> float:
        - Find mean value of v within some depth range d1 to d2
    7. _add_noise_to_starting_model(model:VsvModel, depth_limits:tuple) -> VsvModel:
        - Add random noise to a VsvModel
    9. _add_random_noise(a:np.array, sc:float, pdf='normal') -> np.array:
        - Add random noise to an array of a given scale (i.e. standard deviation for normal pdf)
    10. convert_vsv_model_to_mineos_model(vsv_model:VsvModel, model_params:ModelParams,
                                                **kwargs) -> pd.DataFrame:
        - Convert VsvModel to have all values necessary for MINEOS, and write to disk as csv
    11. _write_mineos_card(mineos_card_model:pd.DataFrame, name:str):
        - Add header information and write MINEOS compatible model as a .card file
    12. _set_earth_layer_indices(model_params:ModelParams, model:VsvModel, **kwargs) -> EarthLayerIndices:
        - Find the indices of various geological layers
    13. save_model(model:VsvModel, fname:str):
        - Save VsvModel to file
    14. read_model(fname:str):
        - Read VsvModel from file
    15. load_all_models(z:np.array, lats:np.array, lons:np.array, t_LAB:int, lab:str=''
                        ) -> (np.array, np.array, np.array):
        - Load a bunch of models saved with predictable file names for plotting purposes

"""

#import collections
import typing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
from scipy import stats
from util import constraints
import csv

# =============================================================================
# Set up classes for commonly used variables
# =============================================================================

class ModelParams(typing.NamedTuple):
    """ Fixed input parameters for a velocity model.

    This records values that will remain fixed for a given inversion.  This includes an id, various fixed values around the boundaries in velocity (e.g. Moho, LAB), and various parameters needed to convert between the Vsv model of the crust and uppermost mantle calculated by the inversion and the earth model used by MINEOS.

    MINEOS is used to calculate the predicted phase velocities and associated kernels. This needs a more complete earth model (i.e. surface to the core; Vsv, Vsh, Vpv, Vph, eta).  Here, we set the depth limits of the model to invert, a reference earth model to use outside of that range, and a linear scaling from Vsv to Vsh, from Vsv to Vpv, and from Vpv to Vph.  These all have default values if the user has no preference.

    Fields:
        id:
            - str
            - Unique name of model for saving things.
        boundaries:
            - tuple - (tuple, tuple)
            - Items in tuple (each of these tuples is of length n_boundaries):
                boundary names:
                    - tuple - (str, str, ...)
                    - Units: none
                    - Labels for each of the boundaries
                    - Default = ('Moho', 'LAB')
                boundary widths:
                    - tuple - (float, float, ...)
                    - Units: kilometres
                    - (Fixed) width of the boundary layer.  This is fixed for a given inversion (though we will allow the depth of the boundary layer as a whole to change).
                    - Default = [3., 10.]
        depth_limits:
            - tuple - (float, float)
            - Units:    km
            - Top and base of the model that we are inverting for.
            - Outside of this range, the model is fixed to our starting MINEOS model card (which extends throughout the whole Earth).
            - Default value = (0, 300)
        min_layer_thickness:
            - float
            - Units:    km
            - Default value: 6
            - Minimum thickness of the layer, should cover several (three) knots in the MINEOS model card.
        vsv_vsh_ratio:
            - float
            - Units:    dimensionless
            - Ratio of Vsv to Vsh, default value = 1 (i.e. radial isotropy)
        vpv_vsv_ratio:
            - float
            - Units:    dimensionless
            - Ratio of Vpv to Vsv, default value = 1.75
        vpv_vph_ratio:
            - float
            - Units:    dimensionless
            - Ratio of Vpv to Vph, default value = 1 (i.e. radial isotropy)
        eta:
            - float:
            - Units:    dimensionless
            - Shape factor, a measure of ellipticity of seismic anisotropy, default value = 1. (i.e. radial isotropy)
        ref_card_csv_name:
            - str
            - Default value: 'data/earth_model/prem.csv'
            - Path to a .csv file containing the information for the reference full MINEOS model card that we'll be altering.
            - This could be some reference Earth model (e.g. PREM), or some more specific local model.
            - Note that this csv file should be in SI units (m, kg.m^-3, etc)

    Fields added by JSB:

        perturbation_threshold: float
            - Terminate the inversion if the biggest change to the model was under this value in percent
            - Units are percentage relative to previous model
        randomize_model: bool = True
        randomize_data: bool = True
            - When initializing, do you add noise to the model and the data, yes or not?
            - Turn off if you want the optimial answer, true if you are bootstrapping for uncertainities.

        - These three are the damping parameters, previously hardwired.
        - Order is (sediment, crust, lithosphere, asthenosphere, boundary inds)
        - By setting different values for different layers you can have different damping schemes for different parts of the model.
        - Also keep each zero on boundary inds.
        roughness: tuple
            - Limit the second derivative
        to_m0: tuple
            - Limit the step size. Note this is a creeping scheme - over many iterations you can go far from the starting model.
        to_0_grad: tuple
            - Try to keep the gradient near zero
        """

    id: str
    boundaries: tuple          = (('Moho', 'LAB'), [1, 10], [0.25, 0.5])#name, width, width of smoothing kernel (jsb)
    #boundaries: tuple          = (('Sed', 'Moho', 'LAB'), [0, 1, 10], [0.01, 0.25, 0.5])#name, width, width of smoothing kernel (jsb)
    #boundaries: tuple          = ( ('Moho',), [1], [0.25])#name, width, width of smoothing kernel (jsb)
    depth_limits: tuple        = (0, 400)
    min_layer_thickness: float = 6.
    min_num_layers: int        = 5
    vsv_vsh_ratio: float       = 1.
    vpv_vsv_ratio: float       = 1.76
    vpv_vph_ratio: float       = 1.
    eta: float                 = 1.
    ref_card_csv_name: str     = 'data/earth_models/prem.csv'
    method: str                = 'mineos' #which surface wave code do you want to use?
    q_model: str               = './data/earth_models/qmod3' #name of the q model used for corrections with mineos

    #added by jsb
    perturbation_threshold:float = 0.1
    head_wave: bool              = True
    head_wave_smoothing          = 0.5 #in degrees
    breadth:float                = 1 #0 is values in boundaries, otherwise loadedvalue*breath
    randomize_model: bool        = False
    randomize_data: bool         = False
    mineos_intervals:tuple       = (0.5, 3, 0.2) #in km/s (when a new layer is made), km (when a new layer is made), km (when a layer is thin enough for discon )
    #Default values based on Emily's first version of the code + m0 damping and linear lithosphere
    roughness: tuple = (2,2,1000,2,0)
    to_m0:tuple      = (0,0,0,   0,0)
    to_0_grad:tuple  = (0,0,0,   0,0)

    min_vs:float = 0.5
    extra_layers: np.array = np.array([])

class VsvModel(typing.NamedTuple):
    """ Vsv model with some additional information for the inversion.

    The Vsv model is made up of a vector of Vsv values (VsvModel.vsv; referred to as 's' in the documentation) at certain depths (defined as the sum of overlying layer thicknesses, VsvModel.thickness).  Boundary depths (e.g. Moho, LAB) are allowed to vary by solving for the thickness of the directly overlying layers (the indices of these layers are given in .boundary_inds; the mutable widths of these layers are referred to as 't' in the documentation).  All other values in VsvModel.thickness are not inverted for.  However, these values will be updated throughout the inversion as the model is adjusted to keep all layers approximately the same thickness via _return_evenly_spaced_model().

    Thus, the actual model that goes into the least squares inversion is
        m = [s; t] = np.vstack((VsvModel.vsv,
                                VsvModel.thickness[VsvModel.boundary_inds -1]))

    Fields:
        vsv:
            - (n_layers, 1) np.array
            - Units:    km/s
            - Shear velocity at top of layer in the model.
            - Velocities are assumed to be piecewise linear.
        thickness:
            - (n_layers, 1) np.array
            - Units:    km
            - Thickness of layer above defined vsv point, such that depth of .vsv[i] point is at sum(thickness[:i+1]) km.
                Note that this means the sum of thicknesses up to and including the ith point.
            - That is, as the first .vsv point is defined at the surface, the first value of .thickness will be 0 always.
        boundary_inds:
            - (n_boundaries, ) np.array of integers
            - Units:    n/a
            - Indices in .vsv and .thickness identifying the boundaries of special interest, e.g. Moho, LAB.  For these boundaries, we will want to specifically prescribe the width of the layer (originally set as ModelParams.boundary_widths), and to invert for the thickness of the overlying layer (i.e. the depth to the top of this boundary).
            - That is, VsvModel.vsv[boundary_inds[i]] is the velocity at the top of the boundary and VsvModel.thickness[boundary_inds[i]] is the thickness of the layer above it, defining boundary depth.
            - VsvModel.vsv[boundary_inds[i + 1]] is the velocity at the bottom of the boundary and VsvModel.thickness[boundary_inds[i + 1]] is the thickness of the boundary layer itself, fixed for an inversion run in ModelParams.boundaries.
        d_inds:
            - (n_velocities_inverted_for, ) np.array of indices
            - Units:    n/a
            - Indices in VsvModel.vsv identifying the depths within ModelParams.depth_limits.  Note that if the upper depth limit is 0 (i.e. inversion starts at the free surface), this will be all [True, True, True, ...., True, False] - that is, everything but the last velocity value is inverted for.  The last velocity value is never inverted for as this is pinned to the value taken from ModelParams.ref_card_csv_name.

    """
    vsv: np.array
    thickness: np.array
    boundary_inds: np.array
    d_inds: np.array

class EarthLayerIndices(typing.NamedTuple):
    """ Indices for geologically significant layers used for different levels of regularisation.

    Fields:
        sediment:
            - (n_depth_points_in_this_layer, ) np.array
            - Depth indices for sedimentary layer.
        crust:
            - (n_depth_points_in_this_layer, ) np.array
            - Depth indices for crust.
        lithospheric_mantle:
            - (n_depth_points_in_this_layer, ) np.array
            - Depth indices for lithospheric mantle.
        asthenosphere:
            - (n_depth_points_in_this_layer, ) np.array
            - Depth indices for asthenosphere.
        discontinuities:
            - (n_different_layers + 1, ) np.array
            - Depth indices of model discontinuties, including the surface
              and the base of the model.
            - i.e. [0, mid-crustal boundary, Moho, LAB, base of model]
        depth:
            - (n_depth_points, ) np.array
            - Units:    kilometres
            - Depth vector for the whole model space.
        layer_names:
            - list
            - All layer names in order of increasing depth.

    """

    sediment: np.array
    crust: np.array
    lithospheric_mantle: np.array
    asthenosphere: np.array
    boundary_layers: np.array
    depth: np.array
    layer_names: list = ['sediment', 'crust',
                         'lithospheric_mantle', 'asthenosphere',
                         'boundary_layers']

# =============================================================================
#       Set up the models for various stages of the calculation
# =============================================================================

def setup_starting_model(model_params: ModelParams, location: tuple, breadth:np.array) -> VsvModel:
    """ Convert from ModelParams to VsvModel.

    ModelParams is the bare bones of fixed parameters for the starting model. Here we create a VsvModel object using ModelParams. Note that this is in a different format to the model that we actually want to invert, m = np.vstack(
                    (VsvModel.vsv[VsvModel.d_inds],
                     VsvModel.thickness[VsvModel.boundary_inds)
                     )

    We calculate appropriate layer thicknesses such that the inversion will have all the required flexibility when inverting for the depth of the boundaries of interest.  Starting model Vs is kind of just randomly bodged here (taken from a reference model at your input location), but that is not important as the inversion is not regularised by tending towards the starting model.  That is, the starting model has no significant impact on the final model.

    Arguments:
        model_params:
            - ModelParams
            - Units:    seismological, i.e. km, km/s
            - Starting model parameters, defined elsewhere
        location:
            - tuple, length 2 (latitude, longitude)
            - Units:    degrees latitude (N), degrees longitude (E)
            - Location for constraints - used here to put in appropriate crustal structure, Moho, and LAB

    Returns:
        vsv_model:
            - VsvModel
            - Units:    seismological, i.e. km, km/s
            - Model primed for use in the inversion
    """
    # Set up directory to save to
    if not os.path.exists('output'):
        os.mkdir('output')
    if not os.path.exists('output/' + model_params.id):
        os.mkdir('output/' + model_params.id)
    #else:
    #    print('This model ID has already been used!') jsb

    ref_model =  pd.read_csv(model_params.ref_card_csv_name)

    vs    = np.flip(ref_model["vsv"].to_numpy())/1000
    depth = ref_model["radius"].to_numpy()/1000
    depth -= depth[-1]
    depth = np.flip(depth)
    depth *= -1

    vs    = vs[ depth < model_params.depth_limits[1] ]
    depth = depth[ depth < model_params.depth_limits[1] ]
    thick = np.hstack((0,np.diff(depth)))
    #thick[ thick < 0.1 ] = 0.1

    vs    = np.ndarray.tolist(vs)
    thick = np.ndarray.tolist(thick)

    # Fill in to the base of the model
    _fill_in_base_of_model(thick, vs, model_params)

    # Add on the boundary layers at arbitrary depths (will be fixed by inversion)
    bi = _add_BLs_to_starting_model(thick, model_params, breadth)

    # Fix the model spacing
    thickness, vsv, boundary_inds = _return_evenly_spaced_model(
        VsvModel(_list_to_col(vs), _list_to_col(thick), np.array(bi), []),
        model_params.min_layer_thickness,
    )
    depth_inds = _find_depth_indices(thickness, model_params.depth_limits)

    # Add random noise
    model = _add_noise_to_starting_model(
        VsvModel(vsv, thickness, boundary_inds, depth_inds), model_params.depth_limits
    , model_params.randomize_model) #jsb added randomize

    return model

def setup_starting_model_NoMelt(model_params: ModelParams, breadth:np.array) -> VsvModel:
    """ Convert from ModelParams to VsvModel.

    ModelParams is the bare bones of fixed parameters for the starting model. Here we create a VsvModel object using ModelParams. Note that this is in a different format to the model that we actually want to invert, m = np.vstack(
                    (VsvModel.vsv[VsvModel.d_inds],
                     VsvModel.thickness[VsvModel.boundary_inds)
                     )

    We calculate appropriate layer thicknesses such that the inversion will have all the required flexibility when inverting for the depth of the boundaries of interest.  Starting model Vs is kind of just randomly bodged here (taken from a reference model at your input location), but that is not important as the inversion is not regularised by tending towards the starting model.  That is, the starting model has no significant impact on the final model.

    Arguments:
        model_params:
            - ModelParams
            - Units:    seismological, i.e. km, km/s
            - Starting model parameters, defined elsewhere
        location:
            - tuple, length 2 (latitude, longitude)
            - Units:    degrees latitude (N), degrees longitude (E)
            - Location for constraints - used here to put in appropriate crustal structure, Moho, and LAB

    Returns:
        vsv_model:
            - VsvModel
            - Units:    seismological, i.e. km, km/s
            - Model primed for use in the inversion

    JSB copied from setup_starting_model for NoMelt
    """

    NoMelt          = pd.read_csv('./NoMelt_JRussel/starting_model_NoMelt2.csv')
    NoMelt.columns  = ['thickness', 'Vs']

    thick = NoMelt['thickness'].tolist()
    vs = NoMelt['Vs'].tolist()

    # Fill in to the base of the model
    _fill_in_base_of_model(thick, vs, model_params)

    # Add on the boundary layers at arbitrary depths (will be fixed by inversion)
    #bi = _add_BLs_to_starting_model(thick, model_params, breadth)

    bi = [2, 17]

    # Fix the model spacing
    thickness, vsv, boundary_inds = _return_evenly_spaced_model(
        VsvModel(_list_to_col(vs), _list_to_col(thick), np.array(bi), []),
        model_params.min_layer_thickness,
    )

    #boundary_inds = np.array([11, 22])
    #thickness = np.array(thick, ndmin=2).T
    #vsv       = np.array(vs, ndmin=2).T

    if np.sum(thickness) < model_params.depth_limits[1]:
        thickness[-1] = model_params.depth_limits[1] - np.sum(thickness)

    depth_inds = _find_depth_indices(thickness, model_params.depth_limits)

    # Add random noise
    model = _add_noise_to_starting_model(
        VsvModel(vsv, thickness, boundary_inds, depth_inds), model_params.depth_limits
    , model_params.randomize_model) #jsb added randomize

    #model.vsv[model.vsv<model_params.min_vs] = model_params.min_vs

    return model

def _list_to_col(l):
    """ Convert list into column vector (np array of size (n_elements, 1))

    Arguments:
        l:
            - list of values compatible with numpy array
    Returns:
        Column vector of same values:
            - (n_elements, 1) np.array
    """
    return np.reshape(np.array(l), (np.size(np.array(l)), 1))

def _fill_in_base_of_model(thick:list, vs:list, model_params:ModelParams):
    """ Ensure model has the correct depth limits.

    Given a list of Vsv values (vs) and layer thicknesses (thick), append any necessary values to fill in any necessary values to reach the depth limit of the inversion.  These values are taken from the reference card listed in model_params, with the deepest value of Vs always pinned to the relevant value in the reference card.

    Arguments:
        thick:
            - list of floats
            - Units:    kilometres
            - Layer thicknesses, like VsvModel.thickness
            - This list is changed inside this function!
        vs:
            - list of floats
            - Units:    km/s
            - Vsv at nodes, like VsvModel.vsv
            - This list is changed inside this function!
        model_params:
            - ModelParams object
            - Fixed parameters for velocity model
    Returns:
        thick and vs lists are changed in this function
    """

    # Load reference Vs model
    ref_model =  pd.read_csv(model_params.ref_card_csv_name)
    ref_depth = (ref_model['radius'].iloc[-1] - ref_model['radius']) * 1e-3
    ref_vs = ref_model['vsv'] * 1e-3
    # Remove discontinuities and make depth increasing for purposes of interp
    ref_depth[np.append(np.diff(ref_depth), -1) == 0] += 0.01
    ref_depth = ref_depth.iloc[::-1].values
    ref_vs = ref_vs.iloc[::-1].values

    # If model already exceeds the depth limit, crop thick and vs and return
    if sum(thick) >= model_params.depth_limits[1]:
        i = np.argmax(np.cumsum(thick) >= model_params.depth_limits[1])
        thick[i] = model_params.depth_limits[1] - sum(thick[:i])
        vs[i] = np.interp(model_params.depth_limits[1], ref_depth, ref_vs)
        vs = vs[:i + 1]
        thick = thick[:i + 1]
        return

    remaining_depth = model_params.depth_limits[1] - sum(thick)
    n_layers = int(remaining_depth // model_params.min_layer_thickness)

    if n_layers > 0:
        thickness_to_end = [remaining_depth / n_layers] * n_layers
    else:
        thickness_to_end = [0.] #needs to be a list

    vs += list(np.interp(sum(thick) + np.cumsum(thickness_to_end), ref_depth, ref_vs))

    thick += thickness_to_end

    return

def _add_BLs_to_starting_model(thick:list, model_params:ModelParams, bwidths:np.array) -> list:
    """ Return boundary layer indices, evenly distributed in starting model depth.

    As the inversion is not regularised with respect to the starting model, the boundaries can be seeded anywhere in the starting model without affecting the final model.  Here, we evenly distribute them in depth.  The thicknesses of the relevant layers are updated to match the fixed values given in model_params.

    Arguments:
        thick:
            - list of floats
            - Units:    kilometres
            - Layer thicknesses, like VsvModel.thickness
            - This list is changed inside this function!
        model_params:
            - ModelParams object
            - Fixed parameters for velocity model
    Returns:
        boundary_inds:
            - list of inds
            - Units:    none
            - Indices of boundary layers, like VsvModel.boundary_inds
        thick is updated in this function

    """
    #_, bwidths = model_params.boundaries
    n_bounds = bwidths.size
    boundary_inds = []

    # For now, distribute the boundary layers evenly over the depth range
    # (and space away from surface and base of model)
    spacing = np.diff(model_params.depth_limits) / (n_bounds + 2)

    i = 0
    for i_b in range(n_bounds):

        while sum(thick[:i + 1]) < spacing * (i_b + 1):
            i += 1
        boundary_inds += [i]
        thick[i+1] = bwidths[i_b]

    thick[-1] -= sum(thick) - model_params.depth_limits[1]
    #thick = (thick/sum(thick))*model_params.depth_limits[1] #jsb make negative element

    return boundary_inds

def _find_depth_indices(thick:np.array, depth_limits:tuple) -> list:
    """ Return a list of the indices of thickness within the depth limits

    Arguments:
        thick:
            - (n_layers, 1) np.array
            - Units:    kilometres
            - Layer thicknesses, as VsvModel.thickness
        depth_limits:
            - tuple (float, float)
            - Units:    kilometres
            - Depth limits for inversion, as ModelParams.depth_limits field
    Returns:
        List of indices in the input list, thick, that are within the input depth limits
    """

    dd = np.round(np.cumsum(thick), 3) # round to the nearest metre
    # argmax gives earliest index with maximum value in array (here 0s & 1s only)

    return list(range(np.argmax(dd >= depth_limits[0]),
                      np.argmax(dd >= depth_limits[1])))

def _return_evenly_spaced_model(model: VsvModel, min_layer_thickness:float, min_num_layers:int=5) -> (np.array, np.array, np.array):
    """ Refactor the Vsv model so layer thicknesses are more even.

    As the inversion updates some layer thicknesses, this will make the model layer thicknesses very uneven unless the starting model is approximately correct.  Indeed, the inversion can lead to negative layer thicknesses if a boundary layer is too deep.  As we do not want dependence on the starting model, we need to respace the model.

    It is very important to keep the boundary layer thicknesses the same!

    Arguments:
        model:
            - VsvModel object
            - Units:    seismological - km, km/s etc
            - Velocity model
            - All fields except d_inds are used here
        min_layer_thickness:
            - float
            - Units:    kilometres
            - Minimum allowable layer thickness in VsvModel.thickness, as ModelParams.minimum_layer_thickness
    Returns:
        new_thick:
            - (n_layers, 1) np.array
            - Units:    kilometres
            - Layer thicknesses, as VsvModel.thickness
            - Note that n_layers in outputs may be different to the input model!
        new_v:
            - (n_layers, 1) np.array
            - Units:    seismological - km, km/s
            - Node velocities, as VsvModel.vsv
        new_bi:
            - (n_boundaries, ) np.array of integers
            - Units:    none
            - Boundary indices, as VsvModel.boundary_inds


    """
    thick         = model.thickness.flatten().tolist()
    vs            = model.vsv.flatten().tolist()
    boundary_inds = model.boundary_inds.flatten().tolist()

    # Set uppermost value of thick
    new_thick = [thick[0]]
    bi        = [-1] + boundary_inds
    new_bi    = []

    # Loop through all BLs to get velocities above and within them
    for ib in range(len(bi[:-1])):
        #inter_boundary_depth = (sum(thick[:bi[ib + 1] + 1])
        #                        - sum(thick[:bi[ib] + 2]))

        inter_boundary_depth = max((sum(thick[:bi[ib + 1] + 1])
                                - sum(thick[:bi[ib] + 2])),1)#cannot be thinner than 1km

        n_layers = max(int(inter_boundary_depth // min_layer_thickness), min_num_layers)
        #if ib < 2: #JSB Can cause serious issues if the mineos depth vector becomes coarser than the model
        #    n_layers *= 3 # Have denser spacing in the crust
        layer_t = inter_boundary_depth / n_layers

        # set uppermost value of Vs
        if ib == 0:
            new_v = [_mean_val_in_interval(vs, thick, sum(new_thick), sum(new_thick) + layer_t / 2)]

        for il in range(n_layers - 1):
            new_thick += [layer_t]
            new_v += [_mean_val_in_interval(vs, thick, sum(new_thick) - layer_t / 2,
                                         sum(new_thick) + layer_t / 2)]
        new_bi += [len(new_thick)]
        new_thick += [layer_t, thick[bi[ib + 1] + 1]]
        new_v += vs[bi[ib + 1]:bi[ib + 1] + 2]

    # For the deepest BL to the base of the model
    inter_boundary_depth = sum(thick) - sum(thick[:boundary_inds[-1] + 2])
    # need at least one layer
    n_layers = max(int(inter_boundary_depth // min_layer_thickness), 1)
    layer_t = inter_boundary_depth / n_layers
    for il in range(n_layers - 1):
        new_thick += [layer_t]
        new_v += [_mean_val_in_interval(vs, thick, sum(new_thick) - layer_t / 2,
                                     sum(new_thick) + layer_t / 2)]
    new_thick += [layer_t]
    new_v += [vs[-1]]
    return _list_to_col(new_thick), _list_to_col(new_v), np.array(new_bi)


def _mean_val_in_interval(v:list, thick:list , d1:float, d2:float) -> float:
    """ Calculate the mean value of v in some depth interval.

    Arguments:
        - v:
            - list of floats
            - Units:    unspecified
            - Some list of values defined at nodes (e.g. list of VsvModel.vsv)
        - thick:
            - list of floats
            - Units:    same as d1, d2; e.g. kilometres
            - Layer thicknesses, like VsvModel.thickness
        - d1:
            - float
            - Units:    same as d2, thick
            - Minimum depth of interval (range is inclusive at minimum)
        - d2:
            - float
            - Units:    same as thick, d1
            - Maximum depth of interval (range is exclusive at maximum)


    Returns:
        Mean value within the given range d1-d2:
            - float
            - Units:    same as v

    """
    interp_d  = np.arange(thick[0], sum(thick), 0.1)
    interp_vs = np.interp(interp_d, np.cumsum(thick), v)

    return np.mean(interp_vs[np.logical_and(
                d1 <= interp_d, interp_d < d2
            )])

def _add_noise_to_starting_model(model:VsvModel, depth_limits:tuple, randomize:bool) -> VsvModel:
    """ Add random noise to the starting model

    To investigate dependence on starting model (which is negligible), we add random noise to the starting model at the beginning of the inversion.  We perturb both Vsv and layer thickness.  We maintain layer thickness of the boundary layers, the thickness of the surface layer (i.e. 0), and the total thickness of the velocity model to maintain the depth limits.

    Arguments:
        model:
            - VsvModel
            - Units:    seismological, km, km/s
            - Velocity model
        depth_limits:
            - tuple, (float, float)
            - Units:    kilometres
            - Depth limits for inversion, i.e. only perturb values within these limits
    Returns:
        new model
            - VsvModel
            - Units:    seismological - km, km/s
            - Input velocity model with some perturbations to vsv and thickness
    """
    vs = model.vsv
    thick = model.thickness
    bi = model.boundary_inds
    d_inds = model.d_inds

    # Perturb all Vs that we are inverting forß

    if randomize:
        vs[d_inds] = _add_random_noise(vs[d_inds], 0.2)

        # For thickness, as layer thickness can be very different, scale
        # perturbations by thickness of layer
        for i in range(len(thick) - 1):
            if i - 1 not in bi and i - 1 in d_inds:
                thick[i] = _add_random_noise(np.array(thick[i]), np.array(thick[i]) / 10)

    #thick[-1] = depth_limits[1] - sum(thick[:-1]) jsb editted out
    #jsb - it is possible for the z of the next to last index to exceed depthlimit[1]
    #instead of clamping the last node to the last depth, non-dimensionalize and then
    # re-dimensionalize to the right depth
    depth = np.sum(thick)
    thick = thick*(depth_limits[1]/depth)

    return VsvModel(vs, thick, bi, _find_depth_indices(thick, depth_limits))

def _add_random_noise(a:np.array, sc:float, pdf='normal') -> np.array:
    """ Add random noise to an array of mean 0, scaled by sc.

    Arguments:
        a:
            - np.array
            - Units:    should be the same as value for sc
        sc:
            - float
            - Units:    should be the same as values in a
            - Size of noise to use
                e.g. standard deviation of normal distribution
                e.g. maximum value for a uniform distribution
        pdf:
            - str
            - Label for type of noise distribution - normal or uniform
            - Default value: 'normal'
    Returns:
        np.array of same shape as a with random perturbation of each element
            - np.array
    """
    if pdf == 'normal':
        return a + np.random.normal(loc=0, scale=sc, size=a.shape)
    if pdf == 'uniform':
        return a + np.random.uniform(low=-sc, high=sc, size=a.shape)

def convert_vsv_model_to_mineos_model(vsv_model:VsvModel, model_params:ModelParams,
                                            **kwargs) -> pd.DataFrame:
    """ Generate model that is used for all the MINEOS interfacing.

    MINEOS requires radius, rho, vpv, vsv, vph, vsh, bulk and shear Q, and eta, where eta is the shape factor and is 1 always for isotropic materials. Rows are ordered by increasing radius.  There should be some reference MINEOS card that can be loaded in and have this pasted on the bottom for using with MINEOS, as MINEOS requires a card that goes all the way to the centre of the Earth.

    Arguments:
        - vsv_model:
            - VsvModel
            - Units:    seismological
        - model_params:
            - ModelParams
            - Units:    seismological
        - kwargs
            - key word argument of format:
                - Moho = some_float
            - Units:    km
            - Depth of the Moho - needed to define density structure
    Returns:
        - mineos_model:
            - pd.DataFrame
            - Units:    SI - metres, m/s, etc
            - Fields: radius, rho [density], vpv, vsv, q_kappa [bulk attenuation], q_mu [shear attenuation], vph, vsh, eta [shape factor]
            - This is also written to a csv file
              - output/[model_params.id]/[model_params.id].csv

    """
    if not os.path.exists('output'):
        os.mkdir('output')
    if not os.path.exists('output/' + model_params.id):
        os.mkdir('output/' + model_params.id)
        #print("This test ID hasn't been used before!")

    # Load PREM (http://ds.iris.edu/ds/products/emc-prem/)
    # Slightly edited to remove the water layer and give the model point
    # at 24 km depth lower crustal parameter values.
    ref_model =  pd.read_csv(model_params.ref_card_csv_name)

    radius_Earth = ref_model['radius'].iloc[-1] * 1e-3
    radius_model_top = radius_Earth - model_params.depth_limits[0]
    radius_model_base = radius_Earth - model_params.depth_limits[1]

    #I want to change these layers
    depth = np.array([0])

    #start at the top with Vs0, make a new layer when ΔVs > 0.1 jsb
    ddz = 0.01#depth increment to search by
    Vs0 = vsv_model.vsv[0]

    """
    while np.sum(depth) < model_params.depth_limits[1]:
        #find the next point to which the velocity reaches Vs0+0.1 by stepping in tiny increments

        #initialize the search
        dz  = 0
        ΔVs = 0

        ct = np.interp(np.sum(depth),
                        np.cumsum(vsv_model.thickness)[1:],
                        vsv_model.thickness[1:,0])

        while (ΔVs < model_params.mineos_intervals[0]) and \
            (dz<model_params.mineos_intervals[1]) and dz < ct/3:
            #layer at high gradient, layer at threshold, layer at model thickness jsb
            dz += ddz
            ΔVs = abs(Vs0 - np.interp(np.sum(depth) + dz,
                            np.cumsum(vsv_model.thickness),
                            vsv_model.vsv.flatten()))

        depth = np.append(depth, dz)

        if dz < model_params.mineos_intervals[2]:
            depth = np.append(depth, 0)

        Vs0   = np.interp(np.sum(depth),
                        np.cumsum(vsv_model.thickness),
                        vsv_model.vsv.flatten())

    depth     = np.flipud(np.cumsum(depth))
    depth[0]  = model_params.depth_limits[1]#it will have overshot
    radius    = (radius_Earth - depth)*1e3
    """

    #old block that makes a card model every 2 km
    depth  = np.cumsum(depth)
    step   = model_params.min_layer_thickness / 3
    radius = np.arange(radius_model_base, radius_model_top, step)
    radius = np.append(radius, radius_model_top)
    depth  = (radius_Earth - radius) # still in km at this point
    radius *= 1e3 # convert to SI

    vsv = np.interp(depth,
                    np.cumsum(vsv_model.thickness),
                    vsv_model.vsv.flatten()) * 1e3 # convert to SI

    discon_inds = np.hstack( (np.diff(depth), 1))

    for k in range(1,len(discon_inds),1):
        if discon_inds[k]==0:
            vsv[k+1]=vsv[k+2]

    rho = np.interp(radius, ref_model['radius'], ref_model['rho'])

    #now check if there are "extra layers" to attached on to the model
    #will be given as thickness and vsv
    if model_params.extra_layers.size != 0:
        #convert thickness to radius
        radius = np.hstack( (radius, np.cumsum(model_params.extra_layers[0]) + max(radius)))
        vsv    = np.hstack( (vsv, model_params.extra_layers[1]))
        rho    = np.hstack( (rho, model_params.extra_layers[2]))

    vsh = vsv / model_params.vsv_vsh_ratio
    vpv = vsv * model_params.vpv_vsv_ratio
    vph = vpv / model_params.vpv_vph_ratio
    eta = np.ones(vsv.shape) * model_params.eta
    q_mu = np.interp(radius, ref_model['radius'], ref_model['q_mu'])
    q_kappa = np.interp(radius, ref_model['radius'], ref_model['q_kappa'])

    vpv[vpv < 1500] = 1500
    vph[vph < 1500] = 1500

    bnames, _, _ = model_params.boundaries
    if 'Moho' in bnames:
        Moho_ind = (vsv_model.boundary_inds[bnames.index('Moho')])
        Moho_depth = np.sum(vsv_model.thickness[:Moho_ind + 1])
    else:
        try:
            Moho_depth = kwargs['Moho']
        except:
            print('Moho depth never specified! Guessing 35 km.')
            Moho_depth = 35
    #rho[(radius >= (6371000 - Moho_depth)) & (2900 < rho)] = 2900
    #rescale radius, jsb
    #radius = 6371000.*radius/max(radius)
    radius = radius + 6371000 - max(radius)

    # Now paste the models together, with 100 km of smoothing between
    new_model = pd.DataFrame({
        'radius': radius,
        'rho': rho,
        'vpv': vpv,
        'vsv': vsv,
        'q_kappa': q_kappa,
        'q_mu': q_mu,
        'vph': vph,
        'vsh': vsh,
        'eta': eta,
    })

    smoothed_below = smooth_to_ref_model_below(ref_model, new_model)
    smoothed_above = smooth_to_ref_model_above(ref_model, new_model)

    mineos_card_model = pd.concat([smoothed_below, new_model,
                                   smoothed_above]).reset_index(drop=True)
    mineos_card_model.to_csv('output/{0}/{0}.csv'.format(model_params.id),
                             index=False)

    _write_mineos_card(mineos_card_model, model_params.id)

    return mineos_card_model

def convert_vsv_model_to_disba_model(vsv_model:VsvModel, model_params:ModelParams,
                                            **kwargs) -> pd.DataFrame:
    """ Generate model that is used for all the disba runs. Copied from the mineos version by JSB
    """

    if not os.path.exists('output'):
        os.mkdir('output')
    if not os.path.exists('output/' + model_params.id):
        os.mkdir('output/' + model_params.id)
        #print("This test ID hasn't been used before!")

    # Load PREM (http://ds.iris.edu/ds/products/emc-prem/)
    # Slightly edited to remove the water layer and give the model point
    # at 24 km depth lower crustal parameter values.
    ref_model =  pd.read_csv(model_params.ref_card_csv_name)

    radius_Earth = ref_model['radius'].iloc[-1] * 1e-3
    radius_model_top = radius_Earth - model_params.depth_limits[0]
    radius_model_base = radius_Earth - model_params.depth_limits[1]

    #I want to change these layers
    depth = np.array([0])

    """
    #start at the top with Vs0, make a new layer when ΔVs > 0.1 jsb
    ddz = 0.01#depth increment to search by
    Vs0 = vsv_model.vsv[0]
    while np.sum(depth) < model_params.depth_limits[1]:
        #find the next point to which the velocity reaches Vs0+0.1 by stepping in tiny increments

        #initialize the search
        dz  = 0
        ΔVs = 0

        ct = np.interp(np.sum(depth),
                        np.cumsum(vsv_model.thickness)[1:],
                        vsv_model.thickness[1:,0])

        while (ΔVs < model_params.mineos_intervals[0]) and \
            (dz<model_params.mineos_intervals[0]) and dz < ct/3:
            #layer at high gradient, layer at threshold, layer at model thickness jsb
            dz += ddz
            ΔVs = abs(Vs0 - np.interp(np.sum(depth) + dz,
                            np.cumsum(vsv_model.thickness),
                            vsv_model.vsv.flatten()))

        depth = np.append(depth, dz)

        if dz < model_params.mineos_intervals[2]:
            depth = np.append(depth, 0)

        Vs0   = np.interp(np.sum(depth),
                        np.cumsum(vsv_model.thickness),
                        vsv_model.vsv.flatten())

    depth     = np.flipud(np.cumsum(depth))
    depth[0]  = model_params.depth_limits[1]#it will have overshot
    radius    = (radius_Earth - depth)*1e3
    """

    #old block that makes a card model every 2 km
    depth  = np.cumsum(depth)
    step   = model_params.min_layer_thickness / 3
    radius = np.arange(radius_model_base, radius_model_top, step)
    radius = np.append(radius, radius_model_top)
    depth  = (radius_Earth - radius) # still in km at this point
    radius *= 1e3 # convert to SI

    vsv = np.interp(depth,
                    np.cumsum(vsv_model.thickness),
                    vsv_model.vsv.flatten()) * 1e3 # convert to SI
    rho = np.interp(radius, ref_model['radius'], ref_model['rho'])

    #now check if there are "extra layers" to attached on to the model
    #will be given as thickness and vsv
    if model_params.extra_layers.size != 0:
        #convert thickness to radius
        radius = np.hstack( (radius, np.cumsum(model_params.extra_layers[0]) + max(radius)))
        vsv    = np.hstack( (vsv, model_params.extra_layers[1]))
        rho    = np.hstack( (rho, model_params.extra_layers[2]))

    vsh = vsv / model_params.vsv_vsh_ratio
    vpv = vsv * model_params.vpv_vsv_ratio
    vph = vpv / model_params.vpv_vph_ratio
    eta = np.ones(vsv.shape) * model_params.eta
    q_mu = np.interp(radius, ref_model['radius'], ref_model['q_mu'])
    q_kappa = np.interp(radius, ref_model['radius'], ref_model['q_kappa'])

    vpv[vpv<1500] = 1500
    vph[vph<1500] = 1500

    bnames, _, _ = model_params.boundaries
    if 'Moho' in bnames:
        Moho_ind = (vsv_model.boundary_inds[bnames.index('Moho')])
        Moho_depth = np.sum(vsv_model.thickness[:Moho_ind + 1])
    else:
        try:
            Moho_depth = kwargs['Moho']
        except:
            print('Moho depth never specified! Guessing 35 km.')
            Moho_depth = 35
    #rho[(depth <= Moho_depth) & (2900 < rho)] = 2900

    # Now paste the models together, with 100 km of smoothing between
    new_model = pd.DataFrame({
        'radius': radius,
        'rho': rho,
        'vpv': vpv,
        'vsv': vsv,
        'q_kappa': q_kappa,
        'q_mu': q_mu,
        'vph': vph,
        'vsh': vsh,
        'eta': eta,
    })
    smoothed_below = smooth_to_ref_model_below(ref_model, new_model)
    smoothed_above = smooth_to_ref_model_above(ref_model, new_model)

    mineos_card_model = pd.concat([smoothed_below, new_model,
                                   smoothed_above]).reset_index(drop=True)

    vpv    = np.flip(mineos_card_model['vpv'].to_numpy())/1e3
    vsv    = np.flip(mineos_card_model['vsv'].to_numpy())/1e3
    rho    = np.flip(mineos_card_model['rho'].to_numpy())/1e3
    radius = np.flip(mineos_card_model['radius'].to_numpy())/1e3

    #earth flatten depths/values
    vm = np.empty([len(vsv)-1,4])
    for k in range(len(vsv)-1):

        if k < 2:
            vm[k,0] = radius[k] - radius[k+1]
        else:
            vm[k,0] = radius_Earth*math.log(radius_Earth/radius[k]) - radius_Earth*math.log(radius_Earth/radius[k-1])

        if vm[k,0] == 0:
            vm[k,0] = 0.1

        if k < (len(vsv)-1):

            vm[k,1]     = 0.5*(vpv[k] + vpv[k+1])*(radius_Earth/radius[k])
            vm[k,2]     = 0.5*(vsv[k] + vsv[k+1])*(radius_Earth/radius[k])
            vm[k,3]     = 0.5*(rho[k] + rho[k+1])*(radius_Earth/radius[k])**-2.275

        else:

            vm[k,1]     = vpv[k]*(radius_Earth/radius[k])
            vm[k,2]     = vsv[k]*(radius_Earth/radius[k])
            vm[k,3]     = rho[k]*(radius_Earth/radius[k])**-2.275

    vm = vm[:np.abs(np.cumsum(vm[:,0]) - 6371*math.log(6371/5371)).argmin()+1,:] #1Mm max

    return vm

def _write_mineos_card(mineos_card_model:pd.DataFrame, name:str):
    """ Write the MINEOS card model txt file to (name).card.

    Given a pandas DataFrame with the following columns:
        radius, rho, vpv, vsv, q_kappa, q_mu, vph, vsh, eta
    write a model card text file to output/(name)/(name).card.

    All of the values in the input DataFrame are assumed to have the correct units etc.  This code will work out how many inner core layers and total core layers there are for the header of the model card.

    The layout of the card file:
    First line:
                    name of the model card
    Second line:
                    if_anisotropic   t_ref   if_deck
        - This is hardwired with if_anisotropic = 1, i.e. assume anisotropy
              (even if not truly anisotropic)
        - tref is the reference period for dispersion calculation,
              However, we correct for dispersion separately, so setting it to < 1 means no dispersion corrections are done at this stage
        - NOTE: If set tref > 0, MINEOS will break when running the Q correction through mineos_qcorrectphv, as the automatically generated input file assumes tref < 0, so it enters an if statement in the Fortran file that requires 'y'.  If tref > 0, having y in the runfile breaks the code.
        - if_deck set to 1 for a model card or 0 for a polynomial model
    Third line:
                    total_layers, index_top_of_inner_core, i_top_of_outer_core
    Rest of file (each line is a depth point):
                    radius  rho  vpv  vsv  q_kappa  q_mu  vph  vsh  eta

    Arguments:
        mineos_card_model:
            - pd.DataFrame
            - Units:    SI - m, m/s, etc
            - Columns radius, rho, vpv, vsv, q_kappa, q_mu, vph, vsh, eta
        name:
            - str
            - Used for saving model to output/(name)/(name).card
    Returns:
        MINEOS card file (format described above) is written to disk at output/(name)/(name).card

    """

    # Write MINEOS model to .card (txt) file
    # Find the values for the header line
    outer_core = mineos_card_model[(mineos_card_model.vsv == 0)
                                   & (mineos_card_model.q_mu == 0)]
    n_inner_core_layers = outer_core.iloc[[0]].index[0]
    n_core_layers = outer_core.iloc[[-1]].index[0] + 1

    fid = open('output/{0}/{0}.card'.format(name), 'w')
    fid.write(name + '\n  1   -1   1\n')
    fid.write('  {0:d}   {1:d}   {2:d}\n'.format(mineos_card_model.shape[0],
            n_inner_core_layers, n_core_layers));
    # Now print the model
    for index, row in mineos_card_model.iterrows():
        fid.write('{0:6.0f}. {1:8.2f} {2:8.2f} {3:8.2f} '.format(
            row['radius'], row['rho'], row['vpv'], row['vsv']
        ))
        fid.write('{0:8.1f} {1:8.1f} {2:8.2f} {3:8.2f} {4:8.5f}\n'.format(
            row['q_kappa'], row['q_mu'], row['vph'], row['vsh'], row['eta']
        ))

    fid.close()



def smooth_to_ref_model_below(ref_model:pd.DataFrame, new_model:pd.DataFrame
                              ) -> pd.DataFrame:
    """ Take a model of the relatively shallow Earth and smoothly append the reference model

    Smooth the parameters over the 100 km below the base of new_model, then append the reference model.  Note that the bottom value of new_model should be pinned at the corresponding value for ref_model, so there won't really need to be a lot of smoothing!

    Arguments:
        - ref_model:
            - pd.DataFrame
            - Units:    SI - m, m/s etc
            - Model specified in ModelParams.ref_card_csv_name
        - new_model:
            - pd.DataFrame
            - Units:    SI
            - Model built from VsvModel using values from ModelParams
            - Only extends to ModelParams.depth_limits, and needs to be for the whole Earth for MINEOS
    Returns:
        - smoothed model
            - pd.DataFrame
            - Units:    SI
            - new_model smoothly merged with ref_model.
    """

    smooth_z = 100 * 1e3  # 100 km in SI units - depth range to smooth over
    base_of_smoothing = new_model['radius'].iloc[0] - smooth_z
    unadulterated_ref_model = ref_model[ref_model['radius'] < base_of_smoothing]
    smoothed_ref_model = ref_model[
        (base_of_smoothing <= ref_model['radius'])
        & (ref_model['radius'] < new_model['radius'].iloc[0])
    ].copy()

    fraction_new_model = (
        (smoothed_ref_model['radius'] - base_of_smoothing) / smooth_z
    )

    for col in ref_model.columns.tolist()[1:]: # remove radius
        ref_value_at_model_base = np.interp(new_model['radius'].iloc[0],
                                            ref_model['radius'], ref_model[col])
        smoothed_ref_model[col] += (
            (new_model[col].iloc[0] - ref_value_at_model_base)
            * fraction_new_model
        )

    return pd.concat([unadulterated_ref_model, smoothed_ref_model])

def smooth_to_ref_model_above(ref_model:pd.DataFrame, new_model:pd.DataFrame
                              ) -> pd.DataFrame:
    """ Take a model of the relatively shallow Earth and smoothly prepend the reference model

    Smooth the parameters over the 100 km above the base of new_model, then prepend the reference model.  If the ModelParams.depth_limits starts at 0 kilometres, the smoothed model will be identical to the input new_model.

    Arguments:
        - ref_model:
            - pd.DataFrame
            - Units:    SI - m, m/s etc
            - Model specified in ModelParams.ref_card_csv_name
        - new_model:
            - pd.DataFrame
            - Units:    SI
            - Model built from VsvModel using values from ModelParams
            - Only extends over ModelParams.depth_limits, and needs to be for the whole Earth for MINEOS
    Returns:
        - smoothed model
            - pd.DataFrame
            - Units:    SI
            - new_model smoothly merged with ref_model
    """

    smooth_z = 100 * 1e3  # 100 km in SI units - depth range to smooth over
    top_of_smoothing = new_model['radius'].iloc[-1] + smooth_z
    unadulterated_ref_model = ref_model[top_of_smoothing < ref_model['radius']]
    smoothed_ref_model = ref_model[
        (new_model['radius'].iloc[-1] < ref_model['radius'])
        & (ref_model['radius'] <= top_of_smoothing )
    ].copy()

    fraction_new_model = (
        (top_of_smoothing - smoothed_ref_model['radius']) / smooth_z
    )

    for col in ref_model.columns.tolist()[1:]: # remove radius
        ref_value_at_model_top = np.interp(new_model['radius'].iloc[-1],
                                            ref_model['radius'], ref_model[col])
        smoothed_ref_model[col] += (
            (new_model[col].iloc[-1] - ref_value_at_model_top)
            * fraction_new_model
        )

    return pd.concat([smoothed_ref_model, unadulterated_ref_model])


def _set_earth_layer_indices(model_params:ModelParams, model:VsvModel, **kwargs) -> EarthLayerIndices:
    """ Return the indices of different geological layers in the model.

    This is useful for if we want to damp the different geologically significant layers with different
    parameters.  Here, we distinguish sediment (any Vsv below an arbitrary threshold value), crust, lithosphere, and asthenosphere.  The Moho and LAB will be taken from model_params and model if they are there, or should be specified as a kew word argument if they are not.

    Arguments:
        model_params:
            - ModelParams object
            - Units:    seismological
        model:
            - VsvModel object
            - Units:    seismological
        kwargs:
            - key word arguments of format
                - Moho = some_float
                - LAB = some_float
            - Units:    km
            - If Moho and LAB are not specified in model and model_params, they must be set here to distinguish crust, lithospheric mantle, and asthenosphere
    Returns:
        layer indices:
            - EarthLayerIndices object
            - Units:    mostly indices (ints), depth field is in kilometres
    """

    # Last model point is not included in the inversion
    #depth = np.cumsum(model.thickness[:-1])
    #jsb needs to account for possibly more than one excluded node
    depth = np.cumsum(model.thickness[model.d_inds])

    max_sediment_velocity = 3
    sed_inds = [list(model.vsv).index(v) for v in model.vsv if v <= max_sediment_velocity]

    bnames, _, _ = model_params.boundaries
    if 'Moho' in bnames:
        moho_ind = model.boundary_inds[bnames.index('Moho')]
    else:
        try:
            moho_ind = np.argmin(np.abs(depth - kwargs['Moho']))
        except:
            print('Moho depth never specified! Guessing 35 km.')
            moho_ind = np.argmin(np.abs(depth - 35))

    if 'LAB' in bnames:
        lab_ind = model.boundary_inds[bnames.index('LAB')]
    else:
        try:
            lab_ind = np.argmin(np.abs(depth - kwargs['LAB']))
        except:
            #print('LAB depth never specified! Guessing 80 km.')
            #lab_ind = np.argmin(np.abs(depth - 80))
            print('LAB depth never specified! Guessing base of model, ie no boundary.')
            lab_ind = len(depth)-1#np.argmin(np.abs(depth - 80))

    return EarthLayerIndices(
        sediment = np.array(sed_inds),
        crust = np.arange(len(sed_inds), moho_ind + 1),
        lithospheric_mantle = np.arange(moho_ind + 1, lab_ind + 1),
        asthenosphere = np.arange(lab_ind + 1, len(depth)),
        boundary_layers = len(depth) + np.arange(len(model.boundary_inds)),
        depth = list(depth)
    )

def save_model(model:VsvModel, fname:str):
    """ Save VsvModel to file, to be read with read_model(fname).
    Arguments:
        model:
            - VsvModel object
            - Units:    seismological
        fname:
            - str
            - Model will be saved to output/models/(fname).csv
    Returns:
        Model will be saved to output/models/(fname).csv

    """
    save_dir = 'output/models/'
    #save_dir = 'Qtest/'

    with open('{}{}.csv'.format(save_dir, fname), 'w') as fid:
        fid.write('{}\n\n'.format(fname))
        fid.write('Boundary Indices: \n')
        for b in model.boundary_inds:
            fid.write('{:},'.format(b))

        fid.write('\n\nvsv,thickness,inverted_inds\n')
        for i in range(len(model.vsv)):
            fid.write('{:.2f},{:.2f},'.format(model.vsv.item(i), model.thickness.item(i)))
            if i in model.d_inds:
                fid.write('True\n')
            else:
                fid.write('False\n')

        fid.close()

def save_model_stats(models:list, fname:str, mp:ModelParams):
    """ added by jsb
    This takes a set of vsv values generated by many iterations of the inversion
    and saves the results in a standard format with errors. Stats of interest from
    the ensemble of bootstrapped models are saved in one file, and then in another
    file the actualy model with depths is saved. The files are simple csv files

    """
    save_dir = 'output/models/'

    #first, get the depth and make finely interpolated vs
    m = models[0]
    md = sum(m.thickness)
    md = round(md[0])#its an array, extract value

    depth = np.linspace(0, md, md + 1) #1km spacing
    vs = []
    dvsmoho = []
    dvsnvg = []
    lithosphericthickness = []
    lithospheregradient = []
    asthenospheregradient = []
    mohodepth = []
    vscrust = []
    vsasthenosphere = []

    for k in np.arange(0, len(models)):
        vs.append(np.interp(depth, np.cumsum(models[k].thickness), models[k].vsv.flatten()))

        dvsmoho.append((100*(models[k].vsv[models[k].boundary_inds[0] + 1] - models[k].vsv[models[k].boundary_inds[0]])/models[k].vsv[models[k].boundary_inds[0]])[0])
        dvsnvg.append((100*(models[k].vsv[models[k].boundary_inds[1] + 1] - models[k].vsv[models[k].boundary_inds[1]])/models[k].vsv[models[k].boundary_inds[1]])[0])

        vsasthenosphere.append((models[k].vsv[models[k].boundary_inds[1] + 1])[0])
        vscrust.append((models[k].vsv[models[k].boundary_inds[0]])[0])

        lithosphericthickness.append((sum(models[k].thickness[:models[k].boundary_inds[1]]) - sum(models[k].thickness[:models[k].boundary_inds[0]]))[0])
        mohodepth.append((sum(models[k].thickness[:models[k].boundary_inds[0]]))[0])

        lithospheregradient.append(((models[k].vsv[models[k].boundary_inds[1]] - models[k].vsv[models[k].boundary_inds[0] + 1])/lithosphericthickness[k])[0])
        asthenospheregradient.append((models[k].vsv[models[k].boundary_inds[1] + 2] - models[k].vsv[models[k].boundary_inds[1] + 1])) #1km spacing

    #seperate, so that you can do
    lithosphericthickness = np.add(lithosphericthickness, mohodepth)

    fid = open(save_dir + fname + ".csv", "w")
    fid.write("moho dVs, %, NVG dVs, %, Moho depth, km, NVG depth, km, Vs above moho, km/s, Vs under NVG, km/s, Lithospheric gradient, (km/s)/km, Asthenosphere gradient, (km/s)/km" + "\n")
    fid.write(str(np.mean(dvsmoho))               + "," + str(np.std(dvsmoho))                + "\n")
    fid.write(str(np.mean(dvsnvg))                + "," + str(np.std(dvsnvg))                 + "\n")
    fid.write(str(np.mean(mohodepth))             + "," + str(np.std(mohodepth))              + "\n")
    fid.write(str(np.mean(lithosphericthickness)) + "," + str(np.std(lithosphericthickness))  + "\n")
    fid.write(str(np.mean(vscrust))               + "," + str(np.std(vscrust))                + "\n")
    fid.write(str(np.mean(vsasthenosphere))       + "," + str(np.std(vsasthenosphere))        + "\n")
    fid.write(str(np.mean(lithospheregradient))   + "," + str(np.std(lithospheregradient))    + "\n")
    fid.write(str(np.mean(asthenospheregradient)) + "," + str(np.std(asthenospheregradient)))
    fid.close()

    vsmodel = np.mean(vs, 0)
    vserror = np.std(vs, 0)

    fid = open(save_dir + fname + "-model.csv", "w")
    fid.write("depth, km, model Vs, km/s,  1-sig standard error\n")
    for k in np.arange(0, len(depth)):
        if k < (len(depth)-1):
            fid.write(str(depth[k]) + "," + str(vsmodel[k]) + "," + str(vserror[k]) + "\n")
        else:
            fid.write(str(depth[k]) + "," + str(vsmodel[k]) + "," + str(vserror[k]))
    fid.close()

def save_model_set(models:list, fname:str, mp:ModelParams):
    """ added by jsb
    """
    save_dir = 'output/models/'

    #first, get the depth and make finely interpolated vs
    m = models[0]
    md = sum(m.thickness)
    md = round(md[0])#its an array, extract value

    depth = np.linspace(0, md, md + 1) #1km spacing
    vs = np.zeros((len(depth) + 4*len(models[0].boundary_inds),len(models)+1))

    vs[0:len(depth),0] = depth
    for k in np.arange(0, len(models)):
        t = np.cumsum(models[k].thickness)
        vs[0:len(depth),k+1] = np.interp(depth, np.cumsum(models[k].thickness), models[k].vsv.flatten())
        vs[len(depth):len(depth)+len(models[0].boundary_inds),k+1] = t[models[k].boundary_inds]
        vs[-3*len(models[0].boundary_inds):-2*len(models[0].boundary_inds),k+1] = models[k].thickness[models[k].boundary_inds + 1].flatten()
        vs[-2*len(models[0].boundary_inds):-len(models[0].boundary_inds),k+1] = models[k].vsv[models[k].boundary_inds].flatten()
        vs[-len(models[0].boundary_inds):,k+1] = models[k].vsv[models[k].boundary_inds + 1].flatten()

    fid = open(save_dir + fname + "-models.csv", "w")
    csvw = csv.writer(fid, delimiter=',')
    csvw.writerows(vs)
    fid.close()

def read_model(fname:str, save_dir:str = 'output/models/'): #jsb added option to customize save dir
    """ Read VsvModel from file.
    Arguments:
        fname:
            - str
            - Model will be opened from output/models/(fname).csv
    Returns:
        model:
            - VsvModel object
            - Units:    seismological
    """

    with open('{}{}.csv'.format(save_dir, fname), 'r') as fid:
        bi = pd.read_csv(fid, skiprows=3, header=None, nrows=1)
    with open('{}{}.csv'.format(save_dir, fname), 'r') as fid:
        params = pd.read_csv(fid, skiprows=5)

    return VsvModel(
        vsv = params.vsv.values[:, np.newaxis],
        thickness = params.thickness.values[:, np.newaxis],
        boundary_inds = bi.iloc[0,:-1].astype(int).values,
        d_inds = np.arange(len(params.vsv))[params.inverted_inds.values],
    )

def load_all_models(z:np.array, lats:np.array, lons:np.array, t_LAB:int, lab:str=''
                    ) -> (np.array, np.array, np.array):
    """ Load a bunch of models from disk to make plotting comparisons easier

    This assumes you have saved a bunch of models with filenames of the following format:
        output/models/[lat]N_[lon]W_[t_LAB]kmLAB[lab]
    Also, that you have calculated a rectangle of models, such that any combination of the values in lats and lons will have a model file saved.  This also assumes that your VsvModel has two
     layers in it (i.e. Moho, LAB)

    Arguments:
        z:
            - (n_z, ) np.array
            - Units:    kilometres
            - Whatever vector of depth points you want for plotting
            - All loaded models will be interpolated along this vector
        lats:
            - (n_lats, ) np.array
            - Units:    to match your saved file names
            - This is only used to sub into file names, [lats[i]]N_[lon]W_[t_LAB]kmLAB[lab]
        lons:
            - (n_lons, ) np.array
            - Units:    to match your saved file names
            - This is only used to sub into file names, [lat]N_[lons[i]]W_[t_LAB]kmLAB[lab]
        t_LAB:
            - int
            - Units:    to match your saved file names
            - This is only used to sub into file names, [lat]N_[lon]W_[t_LAB]kmLAB[lab]
            - If you save your model file using a script, make sure that t_LAB is in the same format (float or int), as '{}'.format(10.) -> '10.0' and '{}'.format(10) -> '10'
        lab:
            - str
            - If you have some further identifying information in a label appended to your saved file names, you can pass it in here

    Returns:
        vs:
            - (n_lats, n_lons, n_z) np.array
            - Units:    km/s
            - Tensor of Vsv values loaded in from saved VsvModel files
        bls:
            - (n_lats, n_lons, 4) np.array
            - Units:    seismological - km, km/s
            - Values around boundary layers
            - Assumes you have two boundary layers
            - bls[:, :, 0] is the depth to the centre of the upper boundary layer (e.g. Moho)
            - bls[:, :, 1] is the depth to the centre of the lower boundary layer (e.g. LAB)
            - bls[:, :, 2] is the relative velocity change across the upper boundary layer
                - (velocity at base of BL / velocity at top of BL) - 1
            - bls[:, :, 3] is the relative velocity change across the lower boundary layer
                - (velocity at base of BL / velocity at top of BL) - 1
        bis:
            - (n_lats, n_lons, 4) np.array
            - Units:    none
            - Indices of boundary layers
            - Assumes you have two boundary layers
            - bis[:, :, 0] = Index of top of Moho
            - bis[:, :, 1] = Index of base of Moho
            - bis[:, :, 2] = Index of top of LAB
            - bis[:, :, 3] = Index of base of LAB
            - Note that bis[:, :, 1] = bis[:, :, 0] + 1 (and same for lower boundary layer info)

    """

    vs = np.zeros((lats.size, lons.size, z.size))
    bls = np.zeros((lats.size, lons.size, 4), dtype=float)
    bis = np.zeros((lats.size, lons.size, 4), dtype=int)

    i_lat = 0
    i_lon = 0
    for lat in lats:#range(34, 41):
        for lon in lons:
            fname = '{}N_{}W_{}kmLAB{}'.format(lat, lon, t_LAB, lab)
            m = read_model(fname)
            vs[i_lat, i_lon, :] = (
                np.interp(
                    z, np.cumsum(m.thickness).flatten(), m.vsv.flatten()
                    )
            )
            depth_top_BL1 = np.sum(m.thickness[:m.boundary_inds[0] + 1])
            width_BL1 = m.thickness[m.boundary_inds[0] + 1]
            depth_top_BL2 = np.sum(m.thickness[:m.boundary_inds[1] + 1])
            width_BL2 = m.thickness[m.boundary_inds[1] + 1]
            bls[i_lat, i_lon, :] = [
                depth_top_BL1 +  width_BL1 / 2, # Depth to Moho
                depth_top_BL2 + width_BL2 / 2, # Depth to LAB
                m.vsv[m.boundary_inds[0] + 1] / m.vsv[m.boundary_inds[0]] - 1, # Moho dV
                m.vsv[m.boundary_inds[1] + 1] / m.vsv[m.boundary_inds[1]] - 1, # LAB dV
                ]
            bis[i_lat, i_lon, :] = [
                int(np.argmax(depth_top_BL1 < z) - 1), # Index of top of Moho
                int(np.argmax(depth_top_BL1 + width_BL1 <= z)), # Index of base of Moho
                int(np.argmax(depth_top_BL2 < z) - 1), # Index of top of LAB
                int(np.argmax(depth_top_BL2 + width_BL2 <= z)), # Index of base of LAB
                ]
            i_lon += 1
        i_lat += 1
        i_lon = 0

    return vs, bls, bis
