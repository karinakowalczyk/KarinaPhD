class Parameters:
    N = 0.01  # Brunt-Vaisala frequency (1/s)
    cp = 1004.5  # SHC of dry air at const. pressure (J/kg/K)
    R_d = 287.  # Gas constant for dry air (J/kg/K)
    kappa = 2.0/7.0  # R_d/c_p
    p_0 = 1000.0*100.0  # reference pressure (Pa, not hPa)
    cp = 1004.
    g = 9.80665


def thermodynamics_rho(parameters, theta_v, pi):
    """
    Returns an expression for the dry density rho in kg / m^3
    from the (virtual) potential temperature and Exner pressure.

    :arg parameters: a CompressibleParameters object.
    :arg theta_v: the virtual potential temperature in K.
    :arg pi: the Exner pressure.
    """

    kappa = parameters.kappa
    p_0 = parameters.p_0
    R_d = parameters.R_d

    return p_0 * pi ** (1 / kappa - 1) / (R_d * theta_v)

def thermodynamics_pi(parameters, rho, theta_v):
    """
    Returns an expression for the Exner pressure.

    :arg parameters: a CompressibleParameters object.
    :arg rho: the dry density of air in kg / m^3.
    :arg theta: the potential temperature (or the virtual
                potential temperature for wet air), in K.
    """

    kappa = parameters.kappa
    p_0 = parameters.p_0
    R_d = parameters.R_d

    return (rho * R_d * theta_v / p_0) ** (kappa / (1 - kappa))
