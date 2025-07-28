from chablis.preamble import *

A = 1.714e5 # in cm^-2
B = 1e5 # in cm^-2
D = -0.3 # in cm^-1
Gamma_0 = 2.9 # in cm^-1
a = 5.43e-8  # in cm

#mu = np.log(80e-7)
#sigma = 0.149

def get_integral_for_diameter(diameter):
    return functools.partial(calc_intensity_from_raman_shift, diameter = diameter)

vec_get_integral_for_diameter = np.vectorize(get_integral_for_diameter)

def get_spectrum_for_diameter(diameter, num_points: int):
    raman_shift = np.linspace(500, 540, num_points)
    intensities = calc_intensity_from_raman_shift_vectorized(raman_shift, diameter)
    return intensities / np.max(intensities)

get_spectrum_for_diameter_vectorized = np.vectorize(get_spectrum_for_diameter, otypes=[np.ndarray])

def main(particle_size: float, num_points: int): # brings everything together
    raman_shift = np.linspace(500, 530, num_points) # defines the domain over which the intensity is calculated
    integral = functools.partial(calc_intensity_from_raman_shift, diameter=particle_size) # defines the array 'integral' by calculating the intensity at every Raman shift for a given diameter
    plot_graph(raman_shift, integral)
    x_opt, fwhm = get_func_properties(integral)
    print(f'Peak: {x_opt}. FWHM: {fwhm}')
    print(type(integral))

def dist_main(mu, sigma, num_points: int):
    raman_shift = np.linspace(500, 530, num_points) # defines the domain over which the intensity is calculated
    sizes = np.linspace(1e-7, 200e-7, num_points) # defines the domain over which the size distribution is calculated
    calculated_distribution = distribution_vectorized(mu, sigma, sizes)
    plt.plot(sizes, calculated_distribution)
    plt.show()
    contributions = get_spectrum_for_diameter_vectorized(sizes, num_points)
    spectrum = np.matmul(calculated_distribution, contributions)
    spectrum = spectrum/np.max(spectrum)
    plt.xlabel('Raman shift (cm$^{-1}$)')
    plt.ylabel('Intensity (a.u.)')
    plt.plot(raman_shift, spectrum)
    plt.show()
    x_opt, fwhm = get_func_properties(spectrum)

def compare(mu, sigma, particle_size: float, num_points: int):
    raman_shift = np.linspace(500, 530, num_points) # defines the domain over which the intensity is calculated
    integral = functools.partial(calc_intensity_from_raman_shift, diameter=particle_size) # defines the array 'integral' by calculating the intensity at every Raman shift for a given diameter
    integral_vectorized = np.vectorize(integral)
    intensity = integral_vectorized(raman_shift)
    # normalize intensity to have a maximum value of 1
    intensity = intensity / np.max(intensity)
    
    sizes = np.linspace(1e-7, 200e-7, num_points) # defines the domain over which the size distribution is calculated
    calculated_distribution = distribution_vectorized(mu, sigma, sizes)
    plt.plot(sizes, calculated_distribution)
    plt.show()
    contributions = get_spectrum_for_diameter_vectorized(sizes, num_points)
    spectrum = np.matmul(calculated_distribution, contributions)
    spectrum = spectrum/np.max(spectrum)
    
    plt.plot(raman_shift, intensity)
    plt.plot(raman_shift, spectrum)
    plt.xlabel('Raman shift (cm$^{-1}$)')
    plt.ylabel('Intensity (a.u.)')
    plt.xlim([500, 530])
    
    x_opt, fwhm = get_func_properties(integral)
    print(f'Peak: {x_opt}. FWHM: {fwhm}')
    
def plot_graph(raman_shift, integral):
    # vectorize `integral` so that it can be applied to raman_shift
    integral_vectorized = np.vectorize(integral)
    intensity = integral_vectorized(raman_shift)
    # normalize intensity to have a maximum value of 1
    intensity = intensity / np.max(intensity)
    plt.xlabel('Raman shift (cm$^{-1}$)')
    plt.ylabel('Intensity (a.u.)')
    plt.plot(raman_shift, intensity)
    plt.show()

def get_func_properties(integral):
    x_opt, f_opt, _, _, _, = optimize.fmin(lambda x: -integral(x), 520.76, full_output=True)
    shifted_integral = lambda x: integral(x) + f_opt / 2
    x_left = optimize.root_scalar(shifted_integral, bracket=[450, x_opt]).root
    x_right = optimize.root_scalar(shifted_integral, bracket=[x_opt, 550]).root
    fwhm = x_right - x_left
    return x_opt[0], fwhm

vec_get_func_properties = np.vectorize(get_func_properties)

def calc_intensity_from_raman_shift(raman_shift: float, diameter: float):
    def C_squared(q):
        return np.exp(-q**2 * (diameter)**2 / (16 * np.pi**2))

    def omega(q):
        return np.sqrt(A + B * np.cos(q * a/np.pi)) + D

    def integrand(q):
        return q**2 * 1e-17 * C_squared(q) / ((raman_shift - omega(q))**2 + (Gamma_0 / 2)**2)

    integral_result = integrate.quad(integrand, 0, np.pi/a, limit=2000)
    return integral_result[0]

calc_intensity_from_raman_shift_vectorized = np.vectorize(calc_intensity_from_raman_shift)

def distribution(mu, sigma, x):
    dis1 = np.log(x) - mu
    dis2 = sigma*np.sqrt(2)
    dis3 = np.exp(-(dis1/dis2)**2)
    dis4 = x*sigma*np.sqrt(2*np.pi)
    return dis3/dis4
 
distribution_vectorized = np.vectorize(distribution)

def pos_fwhm():
    diameters = np.linspace(4 * 10**-7,100 * 10**-7, 30)
    integrals = vec_get_integral_for_diameter(diameters)
    peak, fwhm = vec_get_func_properties(integrals)
    plt.plot(peak, fwhm)
    plt.xlabel('Raman shift (cm$^{-1}$)')
    plt.ylabel('FWHM (cm$^{-1}$)')
    plt.xlim(510, 522)
    plt.ylim(0, 30)
    print(integrals)

def isolate_rcf_data(map_data):
    map_data_reduced = map_data[(map_data['Wavenumber'] > 500) & (map_data['Wavenumber'] < 540)]
    baselines = map_data_reduced.iloc[:,1:].apply(baseline_als)
    map_data_reduced.iloc[:,1:] = map_data_reduced.iloc[:,1:] - baselines
    return map_data_reduced

def solve_distribution(spectrum):
    num_points = len(spectrum.iloc[:,0])
    sizes = np.linspace(5e-7, 30e-7, num_points) # defines the domain over which the size distribution is calculated
    contributions = get_spectrum_for_diameter_vectorized(sizes, num_points) # calculates the contribution of each particle size to the final spectrum
    unpacked_contributions = np.array([np.array(contributions[i]/np.max(contributions[i])) for i in range(num_points)]) # unpacks the contributions into a 2D array, normalising each

    transpose = np.transpose(unpacked_contributions) # transposes the array so that the rows are the Raman shifts and the columns are the sizes

    #calculated_distribution = distribution_vectorized(np.log(40e-7), 0.249, sizes)
    #spectrum = np.matmul(calculated_distribution, contributions)
    #spectrum = spectrum/np.max(spectrum)

    #spectrum = get_spectrum_for_diameter(10e-7, num_points) # calculates the spectrum for a single particle size (to be tested)

    #spectrum.reshape(num_points,1) # reshapes the spectrum into a column vector

    # Now need to solve the linear system of equations to find the size distribution. 
    # We have the form Ax = b, where A is the transpose of the contributions array, x is the size distribution and b is the spectrum, both column vectors. 
    # We could solve this using the numpy function np.linalg.solve(A,b) but this returns negative values for some sizes
    # Using the following minimisation function finds the best-fit size distribution without allowing the non-physical negative elements
    # It essentially solves the equation, but replaces any negative elements with 0 and deletes those rows/columns before solving again.
    # This is repeated until a solution is found where all elements are positive, and the extra 0s are added back in

    func = lambda x: np.linalg.norm(np.dot(transpose,x)-spectrum.iloc[:,1])
    # xo = np.linalg.solve(A,b)
    # sol = minimize(fun, xo, method='SLSQP', constraints={'type': 'ineq', 'fun': lambda x:  x})
    sol = minimize(func, np.zeros(num_points), method='L-BFGS-B', bounds=[(0.,None) for x in sizes])

    x = sol['x']
    return pd.DataFrame({'size': sizes, 'frequency': x})

def get_size_distributions(data):
    distributions = pd.DataFrame({'sizes': np.linspace(5e-7, 30e-7, len(data.iloc[:,0]))})
    for i in range(1, len(data.columns)):
        distribution = solve_distribution(data.iloc[:,[0,i]])
        distributions[f'spectrum_{i}'] = distribution['frequency']
    return distributions