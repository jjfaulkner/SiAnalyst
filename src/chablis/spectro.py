from chablis.preamble import *

##################
# Peak functions #
##################

# c-Si
def lorentzian(x, amp_l, cen_l, wid_l, y):
    return amp_l*wid_l**2/((x-cen_l)**2+wid_l**2) + y
# a-Si
def gaussian(x, amp_g, cen_g, wid_g, y): # divide by 2.354 to convert from FWHM to sigma
    return amp_g*np.exp(-(x.astype(float)-cen_g)**2/(2*(wid_g/2.354)**2)) + y
# standard a-Si with c-Si
def gaussian_plus_lorentzian(x, amp_g1, cen_g1, wid_g1, amp_l, cen_l, wid_l, y):
    return amp_g1*np.exp(-(x.astype(float)-cen_g1)**2/(2*(wid_g1/2.354)**2)) + amp_l*wid_l**2/((x-cen_l)**2+wid_l**2) + y
# thorough a-Si with c-Si (need to read literature)
def triple_gaussian_plus_lorentzian(x, amp_g1, cen_g1, wid_g1, amp_g2, cen_g2, wid_g2, amp_g3, cen_g3, wid_g3, amp_l, cen_l, wid_l, y):
    return amp_g1*np.exp(-(x.astype(float)-cen_g1)**2/(2*(wid_g1/2.354)**2)) + amp_g2*np.exp(-(x.astype(float)-cen_g2)**2/(2*(wid_g2/2.354)**2)) + amp_g3*np.exp(-(x.astype(float)-cen_g3)**2/(2*(wid_g3/2.354)**2)) + amp_l*wid_l**2/((x-cen_l)**2+wid_l**2) + y
# a-C
def fano_plus_lorentzian(x, amp_l, cen_l, wid_l, amp_f, cen_f, wid_f, q, gradient, y_0): # sum of Fano and Lorentzian with a common offset
    s = (x - cen_f) / wid_f
    return y_0 + x*gradient + (amp_f * (1 + s / q) ** 2) / (1 + s ** 2) + amp_l * wid_l ** 2/((x - cen_l) ** 2 + wid_l ** 2)

#########################
# Reading spectral data #
#########################

# a clever nonlinear baseline removal
def baseline_als(y, lam=1e8, p=0.005, niter=10):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def get_map(name):
    global spectra
    spectra = pd.read_csv(name, sep='\t', header=None, skiprows = 41, encoding = 'utf-8').transpose()
    spectra.columns = spectra.iloc[0] # I ideally want the columns to be co-ordinates, rather than the y vals being in row 0
    spectra = spectra[1:]
    spectra = spectra.reset_index(drop=True)
    spectra_shifted = spectra.copy()
    spectra_shifted.iloc[1:,1:] = spectra_shifted.iloc[1:,1:] - spectra_shifted.iloc[1:,1:].min()
    spectra_normalised = spectra_shifted.copy()
    spectra_normalised.iloc[1:,1:] = spectra_shifted.iloc[1:,1:] / spectra_shifted.iloc[1:,1:].max()
    #return spectra_normalised

    # Get the first row values as a list
    first_row_values = spectra_normalised.iloc[0].values.tolist()

    # Create new column names by combining old column names with first row values
    new_columns = [f"{col}_{val}" for col, val in zip(spectra_normalised.columns, first_row_values)]

    # Rename the columns
    spectra_normalised.columns = new_columns
    spectra_normalised.rename(columns={spectra_normalised.columns[0]: "Wavenumber"}, inplace = True)

    # Optionally, if you want to remove the first row after renaming the columns
    spectra_normalised = spectra_normalised.iloc[1:]
    return spectra_normalised

#########################
# Fitting spectral data #
#########################

def get_ref_properties(file):
    i = 0

    global wavenums, intensities, xlims, func
    wavenums = []
    intensities = []
    xlims, guess, func, constraints = [400, 600], [1000, 520, 3, 0], lorentzian, ((0, 510, 0, 0),(np.inf, 530, np.inf, np.inf)) # lorentzian si
    
    while file.iloc[1,0] > xlims[1]:
        i+=1
    while file.iloc[1,0] > xlims[0]:
        wavenums.append(file.iloc[i,0])
        intensities.append(file.iloc[i,1])
        i+=1

    popt, pcov = scipy.optimize.curve_fit(func, wavenums, intensities, p0 = guess, bounds=constraints)    # fit the function to the data with the x and y series transformed to lists
    perr = np.sqrt(np.diag(pcov))
    global properties
    properties = [popt, perr]
    return properties

def get_peak_properties(spectra, column, peak):

    i = 0

    global wavenums, intensities, xlims, func
    wavenums = []
    intensities = []

    if peak == 'c_si':
        xlims, guess, func, constraints = [500, 900], [1, 520, 3, 0], lorentzian, ((0, 510, 0, 0),(np.inf, 530, np.inf, np.inf)) # lorentzian si
    if peak == 'a_si_gauss':
        xlims, guess, func, constraints = [263,900], [0.2, 300, 50, 0.2, 350, 50, 0.2, 480, 50, 1, 520, 3, 0], triple_gaussian_plus_lorentzian, ((0, 280, 0, 0, 330, 0, 0, 450, 0, 0, 510, 0, 0), (np.inf, 310, np.inf, np.inf, 380, np.inf, np.inf, 490, np.inf, np.inf, 530, np.inf, np.inf))
    if peak == 'a_si_single_gauss':
        xlims, guess, func, constraints = [263,900], [0.2, 400, 100, 1, 520, 3, 0], gaussian_plus_lorentzian, ((0, 300, 0, 0, 510, 0, 0), (np.inf, 480, np.inf, np.inf, 530, np.inf, np.inf))
    if peak == 'a_carbon':
        xlims, guess, func, constraints = [800, 2000], [0.4, 1350, 50, 0.4, 1600, 50, -10, 0, 0], fano_plus_lorentzian, ((0, 1300, 0, 0, 1500, 0, -np.inf, -np.inf, -np.inf), (np.inf, 1500, np.inf, np.inf, 1700, np.inf, 0, np.inf, np.inf))

    while spectra.iloc[i,0] < xlims[0]:
        i+=1
    while spectra.iloc[i,0] < xlims[1]:
        wavenums.append(spectra.iloc[i,0])
        intensities.append(spectra.iloc[i,column])
        i+=1

    popt, pcov = scipy.optimize.curve_fit(func, wavenums, intensities, p0 = guess, bounds=constraints)    # fit the function to the data with the x and y series transformed to lists
    perr = np.sqrt(np.diag(pcov))
    global properties
    properties = [popt, perr]
    return properties

def a_si_ratio(spectra, column, peak):
    properties = get_peak_properties(spectra, column, peak)
    if peak == 'a_si_gauss':
        #a_area = (properties[0][0]*properties[0][2] + properties[0][3]*properties[0][5] + properties[0][6]*properties[0][8])
        a_area = (properties[0][6]*properties[0][8] / (2.355 * 0.3989)) # height * width of only the uppermost a-Si mode
        c_area = (properties[0][9]*properties[0][11] * np.pi) # pi/2 * height * width of c-Si mode
    if peak == 'a_si_single_gauss':
        a_area = properties[0][0]*properties[0][2] / (2.355 * 0.3989) # conversion factor: 2 sqrt(2 ln(2)) / sqrt(2pi)
        c_area = (properties[0][3]*properties[0][5] * np.pi)
    a_ratio = a_area * 0.8 / (0.8*a_area + (c_area))
    return a_ratio