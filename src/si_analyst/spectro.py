from si_analyst.preamble import *

# single Lorentzian peak
def lorentzian(x, amp_l, cen_l, wid_l):
    return amp_l*wid_l**2/((x-cen_l)**2+wid_l**2)

# single Gaussian peak
def gaussian(x, amp_g, cen_g, wid_g):
    return amp_g*np.exp(-(x.astype(float)-cen_g)**2/(2*wid_g**2))

# overlapping Lorentzian and Gaussian
def gaussian_plus_lorentzian(x, amp_g1, cen_g1, wid_g1, amp_l, cen_l, wid_l):
    return amp_g1*np.exp(-(x.astype(float)-cen_g1)**2/(2*wid_g1**2)) + amp_l*wid_l**2/((x-cen_l)**2+wid_l**2)

# overlapping Lorentzian with triple Gaussian (possibly useful for a-Si)
def triple_gaussian_plus_lorentzian(x, amp_g1, cen_g1, wid_g1, amp_g2, cen_g2, wid_g2, amp_g3, cen_g3, wid_g3, amp_l, cen_l, wid_l):
    return amp_g1*np.exp(-(x.astype(float)-cen_g1)**2/(2*wid_g1**2)) + amp_g2*np.exp(-(x.astype(float)-cen_g2)**2/(2*wid_g2**2)) + amp_g3*np.exp(-(x.astype(float)-cen_g3)**2/(2*wid_g3**2)) + amp_l*wid_l**2/((x-cen_l)**2+wid_l**2)

