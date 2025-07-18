import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
from mpl_toolkits.mplot3d import axes3d
'''
Data
'''

Ene = [
np.array([-1.,  3.]),
np.array([-3., -1.,  1.,  5.]),
np.array([-3., -1.,  1.,  3.,  7.]),
np.array([-5., -3., -1.,  1.,  3.,  5.,  9.]),
np.array([-5., -3., -1.,  1.,  3.,  5.,  7., 11.]),
np.array([-7., -5., -3., -1.,  1.,  3.,  5.,  7.,  9., 13.]),
np.array([-7., -5., -3., -1.,  1.,  3.,  5.,  7.,  9., 11., 15.]),
np.array([-9., -7., -5., -3., -1.,  1.,  3.,  5.,  7.,  9., 11., 13., 17.]),
np.array([-9., -7., -5., -3., -1.,  1.,  3.,  5.,  7.,  9., 11., 13., 15., 19.]),
np.array([-11.,  -9.,  -7.,  -5.,  -3.,  -1.,   1.,   3.,   5.,   7.,   9., 11.,  13.,  15.,  17.,  21.]),
np.array([-11.,  -9.,  -7.,  -5.,  -3.,  -1.,   1.,   3.,   5.,   7.,   9., 11.,  13.,  15.,  17.,  19.,  23.]),
np.array([-13., -11.,  -9.,  -7.,  -5.,  -3.,  -1.,   1.,   3.,   5.,   7., 9.,  11.,  13.,  15.,  17.,  19.,  21.,  25.]),
np.array([-13., -11.,  -9.,  -7.,  -5.,  -3.,  -1.,   1.,   3.,   5.,   7., 9.,  11.,  13.,  15.,  17.,  19.,  21.,  23.,  27.]),
np.array([-15., -13., -11.,  -9.,  -7.,  -5.,  -3.,  -1.,   1.,   3.,   5., 7.,   9.,  11.,  13.,  15.,  17.,  19.,  21.,  23.,  25.,  29.]),
np.array([-15., -13., -11.,  -9.,  -7.,  -5.,  -3.,  -1.,   1.,   3.,   5., 7.,   9.,  11.,  13.,  15.,  17.,  19.,  21.,  23.,  25.,  27., 31.]),
np.array([-17., -15., -13., -11.,  -9.,  -7.,  -5.,  -3.,  -1.,   1.,   3., 5.,   7.,   9.,  11.,  13.,  15.,  17.,  19.,  21.,  23.,  25., 27.,  29.,  33.]),
np.array([-17., -15., -13., -11.,  -9.,  -7.,  -5.,  -3.,  -1.,   1.,   3., 5.,   7.,   9.,  11.,  13.,  15.,  17.,  19.,  21.,  23.,  25., 27.,  29.,  31.,  35.]),
np.array([-19., -17., -15., -13., -11.,  -9.,  -7.,  -5.,  -3.,  -1.,   1., 3.,   5.,   7.,   9.,  11.,  13.,  15.,  17.,  19.,  21.,  23., 25.,  27.,  29.,  31.,  33.,  37.]),
np.array([-19., -17., -15., -13., -11.,  -9.,  -7.,  -5.,  -3.,  -1.,   1., 3.,   5.,   7.,   9.,  11.,  13.,  15.,  17.,  19.,  21.,  23., 25.,  27.,  29.,  31.,  33.,  35.,  39.]),
np.array([-21., -19., -17., -15., -13., -11.,  -9.,  -7.,  -5.,  -3.,  -1., 1.,   3.,   5.,   7.,   9.,  11.,  13.,  15.,  17.,  19.,  21., 23.,  25.,  27.,  29.,  31.,  33.,  35.,  37.,  41.]),
np.array([-21., -19., -17., -15., -13., -11.,  -9.,  -7.,  -5.,  -3.,  -1., 1.,   3.,   5.,   7.,   9.,  11.,  13.,  15.,  17.,  19.,  21., 23.,  25.,  27.,  29.,  31.,  33.,  35.,  37.,  39.,  43.]),
np.array([-23., -21., -19., -17., -15., -13., -11.,  -9.,  -7.,  -5.,  -3., -1.,   1.,   3.,   5.,   7.,   9.,  11.,  13.,  15.,  17.,  19., 21.,  23.,  25.,  27.,  29.,  31.,  33.,  35.,  37.,  39.,  41., 45.]),
np.array([-23., -21., -19., -17., -15., -13., -11.,  -9.,  -7.,  -5.,  -3., -1.,   1.,   3.,   5.,   7.,   9.,  11.,  13.,  15.,  17.,  19., 21.,  23.,  25.,  27.,  29.,  31.,  33.,  35.,  37.,  39.,  41., 43.,  47.]),
np.array([-25., -23., -21., -19., -17., -15., -13., -11.,  -9.,  -7.,  -5., -3.,  -1.,   1.,   3.,   5.,   7.,   9.,  11.,  13.,  15.,  17., 19.,  21.,  23.,  25.,  27.,  29.,  31.,  33.,  35.,  37.,  39., 41.,  43.,  45.,  49.]),
np.array([-25., -23., -21., -19., -17., -15., -13., -11.,  -9.,  -7.,  -5., -3.,  -1.,   1.,   3.,   5.,   7.,   9.,  11.,  13.,  15.,  17., 19.,  21.,  23.,  25.,  27.,  29.,  31.,  33.,  35.,  37.,  39., 41.,  43.,  45.,  47.,  51.]),
np.array([-27., -25., -23., -21., -19., -17., -15., -13., -11.,  -9.,  -7., -5.,  -3.,  -1.,   1.,   3.,   5.,   7.,   9.,  11.,  13.,  15., 17.,  19.,  21.,  23.,  25.,  27.,  29.,  31.,  33.,  35.,  37., 39.,  41.,  43.,  45.,  47.,  49.,  53.])
]

Deg = [
np.array([6., 2.]),
np.array([2., 8., 4., 2.]),
np.array([ 8., 10.,  8.,  4.,  2.]),
np.array([ 2., 16., 20., 10., 10.,  4.,  2.]),
np.array([10., 28., 36., 24., 12., 12.,  4.,  2.]),
np.array([ 2., 26., 56., 58., 50., 30., 14., 14.,  4.,  2.]),
np.array([ 12.,  56., 108., 110.,  84.,  68.,  36.,  16.,  16.,   4.,   2.]),
np.array([  2.,  38., 122., 196., 212., 166., 116.,  88.,  42.,  18.,  18., 4.,   2.]),
np.array([ 14.,  96., 256., 372., 384., 336., 234., 152., 110.,  48.,  20., 20.,   4.,   2.]),
np.array([  2.,  52., 230., 510., 718., 726., 624., 492., 312., 192., 134., 54.,  22.,  22.,   4.,   2.]),
np.array([  16.,  150.,  524., 1016., 1356., 1408., 1200.,  932.,  680., 400.,  236.,  160.,   60.,   24.,   24.,    4.,    2.]),
np.array([2.000e+00, 6.800e+01, 3.940e+02, 1.132e+03, 2.028e+03, 2.588e+03, 2.670e+03, 2.374e+03, 1.820e+03, 1.312e+03, 9.020e+02, 4.980e+02, 2.840e+02, 1.880e+02, 6.600e+01, 2.600e+01, 2.600e+01, 4.000e+00, 2.000e+00]),
np.array([1.800e+01, 2.200e+02, 9.700e+02, 2.392e+03, 3.992e+03, 5.008e+03,5.102e+03, 4.568e+03, 3.674e+03, 2.600e+03, 1.770e+03, 1.160e+03,6.060e+02, 3.360e+02, 2.180e+02, 7.200e+01, 2.800e+01, 2.800e+01, 4.000e+00, 2.000e+00]),
np.array([2.000e+00, 8.600e+01, 6.300e+02, 2.254e+03, 4.996e+03, 7.840e+03, 9.642e+03, 9.896e+03, 8.820e+03, 7.180e+03, 5.356e+03, 3.556e+03,2.312e+03, 1.456e+03, 7.240e+02, 3.920e+02, 2.500e+02, 7.800e+01,3.000e+01, 3.000e+01, 4.000e+00, 2.000e+00]),
np.array([2.0000e+01, 3.0800e+02, 1.6680e+03, 5.0560e+03, 1.0272e+04, 1.5452e+04, 1.8592e+04, 1.9098e+04, 1.7288e+04, 1.4032e+04, 1.0632e+04, 7.4720e+03, 4.7040e+03, 2.9440e+03, 1.7920e+03, 8.5200e+02, 4.5200e+02, 2.8400e+02, 8.4000e+01, 3.2000e+01, 3.2000e+01, 4.0000e+00, 2.0000e+00]),
np.array([2.0000e+00, 1.0600e+02, 9.5600e+02, 4.1440e+03, 1.1090e+04, 2.0914e+04, 3.0370e+04, 3.6100e+04, 3.6868e+04, 3.3670e+04, 2.7856e+04, 2.1040e+04, 1.5060e+04, 1.0076e+04, 6.0600e+03, 3.6720e+03, 2.1700e+03, 9.9000e+02, 5.1600e+02, 3.2000e+02, 9.0000e+01, 3.4000e+01, 3.4000e+01, 4.0000e+00, 2.0000e+00]),
np.array([2.2000e+01, 4.1600e+02, 2.7100e+03, 9.8480e+03, 2.3838e+04, 4.2400e+04, 5.9604e+04, 7.0116e+04, 7.1664e+04, 6.5472e+04, 5.4870e+04, 4.2336e+04, 3.0172e+04, 2.0608e+04, 1.3224e+04, 7.6400e+03, 4.5020e+03, 2.5920e+03, 1.1380e+03, 5.8400e+02, 3.5800e+02, 9.6000e+01, 3.6000e+01, 3.6000e+01, 4.0000e+00, 2.0000e+00]),
np.array([2.00000e+00, 1.28000e+02, 1.39200e+03, 7.16400e+03, 2.26920e+04, 5.04560e+04, 8.55180e+04, 1.17190e+05, 1.36218e+05, 1.39350e+05, 1.28124e+05, 1.07712e+05, 8.44100e+04, 6.15440e+04, 4.17800e+04, 2.74280e+04, 1.69740e+04, 9.46000e+03, 5.44000e+03, 3.06000e+03, 1.29600e+03, 6.56000e+02, 3.98000e+02, 1.02000e+02, 3.80000e+01, 3.80000e+01, 4.00000e+00, 2.00000e+00]), 
np.array([2.40000e+01, 5.46000e+02, 4.20800e+03, 1.79880e+04, 5.09840e+04, 1.05700e+05, 1.71740e+05, 2.30424e+05, 2.65468e+05, 2.70924e+05, 2.50680e+05, 2.12700e+05, 1.67424e+05, 1.24226e+05, 8.63760e+04, 5.62400e+04, 3.56800e+04, 2.13860e+04, 1.15360e+04, 6.49200e+03, 3.57600e+03, 1.46400e+03, 7.32000e+02, 4.40000e+02, 1.08000e+02, 4.00000e+01, 4.00000e+01, 4.00000e+00, 2.00000e+00]),
np.array([2.00000e+00, 1.52000e+02, 1.96000e+03, 1.17900e+04, 4.34960e+04, 1.12264e+05, 2.19510e+05, 3.44174e+05, 4.52864e+05, 5.18052e+05, 5.28298e+05, 4.89966e+05, 4.19666e+05, 3.33956e+05, 2.49020e+05, 1.76486e+05, 1.17810e+05, 7.39520e+04, 4.55320e+04, 2.65220e+04, 1.38840e+04, 7.66400e+03, 4.14200e+03, 1.64200e+03, 8.12000e+02, 4.84000e+02, 1.14000e+02, 4.20000e+01, 4.20000e+01, 4.00000e+00, 2.00000e+00]), 
np.array([2.600000e+01, 7.000000e+02, 6.296000e+03, 3.119200e+04, 1.020620e+05, 2.434720e+05, 4.525300e+05, 6.884080e+05, 8.907660e+05, 1.011416e+06, 1.031562e+06, 9.600400e+05, 8.263760e+05, 6.651480e+05, 5.019280e+05, 3.575200e+05, 2.436000e+05, 1.569080e+05, 9.534000e+04, 5.716000e+04, 3.244600e+04, 1.652000e+04, 8.962000e+03, 4.760000e+03, 1.830000e+03, 8.960000e+02, 5.300000e+02, 1.200000e+02, 4.400000e+01, 4.400000e+01, 4.000000e+00, 2.000000e+00]),
np.array([2.000000e+00, 1.780000e+02, 2.684000e+03, 1.863400e+04, 7.902400e+04, 2.337280e+05, 5.215480e+05, 9.279120e+05, 1.374292e+06, 1.753052e+06, 1.977616e+06, 2.014892e+06, 1.882970e+06, 1.630406e+06, 1.320878e+06, 1.009606e+06, 7.282380e+05, 4.985860e+05, 3.282300e+05, 2.048180e+05, 1.208520e+05, 7.074800e+04, 3.922400e+04, 1.946000e+04, 1.039200e+04, 5.432000e+03, 2.028000e+03, 9.840000e+02, 5.780000e+02, 1.260000e+02, 4.600000e+01, 4.600000e+01, 4.000000e+00, 2.000000e+00]),
np.array([2.800000e+01, 8.800000e+02, 9.132000e+03, 5.181000e+04, 1.934240e+05, 5.250320e+05, 1.105744e+06, 1.894396e+06, 2.740440e+06, 3.450352e+06, 3.871464e+06, 3.941132e+06, 3.692676e+06, 3.218484e+06, 2.626836e+06, 2.022856e+06, 1.479516e+06, 1.026212e+06, 6.785560e+05, 4.333000e+05, 2.627760e+05, 1.509600e+05, 8.648800e+04, 4.692400e+04, 2.272000e+04, 1.196000e+04, 6.160000e+03, 2.236000e+03, 1.076000e+03, 6.280000e+02, 1.320000e+02, 4.800000e+01, 4.800000e+01, 4.000000e+00, 2.000000e+00]),
np.array([2.000000e+00, 2.060000e+02, 3.590000e+03, 2.846800e+04, 1.372820e+05, 4.603280e+05, 1.160980e+06, 2.324960e+06, 3.852884e+06, 5.459976e+06, 6.794116e+06, 7.583656e+06, 7.717950e+06, 7.249322e+06, 6.349068e+06, 5.224520e+06, 4.057052e+06, 2.991352e+06, 2.105720e+06, 1.411000e+06, 9.044780e+05, 5.620060e+05, 3.321080e+05, 1.861600e+05, 1.045800e+05, 5.561600e+04, 2.631600e+04, 1.367200e+04, 6.946000e+03, 2.454000e+03, 1.172000e+03, 6.800000e+02, 1.380000e+02, 5.000000e+01, 5.000000e+01, 4.000000e+00, 2.000000e+00]),
np.array([3.0000000e+01, 1.0880000e+03, 1.2900000e+04, 8.2988000e+04, 3.5004200e+05, 1.0708320e+06, 2.5336280e+06, 4.8553040e+06, 7.8125400e+06, 1.0868816e+07, 1.3385116e+07, 1.4868552e+07, 1.5122818e+07, 1.4244624e+07, 1.2532550e+07, 1.0377960e+07, 8.1342520e+06, 6.0517600e+06, 4.2961120e+06, 2.9238080e+06, 1.8996980e+06, 1.1841440e+06, 7.1782600e+05, 4.1423200e+05, 2.2697200e+05, 1.2523200e+05, 6.5372000e+04, 3.0264000e+04, 1.5534000e+04, 7.7920000e+03, 2.6820000e+03, 1.2720000e+03, 7.3400000e+02, 1.4400000e+02, 5.2000000e+01, 5.2000000e+01, 4.0000000e+00, 2.0000000e+00]),
np.array([2.0000000e+00, 2.3600000e+02, 4.7060000e+03, 4.2250000e+04, 2.2958000e+05, 8.6489000e+05, 2.4445720e+06, 5.4690920e+06, 1.0080836e+07, 1.5802860e+07, 2.1622588e+07, 2.6377612e+07, 2.9177502e+07, 2.9657036e+07, 2.7998560e+07, 2.4752372e+07, 2.0618494e+07, 1.6279044e+07, 1.2234680e+07, 8.7675200e+06, 6.0186780e+06, 3.9745120e+06, 2.5114720e+06, 1.5261240e+06, 9.0453000e+05, 5.1066000e+05, 2.7394000e+05, 1.4866000e+05, 7.6266000e+04, 3.4580000e+04, 1.7552000e+04, 8.7000000e+03, 2.9200000e+03, 1.3760000e+03, 7.9000000e+02, 1.5000000e+02, 5.4000000e+01, 5.4000000e+01, 4.0000000e+00, 2.0000000e+00])
]

'''
Fit Deg Vs Ene as a gassain
'''
LEN = len(Deg)

def z(e,a,sigma):
    return a * np.exp(-((e + 1) ** 2) / (sigma ** 2))

for i in range(LEN):
    Deg[i] = Deg[i]/(np.sum(Deg[i]))

a_list = np.array([])
simga_list = np.array([])

a_error = np.array([])
simga_error = np.array([])

'''
Gas fit
'''

for i in range(1,LEN): #Plots for Deg Vs Ene
    para,error = curve_fit(z,Ene[i],Deg[i])
    print(para)
    
    a_list = np.append(para[0], a_list)
    simga_list = np.append(para[1], simga_list)

    a_error = np.append(np.sqrt(error[0][0]), a_error)
    simga_error = np.append(np.sqrt(error[1][1]), simga_error)

    print(f'a error = +/-{np.sqrt(error[0][0])}. sigma error = +/-{np.sqrt(error[1][1])})')
    
    dn = np.linspace(np.min(Ene[i]), np.max(Ene[i]), 1000)

    plt.figure()
    plt.title('Degeneracy vs Energy')
    plt.ylabel('Degeneracy as a %')
    plt.xlabel('Energy')
    plt.grid()
    plt.bar(Ene[i], Deg[i], label = f'Raw data for {i} spins')
    plt.plot(dn, z(dn, *para), color = 'red', label = 'Fit')
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)

plt.figure()
plt.title('Degeneracy vs Energy')
plt.ylabel('Degeneracy as a %')
plt.xlabel('Energy')
plt.grid()
for i in range(1,LEN):
    plt.bar(Ene[i], Deg[i], width=0.8, edgecolor='black')
plt.text(0.05, 0.95, "Spins of a Triangular system using 4–25 particles" , transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", fc="w"))
plt.show(block = True)

'''
Plots of a and sigma
+
Fitiing points for a and sigma
'''

allowed_points_min_sig = 5
allowed_points_max_sig = 16

allowed_points_min_a = 7
allowed_points_max_a = 17

if 2 == 2: #storing data points and crap
    a_list = a_list[::-1] #Inverting arrays
    simga_list = simga_list[::-1]
    a_error = a_error[::-1]
    simga_error = simga_error[::-1]

    LEN_data = len(a_list)
    dLEN_data = np.linspace(0, LEN_data, 1000)

    a_set = np.array([a_list[i] for i in range(allowed_points_min_a, allowed_points_max_a)])
    a_set_error = np.array([a_error[i] for i in range(allowed_points_min_a, allowed_points_max_a)])

    sigma_set = np.array([simga_list[i] for i in range(allowed_points_min_sig, allowed_points_max_sig)])
    sigma_set_error = np.array([simga_error[i] for i in range(allowed_points_min_sig, allowed_points_max_sig)])

    a_test = a_list[allowed_points_min_a:allowed_points_max_a]
    sigma_test = simga_list[allowed_points_min_sig:allowed_points_max_sig]

    Range_sig = np.array([i for i in range(allowed_points_min_sig, allowed_points_max_sig)])
    Range_a = np.array([i for i in range(allowed_points_min_a, allowed_points_max_a)])

def a_fit(x, a, b, c, d):
    return a * (x + b) ** c + d

def sigma_fit(x, a, b, c):
    return a * np.log(x + b) + c

a_para_fit, a_error_fit = curve_fit(
    a_fit, Range_a, a_test, p0=[1, 1, -1, 1], maxfev=10000
)

sigma_para_fit, sigma_error_fit = curve_fit(
    sigma_fit, Range_sig, sigma_test, p0=[1, 1, 1], maxfev=10000
)

a_eq = r"$a(n) = {:.3f}(n + {:.3f})^{{{:.3f}}}$".format(*a_para_fit)
sigma_eq = r"$\sigma(n) = {:.3f} \cdot \log(n + {:.3f})   {:.3f}$".format(*sigma_para_fit)

#Plot for a
plt.figure()
plt.title('Fit for max peak as a function of n plot')
plt.ylabel('peak')
plt.xlabel('n')
plt.errorbar(np.arange(LEN_data), a_list, yerr=a_error, fmt='o', capsize=3)
plt.errorbar(Range_a, a_set, yerr=a_set_error, fmt='o', color = 'green', capsize=3)
plt.plot(dLEN_data, a_fit(dLEN_data, *a_para_fit), color='red')
plt.text(0.05, 0.95, a_eq, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", fc="w"))
plt.grid()
plt.tight_layout()
plt.show(block=False)

#plot for sigma 
plt.figure()
plt.title('Sigma fit plot')
plt.ylabel('Sigma')
plt.xlabel('n')
plt.errorbar(np.arange(LEN_data), simga_list, yerr=simga_error, fmt='o', capsize=3)
plt.errorbar(Range_sig, sigma_set, yerr=sigma_set_error, fmt='o', color = 'green', capsize=3)
plt.plot(dLEN_data, sigma_fit(dLEN_data, *sigma_para_fit), color='red')
plt.text(0.05, 0.95, sigma_eq, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", fc="w"))
plt.grid()
plt.tight_layout()
plt.show(block=False)

#fitting the fit plots
for i in range(1,LEN): #Plots for Deg Vs Ene
    para,error = curve_fit(z,Ene[i],Deg[i])
    
    dn = np.linspace(np.min(Ene[i]), np.max(Ene[i]), 1000)

    A = a_fit(i, *a_para_fit)
    S = sigma_fit(i, *sigma_para_fit)

    plt.figure()
    plt.title('Degeneracy vs Energy')
    plt.ylabel('Degeneracy as a %')
    plt.xlabel('Energy')
    plt.grid()
    plt.bar(Ene[i], Deg[i], label = f'Raw data for {i} spins')
    plt.plot(dn, z(dn, *para), color = 'red', label = 'Fit')
    plt.plot(dn, z(dn,A,S), color = 'green', label = 'Fitting the original fit')
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)
