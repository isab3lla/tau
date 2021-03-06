## isabella, dec 2017 
## routines to compute reionization quantities 

#########################################################
# ***   we follow Kuhlen & Faucher-Giguere (2012)   *** #
#########################################################

## routines available:

## 1) ionisation rate in number of photons yr^-1 Mpc^-3
##  as in eq. 2
## N_ion(Muv,Phi,f_esc,Muv_max_flag,Muv_max)
## Muv, Phi -----------------> UV luminosity function
## f_esc --------------------> escape function
## Muv_max ------------------> faint end cut magnitude

## 2) recombination time in years, as in eq. 3
## t_rec(z,C_HII=3.0,T0=2.e4)
## C_HII --------------------> the clumping factor
## T0    --------------------> IGM average temperature [K]

## 3) ionised fraction Q (eq. 1)
## Q_sol(z_sample,Nion_sample,h,Omega_M,Omega_b,
##        z1=0.0,z0=20.5,Q0=1e-13,n=1,verb=True)
## z_sample,Nion_sample -----> z and N_dot available
## h,Omega_M,Omega_b --------> cosmology used
## z1 -----------------------> up to which low z to look for Q
## z0, Q0 -------------------> initial conditions
## n ------------------------> spline to interpolate N_dot
## RETURN [z,Q] array

## 4) value of optical depth for CMB  (eq. 4)  
## tau(zt,z_sample,Nion_sample,h,Omega_M,Omega_b,
##      z1=4.5,z0=20.5,Q0=1e-13,n=1,verb=True)
## zt -----------------------> chosen z for tau(z) (or array)
## z_sample,Nion_sample -----> z and N_dot available
## h,Omega_M,Omega_b --------> cosmology used
## z1 -----------------------> up to which low z to look for Q
## z0, Q0 -------------------> initial conditions
## n ------------------------> spline to interpolate N_dot
## RETURN single value tau(zt) or array, according to zt input
## zt HAS TO HAVE A SIZE!

## 5) array of optical depth -- NOW DEPRECATED
## tau_ar(z_ar,z_sample,Nion_sample,h,Omega_M,Omega_b,
##			z1=0.0,z0=20.5,Q0=1e-13,n=1)
## z_ar ---------------------> chosen z-array
## z_sample,Nion_sample -----> z and N_dot available
## h,Omega_M,Omega_b --------> cosmology used
## z1 -----------------------> up to which low z to look for Q
## z0, Q0 -------------------> initial conditions
## n ------------------------> spline to interpolate N_dot
## RETURN 1 array: tau_ar


## 6) redshift of reionization and its duration
## z_re corresponds to z when Q = 0.5
## First delta_re definition is  eq. 3 of Sharma+2018, (1712.06619) 
## but also Monsalve+ 2017
## Second definition from George+ 2015 (kSZ effect on CMB)
## reion_time(zlist,Qlist,increment=1.e-6)
## zlist --------------------> z-array, same length as Q
## Qlist --------------------> Q neutral fraction array
## increment ----------------> for computing the derivative
## RETURN 3 scalars:
## 1. z_re, redshift of reionization
## 2. deltaz_re_1, duration of reionization 
## 3. deltaz_re_2, duration of reionization (George+ 2015)


## 6b) redshift of reionization and its duration
## Following definitions of the Planck XLII 2016 paper
## reion_time_Planck(zlist,Qlist)
## zlist --------------------> z-array, same length as Q
## Qlist --------------------> Q neutral fraction array
## RETURN 2 scalars: z_re and Delta_z, i.e. duration of EoR


## 7) redshift dependent f_esc
## ## a) Sharma+2018
## ## fesc_Sharma(z,f_esc7)
## ## z -----------------------> redshift
## ## f_esc7 ------------------> f_esc at z=7
## ## RETURN 1 scalar: f_esc
##
## ## b) Faisst 2016
## ## fesc_Faisst(z)
## ## z -----------------------> redshift
## ## RETURN 3 arrays: f_esc, f_esc_up, f_esc_down (1sigma)
## ## for the actual Faisst-like parametrization of f_esc:
## ## Faisst_param(z,fesc0,alpha)
##
## ## c) exercise parametrization
## ## fesc_lin(z,f_esc6)
## ## z -----------------------> redshift
## ## f_esc6 ------------------> f_esc at z=6
## ## fesc_quad(z,pl,pq)
## ## RETURN 1 scalar: f_esc (or array, 
## ##        according to input z)




## routines that work only with my LF format

## a) Loading the luminosity function
## LF(z,csm,cosmology,dir)
## z ------------------------> redshift
## csm ----------------------> cosmological model (index)
## cosmology ----------------> array with file names
## dir ----------------------> path of file directory
## RETURN 2 arrays: Muv and Phi(Muv)

## b) Computing the ionisation rate
## N_ion_z(z,csm,f_esc,Muv_max_flag,Muv_max,cosmology,dir)
## z ------------------------> redshift
## csm ----------------------> cosmological model (index)
## f_esc --------------------> escape fraction
## Muv_max_flag -------------> if false, I take all Muv in file
## Muv_max ------------------> faint end cut magnitude
## cosmology ----------------> array with file names
## dir ----------------------> path of file directory
## RETURN 1 array: N_ion_z, same length as the input z
## i.e.
## ionisation rate in number of photons yr^-1 Mpc^-3
## a value for each input redshift

## c) Computing the ionisation rate with z-dependent f_esc
## N_ion_z_fz(z,csm,recipe,f_esc_prm,Muv_max_flag,Muv_max,cosmology,dir)
## z ------------------------> redshift
## csm ----------------------> cosmological model (index)
## recipe -------------------> which f_esc(z) parametrization
## f_esc_prm ----------------> escape fraction parameter
## Muv_max_flag -------------> if false, I take all Muv in file
## Muv_max ------------------> faint end cut magnitude
## cosmology ----------------> array with file names
## dir ----------------------> path of file directory
## RETURN 1 array: N_ion_z, same length as the input z
## i.e.
## ionisation rate in number of photons yr^-1 Mpc^-3
## a value for each input redshift




#########################################################

import numpy as np
import sys
import time
from scipy.integrate import simps
from scipy import interpolate

#########################################################
####################   CONSTANTS  #######################
#########################################################

## gravitational constant ##
G = 4.302e-9 # Mpc (km/s)^2 Msun^-1

## proton mass ##
mp = 1.6726e-27 #kg
mp = mp / (1.989e30) # Msun

## Thomson cross section ##
sigmaT = 6.65e-25  # cm^2
sigmaT = sigmaT / (3.086e24)**2  # Mpc^2

## speed of light ##
c = 3.0e5  # km/s

#########################################################
################   USEFUL FUNCTIONS   ###################
#########################################################

## mean comoving hydrogen number density ##
def n_H(h,Omega_b):
	rho_crit = (3.e4*h**2)/(8*np.pi*G)   # Mpc^-3 Msun
	return 0.75 * Omega_b * rho_crit /mp  # Mpc^-3

## Hubble parameter in [km /Mpc /s] ##
def Hz(z,h,Omega_M):
	a = (Omega_M*(1.+z)**3+1.-Omega_M)**0.5
	return h*100.*a

## to go from time to redshift ##
def dt_dz(z,h,Omega_M): # yr
	Hz_yr = Hz(z,h,Omega_M)*3.15576e7/(3.086e19) 
	return -1./(1.+z)/Hz_yr

## array with decraising (increasing) spacing between elements
## useful for tau integrals 
## from z to inf (from 0 to z) if denser True (False)
def spaced_list(start,stop,how_close,spacing,denser):
	a = np.array([start])

	while a[-1] < (stop-how_close):

		if denser:
			increment = float((stop - a[-1]))/spacing
		else:
			increment = float((a[-1] - start + how_close))/spacing

		a = np.append(a,(increment+a[-1]))

	return a



#########################################################
##################   ionisation rate   ################## 
#########################################################
## 1) ionisation rate in number of photons yr^-1 Mpc^-3
##  as in eq. 2
## N_ion(Muv,Phi,f_esc,Muv_max_flag,Muv_max)
## Muv, Phi -----------------> UV luminosity function
## f_esc --------------------> escape function
## Muv_max ------------------> faint end cut magnitude

# hydrogen ionising photon luminosity (eq. 6)
def g_ion(Muv):
	ABlum = 10**(0.4*(51.63-Muv))
	g = 2.0e25*ABlum # s^-1
	return g*31557600 # yr^-1

# photons per unit time per unit volume:
def N_ion(Muv,Phi,f_esc,Muv_max_flag,Muv_max):	# (eq. 2)

	print('Computing N_ion . .')
	print('Escape fraction = %.2f'%(f_esc))
	
	## sanity check
	if len(Muv)!=len(Phi):
		print('\nLUMINOSITY FUNCTION AND MAGNITUDES')
		print('HAVE NOT SAME DIMENSION!!\n')
		sys.exit()
	
	if Muv_max_flag:
		print('we consider magnitudes Muv <', Muv_max)
		index=np.where(Muv<Muv_max)[0]
		Muv=Muv[index]; Phi=Phi[index]
		del index
	else:
		print('Max magnitude --> Muv = ',np.max(Muv))	
	print(' ')
	
	integrand = Phi*g_ion(Muv)
	#integration with Simpson rule
	Isimps = simps(integrand, Muv)
	
	return f_esc * Isimps # yr^-1 Mpc^-3



#########################################################
##############   recombination timescale   ############## 
#########################################################
## 2) recombination time in years, as in eq. 3
## t_rec(z,C_HII=3.0,T0=2.e4)
## C_HII --------------------> the clumping factor
## T0    --------------------> IGM average temperature in K


def t_rec(z,C_HII=3.0,T0=2.e4): # (eq. 3) 
	clump = 3.0/C_HII
	temp  = (T0/2.e4)**(0.7)
	red   = ((1.+z)/7)**(-3)
	return 9.3e8 * clump * temp * red   # yr
	
	
#########################################################
###############   Q: ionising fraction   ################
#########################################################
## i.e. solving eq. 1 for Q

## 3) ionised fraction Q (eq. 1)
## Q_sol(z_sample,Nion_sample,h,Omega_M,
##        z1=4.5,z0=20.0,Q0=1e-13,n=1,verb=True)
## z_sample,Nion_sample -----> z and N_dot available
## h,Omega_M,Omega_b --------> cosmology used
## z1 -----------------------> up to which low z to look for Q
## z0, Q0 -------------------> initial conditions
## n ------------------------> spline to interpolate N_dot

###############   interpolating N_ion   ################

## creating the N_dot function in redshift
## (this is just because e.g. z = 6-10 not enough to  
## get a reasonable ionised fraction Q
## so we extend it to z = 0-20 )

def z_N_ion(z_sample, Nion_sample):
	
	z_sample = np.append(z_sample,21.0)
	z_sample = np.append(-1.0,z_sample)
	
	Nion_sample = np.append(Nion_sample,0.0)
	Nion_sample = np.append(np.max(Nion_sample),Nion_sample)

	return z_sample, Nion_sample


def Q_analyt(z,z_sample,Nion_sample,h,Omega_M,Omega_b,Q0,n):

	## creating the N_dot function
	## (interpolating N_ion with order n spline)
	z_sample,Nion_sample = z_N_ion(z_sample,Nion_sample)
	N_ion_func = interpolate.interp1d(z_sample,Nion_sample,n)

	## the integrals to solve the ODE
	R = dt_dz(z,h,Omega_M)/t_rec(z)
	P = N_ion_func(z)/n_H(h,Omega_b)*dt_dz(z,h,Omega_M)
	I_exp = simps(R, z)
	exp_term = np.exp(-I_exp)
	int_term = simps(np.exp(I_exp)*P,z)
	
	return exp_term*(int_term+Q0)


## actual solver
def Q_sol(z_sample,Nion_sample,h,Omega_M,Omega_b,z1=4.5,z0=20.5,Q0=1e-13,n=1,verb=True):
	
	arg = (h,Omega_M,Omega_b,Q0,n)
			
	## z-step
	dt = (z0-z1)/1.e2
	
	start_a = time.time()

	eps=z0
	z_ar = np.empty([0]); sol_ar = np.empty([0])
	
	## sanity check
	if len(z_sample)!=len(Nion_sample):
		print('\nREDSHIFTS AND N_DOT')
		print('HAVE NOT SAME DIMENSION!!\n')
		sys.exit()
	if verb:
		print('\n\nSolving for Q . .')	
		print('  %d N_dot points available'%(len(z_sample)))
		print('  at redshifts --> ',z_sample)
	
	while eps > z1:
	#print eps
		eps-=dt
	
		z_int = np.linspace(z0, eps, 10)

		z_ar=np.append(z_ar,eps)
		Q_int = Q_analyt(z_int,z_sample,Nion_sample,*arg)
		sol_ar=np.append(sol_ar,Q_int)
		del Q_int	

	end_a = time.time()
	ts = end_a - start_a
	if verb:
		print('elapsed time for solving: %.3f s\n'%(ts))

   	## deleting all Q > 1 data
	index = np.where(sol_ar>1.)[0]
	sol_ar[index] = 1.0
	del index

	return z_ar, sol_ar
	

#########################################################
#################   optical depth tau    ################
#########################################################
## 4) value of optical depth for CMB  (eq. 4)  
## tau(zt,z_sample,Nion_sample,h,Omega_M,
##      z1=0.0,z0=20.5,Q0=1e-13,n=1,verb=True)
## zt -----------------------> chosen z for tau(z)
## z_sample,Nion_sample -----> z and N_dot available
## h,Omega_M,Omega_b --------> cosmology used
## z1 -----------------------> up to which low z to look for Q
## z0, Q0 -------------------> initial conditions
## n ------------------------> spline to interpolate N_dot

## free electron number ##
def fe(ze): # see eq. 4
	eta = (np.sign(4-ze)+3)/2
	return 1.+(eta*0.25)/(4.*0.75)

def Q_function(z_sample,Nion_sample,h,Omega_M,Omega_b,z1=0.0,z0=20.5,Q0=1e-13,n=1,verb=True):
		
	args = z_sample,Nion_sample,h,Omega_M,Omega_b,z1,z0,Q0,n,verb

	## creating the Q function
	## (interpolating Q_solved with order 1 spline)
	z_ar, sol_a = Q_sol(*args)
	Q_func = interpolate.interp1d(z_ar, sol_a,1)

	return Q_func

def tau_integrand(z_tau,z_sample,Nion_sample,h,Omega_M,Omega_b,z1,z0,Q0,n,verb):

	args = z_sample,Nion_sample,h,Omega_M,Omega_b,z1,z0,Q0,n,verb

	Q_func = Q_function(*args)

	int1 = c*(1+z_tau)**2 / Hz(z_tau,h,Omega_M) 
	int2 = Q_func(z_tau)*sigmaT*n_H(h,Omega_b)*fe(z_tau)

	return int1*int2


# Thompson optical depth for CMB  (eq. 4)
def tau(zt,z_sample,Nion_sample,h,Omega_M,Omega_b,z1=0.0,z0=20.5,Q0=1e-13,n=1,verb=True,stand=True):

	## for stand=True meaning the convential way for the integral (from 0 to z)
	## otherwise stand=False integral solved from z to infinity

	print('\nSolving for tau . . ')
	print('  %d N_dot points available'%(len(z_sample)))
	print('  at redshifts --> ',z_sample,'\n')

	# if zt just one value, I have problem with the len() function
	if hasattr(zt, '__len__'):
		pass
	else:
		zt = [zt]

	args = z_sample,Nion_sample,h,Omega_M,Omega_b,z1,z0,Q0,n,verb

	if len(zt)<2:

		## sanity check
		if zt[0]>20.:
			print('\n redshift cannot be higher than 20!\n')
			sys.exit()

		nsteps = 500

		if stand:
			z_tau = np.linspace(z1,zt,nsteps)
		else:
			z_tau = np.linspace(zt,z0-0.6,nsteps)

		integrand = tau_integrand(z_tau,*args)
		if verb:
			print('Solving for tau at z =', zt)

		#integration with Simpson rule
		Isimps = simps(integrand, z_tau)
		tau_final = Isimps

	else:
		verb = False
		## sanity check
		if zt[0]>min(zt):
			print('\n redshift array must be ordered\n')
			sys.exit()

		if stand:
			z_tau = spaced_list(z1,zt[-1],0.6,200,denser=False)
		else:
			z_tau = spaced_list(zt[1],z0,0.6,200,denser=True)

		integrand = tau_integrand(z_tau,*args)

		tau_tmp   = np.zeros(len(z_tau))
		tau_final = np.zeros(len(zt))

		for i in range(len(z_tau)-1):

			if verb:
				print('Solving for tau at z =', z_tau[i])
		
			#integration with Simpson rule
			if stand:
				Isimps = simps(integrand[0:i+1], z_tau[0:i+1])
			else:
				Isimps = simps(integrand[i:-1], z_tau[i:-1])

			tau_tmp[i] = Isimps

			del Isimps

		tau_final = np.interp(zt,z_tau,tau_tmp)
	
	return tau_final
	



#########################################################
###############   optical depth tau array   #############
#########################################################
## 5) array of optical depth
## tau_ar(z_ar,z_sample,Nion_sample,h,Omega_M,
##			z1=0.0,z0=20.5,Q0=1e-13,n=1)
## z_ar ---------------------> chosen z-array
## z_sample,Nion_sample -----> z and N_dot available
## h,Omega_M,Omega_b --------> cosmology used
## z1 -----------------------> up to which low z to look for Q
## z0, Q0 -------------------> initial conditions
## n ------------------------> spline to interpolate N_dot

## THIS FUNCTION IS NOW DEPRECATED
# def tau_ar(z_ar,z_sample,Nion_sample,h,Omega_M,Omega_b,z1=0.0,z0=20.5,Q0=1e-13,n=1):

# 	verb = False
# 	args = z_sample,Nion_sample,h,Omega_M,Omega_b,z1,z0,Q0,n,verb
	
# 	print('\nSolving for tau . . ')
# 	print('  %d N_dot points available'%(len(z_sample)))
# 	print('  at redshifts --> ',z_sample,'\n')
	
# 	tau_ar = np.zeros(len(z_ar))
# 	for i in range(len(z_ar)):
# 		tau_ar[i] = tau(z_ar[i],*args)

# 	return tau_ar



#########################################################
###############   reionisation timescales   #############
#########################################################
## 6) z_re corresponds to z when Q = 0.5
## First delta_re definition is  eq. 3 of Sharma+2018, (1712.06619) 
## but also Monsalve+ 2017
## Second definition from George+ 2015 (kSZ effect on CMB)
## reion_time(zlist,Qlist,increment=1.e-6)
## zlist --------------------> z-array, same length as Q
## Qlist --------------------> Q neutral fraction array
## increment ----------------> for computing the derivative

def reion_time(zlist,Qlist,increment=1.e-6):
	Q_for_interp = np.array([0.2,0.5-increment,0.5,0.5+increment,0.99])

	z_ar_tmp = np.interp(Q_for_interp,Qlist,zlist)

	Qprime = (2*increment)/(z_ar_tmp[3]-z_ar_tmp[1])
	deltaz_re1 = -1./Qprime # eq. 3 of Sharma+2018

	deltaz_re2 = z_ar_tmp[0] - z_ar_tmp[4]

	return z_ar_tmp[2],deltaz_re1,deltaz_re2


## 6b) z_re corresponds to z when Q = 0.5*f
## Following definitions of the Planck XLII 2016 paper
## reion_time_Planck(zlist,Qlist)
## zlist --------------------> z-array, same length as Q
## Qlist --------------------> Q neutral fraction array

def reion_time_Planck(zlist,Qlist):

	# taking into account electrons injected into the IGM 
	# by first Helium reionization
	f = fe(8)

	Q_for_interp = np.array([0.1*f,0.5*f,0.99*f])
	z_ar_tmp = np.interp(Q_for_interp,Qlist,zlist)

	deltaz_re = z_ar_tmp[0] - z_ar_tmp[2]

	return z_ar_tmp[1],deltaz_re



#########################################################
##############   redshift dependent f_esc   #############
#########################################################
## 7) 


## ## a) Sharma+2018
## ## z -----------------------> redshift
## ## f_esc7 ------------------> f_esc at z=7

def fesc_Sharma(z,f_esc7):
	return f_esc7*(1.+z)/8.


## ## b) Faisst 2016
## ## fesc_Faisst(z)
## ## z -----------------------> redshift

## the actual f_esc parametrization 
def Faisst_param(z,fesc0,alpha):
	return fesc0*((1.0+z)/3.0)**alpha

def fesc_Faisst(z):

	fesc0 = np.array([0.023,0.028,0.018])
	alpha = np.array([1.17,1.19,1.15])

	f_esc      = Faisst_param(z,fesc0[0],alpha[0])
	f_esc_up   = Faisst_param(z,fesc0[1],alpha[1])
	f_esc_down = Faisst_param(z,fesc0[2],alpha[2])

	return f_esc, f_esc_up, f_esc_down


## ## c) exercise parametrization
## ## fesc_lin(z,f_esc6)
## ## z -----------------------> redshift
## ## f_esc6 ------------------> f_esc at z=6
## ## fesc_quad(z,pl,pq)

def fesc_lin(z,f_esc6):
	return f_esc6*(1.+z)/7.

def fesc_quad(z,params):
	return params[0]*(1.+z)/1000 + params[1]*(1.+z)**2/100

#########################################################
#########   FUNCTIONS SPECIFIC FOR MY LF FORMAT   #######
#########################################################


## a) Loading the luminosity function
## LF(z,csm,cosmology,dir)
## z ------------------------> redshift
## csm ----------------------> cosmological model (index)
## cosmology ----------------> array with file names
## dir ----------------------> path of file directory

def LF(z,csm,cosmology,dir):
	file = cosmology[csm]+str(z)+'.dat'
	Muv,Phi = np.loadtxt(dir+file,unpack=True)
	
	# verbose
	print('\nLoading luminosity function:')
	print('-- ',file[:-4],' --')
	#print '\n   how much sampled: ', len(Phi), len(Phi)
	print('   UV magnitude interval:')
	print('  ',np.min(Muv),'    ',np.max(Muv),'\n')
	 
	return [Muv,Phi]



## b) Computing the ionisation rate
## N_ion_z(z,csm,f_esc,Muv_max_flag,Muv_max,cosmology,dir)
## z ------------------------> redshift
## csm ----------------------> cosmological model (index)
## f_esc --------------------> escape fraction 
## Muv_max_flag -------------> if false, I take all Muv in file
## Muv_max ------------------> faint end cut magnitude
## cosmology ----------------> array with file names
## dir ----------------------> path of file directory


def N_ion_z(z,csm,f_esc,Muv_max_flag,Muv_max,cosmology,dir):
	N_ion_z = np.zeros(len(z))
	for i in range(len(z)):
		Muv, Phi = LF(z[i],csm,cosmology,dir)
		N_ion_z[i]= N_ion(Muv,Phi,f_esc,Muv_max_flag,Muv_max)
		del Muv, Phi
	return N_ion_z


## c) Computing the ionisation rate with z-dependent f_esc
## N_ion_z_fz(z,csm,recipe,f_esc_prm,Muv_max_flag,Muv_max,cosmology,dir)
## z ------------------------> redshift
## csm ----------------------> cosmological model (index)
## recipe -------------------> which f_esc(z) parametrization
## f_esc_prm ----------------> escape fraction parameter
## Muv_max_flag -------------> if false, I take all Muv in file
## Muv_max ------------------> faint end cut magnitude
## cosmology ----------------> array with file names
## dir ----------------------> path of file directory


def N_ion_z_fz(z,csm,recipe,f_esc_prm,Muv_max_flag,Muv_max,cosmology,dir):
	N_ion_z = np.zeros(len(z))
	fescz   = np.zeros(len(z))

	if recipe=='sharma':
		for i in range(len(z)):
			fescz[i] = fesc_Sharma(z[i],f_esc_prm)
	elif recipe=='lin':
		for i in range(len(z)):
			fescz[i] = fesc_lin(z[i],f_esc_prm)
	elif recipe=='quad':
		for i in range(len(z)):
			fescz[i] = fesc_quad(z[i],f_esc_prm)
	else:
		print('specify an existing patametrization of the escape fraction!')
		print('available ones: sharma, lin, quad')
		sys.exit()

	for i in range(len(z)):
		Muv, Phi = LF(z[i],csm,cosmology,dir)
		N_ion_z[i]= N_ion(Muv,Phi,fescz[i],Muv_max_flag,Muv_max)
		del Muv, Phi
	return N_ion_z



