import emcee
#import triangle
import corner
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator


# Define the probability function as likelihood * prior.
def lnprior(theta):
    a1, a2, a3, c, s = theta
    if -5.0 < a1 < 5.0 and -5.0 < a2 < 5.0 and -5.0 < a3 < 5.0 and -5.0 < c < 5.0 and -5.0 < s < 5.0:
        return 0.0
    return -np.inf

def lnlike(theta, x, merr, serr, rerr, y, yerr):
    a1, a2, a3, c, s = theta
    model = a1 * x[:,0] + a2 * x[:,1] + a3 * x[:,2] + c
    inv_sigma2 = 1.0/(yerr**2 + s**2 + (a1*merr)**2 + (a2*serr)**2 + (a3*rerr)**2)
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprob(theta, x, merr, serr, rerr, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, merr, serr, rerr, y, yerr)

def regression(x, merr, serr, rerr, y, yerr):# Define data HERE!
    N    = 50
    #x    = 
    #y    = 
    #yerr = 

    # Set up the sampler.
    ndim, nwalkers = 5, 100
    pos = emcee.utils.sample_ball([0,0,0,-2,1],[0.1,0.1,0.1,0.1,0.1],size=nwalkers)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, merr, serr, rerr, y, yerr))

    # Clear and run the production chain.
    print("Running MCMC...")
    sampler.run_mcmc(pos, 1000, rstate0=np.random.get_state())
    print("Done.")

    # Plotting the parameters versus number of steps to examine convergence
    pl.clf()
    fig, axes = pl.subplots(5, 1, sharex=True, figsize=(8, 9))
    axes[0].plot(sampler.chain[:, :, 0].T, color="k", alpha=0.4)
    axes[0].yaxis.set_major_locator(MaxNLocator(5))
    #axes[0].axhline(m_true, color="#888888", lw=2)
    axes[0].set_ylabel(r'Mass [log(M$_*$/M$_{sol}$)]')

    axes[1].plot(sampler.chain[:, :, 1].T, color="k", alpha=0.4)
    axes[1].yaxis.set_major_locator(MaxNLocator(5))
    #axes[1].axhline(b_true, color="#888888", lw=2)
    axes[1].set_ylabel(r'sSFR (yr$^{-1}$)')

    axes[2].plot(np.exp(sampler.chain[:, :, 2]).T, color="k", alpha=0.4)
    axes[2].yaxis.set_major_locator(MaxNLocator(5))
    #axes[2].axhline(f_true, color="#888888", lw=2)
    axes[2].set_ylabel(r'$\Delta$log([O III]/H$\beta$)')

    axes[3].plot(np.exp(sampler.chain[:, :, 3]).T, color="k", alpha=0.4)
    axes[3].yaxis.set_major_locator(MaxNLocator(5))
    #axes[3].axhline(f_true, color="#888888", lw=2)
    axes[3].set_ylabel("$c$")

    axes[4].plot(np.exp(sampler.chain[:, :, 4]).T, color="k", alpha=0.4)
    axes[4].yaxis.set_major_locator(MaxNLocator(5))
    #axes[4].axhline(f_true, color="#888888", lw=2)
    axes[4].set_ylabel("$s$")
    axes[4].set_xlabel("step number")

    fig.tight_layout(h_pad=0.0)
    fig.savefig("../other_plots/line-time.png")

    # Make the triangle plot.
    burnin = 200  # pick burnin step size
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

    fig = corner.corner(samples, labels=[r'Mass [log(M$_*$/M$_{sol}$)]', r'sSFR (yr$^{-1}$)', r'$\Delta$log([O III]/H$\beta$)', "C", "s"])#, range=([-0.2, -0.185], [-0.06, 0], [1.3, 1.7], [-2.3, -1.9], 0.99))
    fig.savefig("../other_plots/line-triangle.png")
    

    # Compute the quantiles.
    a1_mcmc, a2_mcmc, a3_mcmc, c_mcmc, s_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                                    zip(*np.percentile(samples, [16, 50, 84],
                                                                       axis=0)))

    return a1_mcmc, a2_mcmc, a3_mcmc, c_mcmc, s_mcmc



