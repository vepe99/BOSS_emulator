#%%

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

# BOSS spec-lite file: wavelength grid and LSF
h = fits.open('spec-015018-59164-4551583147.fits')
wave = 10 ** h[1].data['LOGLAM']
wdisp = h[1].data['WDISP'].astype(float)

# repair isolated bad WDISP pixels
bad = ~np.isfinite(wdisp) | (np.abs(np.log(np.clip(wdisp, 1e-6, None) / np.nanmedian(wdisp))) > np.log(2))
wdisp[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), wdisp[~bad])

# calibrated LSF sigma [A]: WDISP is in units of 1e-4 dex pixels, kappa = 1.1
sigma_lsf = 1.1 * np.log(10) * 1e-4 * wdisp * wave

plt.figure()
plt.plot(wave, sigma_lsf)
plt.xlabel('wavelength [A]')
plt.ylabel('LSF sigma [A]')
plt.show()

# mock model spectrum at R = lambda/FWHM = 10000, with some absorption lines
np.random.seed(0)
R_in = 10000 * 2.3548                       # R_sigma = lambda / sigma
wm = np.exp(np.arange(np.log(3500), np.log(10500), 1 / (5 * R_in)))
fm = np.ones_like(wm)
for lc, depth in zip(np.random.uniform(3600, 10400, 600), np.random.uniform(0.05, 0.9, 600)):
    fm *= 1 - depth * np.exp(-0.5 * ((wm - lc) / (lc / R_in)) ** 2)

# kernel: remove the model's resolution, add pixel integration
sigma_k = np.sqrt(np.clip(sigma_lsf ** 2 + np.gradient(wave) ** 2 / 12 - (wave / R_in) ** 2, (0.1 * sigma_lsf) ** 2, None))

# variable-width Gaussian convolution: stretch lambda so sigma_k is constant, FFT-convolve, sample at BOSS pixels
sig = np.interp(wm, wave, sigma_k)
cdf = np.cumsum(np.gradient(wm) / sig)
cdf /= cdf[-1]
x_per_sigma = np.median(np.gradient(cdf) / (np.gradient(wm) / sig))
nx = int(2 ** np.ceil(np.log2(2 / x_per_sigma)))
lam = np.interp(np.linspace(0, 1, nx), cdf, wm)
f = np.fft.irfft(np.fft.rfft(np.interp(lam, wm, fm)) * np.exp(-2 * np.pi ** 2 * x_per_sigma ** 2 * np.fft.rfftfreq(nx, 1 / nx) ** 2), n=nx)
flux_boss = np.interp(wave, lam, f)

plt.figure()
plt.plot(wm, fm, color='0.6', lw=0.8, label='model (R=5000)')
plt.plot(wave, flux_boss, 'o-', ms=3, label='smoothed to BOSS')
plt.xlim(5100, 5300)
plt.xlabel('wavelength [A]')
plt.ylabel('flux')
plt.legend()
plt.show()

# %%
