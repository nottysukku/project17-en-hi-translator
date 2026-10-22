"""Design the 16 -> 48 kHz interpolation filter of xl_resample.c and print it as C.

    python resample_design.py

3x upsampling: zeros between samples, then this low-pass at 48 kHz (passband to 7 kHz, stopband
from 9 kHz: the images of the 0-8 kHz speech band), split into 3 phases of 32 taps.
"""
import numpy as np
from scipy import signal

UP, TAPS = 3, 96
h = signal.firwin(TAPS, 8000, width=2000, window=("kaiser", 7.0), fs=48000) * UP
w, resp = signal.freqz(h / UP, worN=8192, fs=48000)
db = 20 * np.log10(np.maximum(np.abs(resp), 1e-12))
print("/* passband ripple 0-7 kHz: %.2f dB; worst stopband >= 9 kHz: %.1f dB */" %
      (db[w <= 7000].max() - db[w <= 7000].min(), db[w >= 9000].max()))
print("static const float taps[XL_RESAMPLE_PHASES][XL_RESAMPLE_TAPS] = {")
for p in range(UP):
    ph = h[p::UP][::-1]  # phase p, reversed: taps[p][k] multiplies history sample k (oldest first)
    print("    {" + ", ".join("%.9gf" % v for v in ph) + "},")
print("};")
