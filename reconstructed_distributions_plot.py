"""
reconstructed_distributions_plot.py

Purpose
-------
Plot the reconstructed (imputed) reading distributions for sensors with missing or
corrupted data using simple Gaussian (Normal) models.

What this plot shows
--------------------
- For each imputed sensor (e.g., S204, S209), we assume its reading follows:
      X ~ Normal(mu, sd)
  where:
    mu = reconstructed mean reading
    sd = reconstructed standard deviation (uncertainty/spread)

- The script computes the Normal probability density function (PDF) for readings
  from 0 to 200 and overlays the curves in a single chart for comparison.

Decision guide (threshold line)
-------------------------------
- A vertical dashed line at reading = 145 is drawn as an anomaly threshold.
  This helps visually assess how much probability mass lies above the threshold
  for each reconstructed sensor distribution.

Assumptions / notes
-------------------
- The reconstructed distributions are modeled as Normal for visualization and
  interpretability. In real deployments, the true distribution could be skewed,
  multimodal, or otherwise non-Gaussian depending on environmental context.
- The x-range is fixed to 0..200 for consistent comparison; adjust if your sensors
  operate on a different scale.
- `sd` must be > 0. If sd is extremely small, the PDF peak becomes very tall and
  narrow; if sd is large, the curve becomes flatter and wider.

Dependencies
------------
- matplotlib
- math (standard library)

Outputs
-------
- Displays an interactive Matplotlib figure window with one curve per sensor.
"""
import matplotlib.pyplot as plt
import math

def normal_pdf(x, mu, sd):
    return (1/(sd*math.sqrt(2*math.pi))) * math.exp(-0.5*((x-mu)/sd)**2)

dists = {
    "S204": (151.40, 3.46),
    "S209": (100.02, 44.18)
}

xs = [i for i in range(0, 201)]  # plot 0..200
plt.figure()
for name, (mu, sd) in dists.items():
    ys = [normal_pdf(x, mu, sd) for x in xs]
    plt.plot(xs, ys, label=f"{name}: Normal({mu:.1f}, {sd:.1f})")

plt.axvline(145, linestyle="--")
plt.xlabel("Reading")
plt.ylabel("Probability density")
plt.title("Reconstructed distributions for missing/corrupted sensors")
plt.legend()
plt.show()
