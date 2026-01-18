"""
reading_vs_confidence_plot.py

Purpose
-------
Visualize sensor readings vs. signal confidence for:
1) observed sensor points from Table 1 and
2) imputed (reconstructed) sensor estimates for missing sensors.

What this plot shows
--------------------
- Each observed sensor (S201, S202, ...) is plotted as a scatter point:
    x-axis: reading value
    y-axis: signal confidence (0 to 1)
  Labels next to points show the sensor ID.

- Each imputed sensor (e.g., S204, S209) is plotted at its estimated mean:
    x = mu (imputed mean reading)
    y = conf (imputed confidence)
  An uncertainty bar is drawn to represent +/- 1 standard deviation around the mean
  (implemented here as a horizontal error bar on the x-axis).

Data assumptions
----------------
- `points` entries are tuples: (sensor_id, region, reading, confidence)
- `imputed` entries are: sensor_id -> (region, mean, confidence, sd)
  where `sd` models uncertainty in the reconstructed reading distribution.

Decision guides (threshold lines)
---------------------------------
- Vertical dashed line at reading = 145:
    Interpreted as an anomaly threshold (values to the right are more suspicious).
- Horizontal dashed line at confidence = 0.20:
    Interpreted as a low-confidence / spoof-suspicion boundary (values below are suspect).

Notes
-----
- Error bars for imputed points use `xerr=sd`, so uncertainty is shown along the reading axis.
  If you prefer uncertainty on confidence instead, switch to `yerr=<sd_confidence>`.

Dependencies
------------
- matplotlib

Outputs
-------
- Displays an interactive Matplotlib figure window.
"""

import matplotlib.pyplot as plt

# Raw data from Table 1
points = [
    ("S201","R2",57,0.91),
    ("S202","R2",141,0.74),
    ("S203","R2",149,0.18),
    ("S205","R3",152,0.55),
    ("S206","R3",150,0.93),
    ("S207","R4",31,0.97),
    ("S208","R4",29,0.94),
    ("S210","R3",153,0.61),
]

# Imputed distributions (from the reconstruction table)
imputed = {
    "S204": ("R3", 151.40, 0.88, 3.46),  # (region, mean, confidence, sd)
    "S209": ("R2", 100.02, 0.12, 44.18)
}

# Plot observed points
x = [p[2] for p in points]
y = [p[3] for p in points]
labels = [p[0] for p in points]
plt.figure()
plt.scatter(x, y)

for lab, xv, yv in zip(labels, x, y):
    plt.annotate(lab, (xv, yv), textcoords="offset points", xytext=(5,5))

# Plot imputed means with vertical error bars = ±1 sd
for sid, (region, mu, conf, sd) in imputed.items():
    plt.errorbar(mu, conf, xerr=sd, fmt='o')  # mean with uncertainty
    plt.annotate(sid, (mu, conf), textcoords="offset points", xytext=(5,5))

# Decision thresholds
plt.axvline(145, linestyle="--")   # anomaly threshold
plt.axhline(0.20, linestyle="--")  # spoof suspicion confidence line

plt.xlabel("Reading")
plt.ylabel("Signal confidence")
plt.title("Reading vs Confidence (observed + imputed means with uncertainty)")
plt.show()
