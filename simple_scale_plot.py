import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext

x = [10000, 100000, 1000000, 10000000]


fairlearn_y = np.array([11.250877, 101.649542, 1013.187704, 12310.272551])

hybrid_5_y = np.array([0.1, 1.783137, 9.305699, 75.93784500000001])

hybrid_combo_y = np.array([3,  29.216354, 339.47565900000006, 4412.193808])

# 12310 = 3.4 hours
# 4412  = 1.2 hours

plt.figure(1, figsize=(10, 6))

plt.plot(x, fairlearn_y, label="fairlearn")
plt.plot(x, hybrid_combo_y, label="hybrid-combo")
plt.plot(x, hybrid_5_y, label="hybrid-5 (no grid, LP-only)")

# plt.plot(x, 100 * (fairlearn_y - hybrid_combo_y) / fairlearn_y)

plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.xticks(x)
ax = plt.gca()
ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))

plt.show()
