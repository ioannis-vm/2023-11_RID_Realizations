# Imports
from pathlib import Path
import os
from itertools import product
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src import models

def add_plot(ax, parameters):

    model = models.Model_1_Weibull()
    model.censoring_limit = 0.0025
    model.parameters = parameters


    model_pid = np.linspace(0.00, 0.06, 1000)
    model_rid_50 = model.evaluate_inverse_cdf(0.50, model_pid)
    model_rid_20 = model.evaluate_inverse_cdf(0.20, model_pid)
    model_rid_80 = model.evaluate_inverse_cdf(0.80, model_pid)

    ax.plot(
        model_rid_50,
        model_pid,
        'black',
        linewidth=1.5,
        label='Conditional Weibull',
        zorder=10,
    )
    ax.plot(model_rid_20, model_pid, 'black', linestyle='dashed', zorder=10)
    ax.plot(model_rid_80, model_pid, 'black', linestyle='dashed', zorder=10)

    text = ''
    text += f'$\\theta_0$ = {model.parameters[0]}\n'
    text += f'$\\theta_1$ = {model.parameters[1]}\n'
    text += f'$\\theta_2$ = {model.parameters[2]}\n'
    ax.text(0.95, 0.25, text, ha='right', va='center', transform=ax.transAxes)
    ax.set(
        xlim=(xmin, xmax),
        ylim=(ymin, ymax),
    )
    ax.grid(which='both')



# plotting parameters
xmin, xmax = -0.00199, 0.0221  # rid
ymin, ymax = -0.00499, 0.0699  # pid


fig, axs = plt.subplots(3, 3, figsize=(8.0, 4.0), sharex=True, sharey=True)
add_plot(axs[0,0], np.array((0.20 / 100.00, 0.25, 2.00)))
add_plot(axs[1,0], np.array((1.00 / 100.00, 0.25, 2.00)))
add_plot(axs[2,0], np.array((2.00 / 100.00, 0.25, 2.00)))
add_plot(axs[0,1], np.array((1.00 / 100.00, 0.05, 2.00)))
add_plot(axs[1,1], np.array((1.00 / 100.00, 0.15, 2.00)))
add_plot(axs[2,1], np.array((1.00 / 100.00, 0.30, 2.00)))
add_plot(axs[0,2], np.array((1.00 / 100.00, 0.25, 6.00)))
add_plot(axs[1,2], np.array((1.00 / 100.00, 0.25, 2.00)))
add_plot(axs[2,2], np.array((1.00 / 100.00, 0.25, 1.00)))


# ax.legend(frameon=False, loc='lower right')
fig.tight_layout()
# plt.show()
plt.savefig('/tmp/model.png', dpi=600)
