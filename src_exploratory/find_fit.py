import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from src import models


from src.handle_data import load_dataset
from src.handle_data import remove_collapse
from src.handle_data import only_drifts

from itertools import product
import tqdm


## gather PID-RID data for all cases

data = {}

if not os.path.exists('tmp/data.pcl'):

    for sys, st, rc, dr in tqdm.tqdm(list(product(
        ('smrf', 'scbf', 'brbf'),
        ('3', '6', '9'),
        ('ii', 'iv'),
        ('1', '2')
    ))):
        for lv in range(1, int(st)):
            lv = int(lv)

            the_case = (sys, st, rc, str(lv), dr)

            df = only_drifts(remove_collapse(load_dataset()[0]))
            case_df = df[the_case].dropna()

            rid_vals = case_df.dropna()["RID"].to_numpy().reshape(-1)
            pid_vals = case_df.dropna()["PID"].to_numpy().reshape(-1)

            model = models.Model()
            model.add_data(pid_vals, rid_vals)
            model.calculate_rolling_quantiles()
            rpid = model.rolling_pid
            rrid = model.rolling_rid_50

            data[the_case] = {
                'pid': rpid,
                'rid': rrid
            }

    with open('tmp/data.pcl', 'wb') as f:
        pickle.dump(data, f)

else:
    with open('tmp/data.pcl', 'rb') as f:
        data = pickle.load(f)


## plot the curves in a multi-page figure

from matplotlib.backends.backend_pdf import PdfPages
from tqdm import trange
with PdfPages('tmp/fig.pdf') as pdf:
    for idx in trange(len(data)):
        fig, ax = plt.subplots()
        for val in data.values():
            ax.plot(val['pid'], val['rid'], color='black', linewidth=0.05)
        idata = list(data.values())[idx]
        ax.plot(idata['pid'], idata['rid'])
        ax.set(xlim=(0.00, 0.08), ylim=(0.00, 0.08), xlabel='PID', ylabel='RID')
        pdf.savefig()
        plt.close()


## interpolate the curves to equalize the weights

from scipy.interpolate import interp1d

data_interpolated = {}
for key, item in data.items():
    pid = item['pid']
    rid = item['rid']
    ifun = interp1d(pid, rid, kind='linear')
    pid_interpolated = np.linspace(pid[0], pid[-1], 1000)
    rid_interpolated = ifun(pid_interpolated)
    data_interpolated[key] = {'pid': pid_interpolated, 'rid': rid_interpolated}


## fit a curve and obtain the residual loss

from scipy.optimize import minimize


def objective(params, pid, rid):
    a, b = params
    # rid_est = a * pid ** b      # Power
    rid_est = np.zeros(len(pid))                 # Piecewise bilinear
    rid_est[pid > a] = (pid[pid > a] - a) * b    # Piecewise bilinear
    diff = rid_est - rid
    loss = diff @ diff
    return loss

# inits = (1.00, 1.00)            # Power
inits = (0.005, 1.00)    # Piecewise bilinear

residual_loss = {}
for key, item in tqdm.tqdm(list(data_interpolated.items())):
    pid = item['pid']
    rid = item['rid']
    # res = minimize(objective, inits, args=(pid, rid), method='BFGS', options={'maxiter': 10000}, tol=1e-4)
    res = minimize(objective, inits, args=(pid, rid), method='Nelder-Mead', options={'maxiter': 10000}, tol=1e-4)
    assert res.success == True
    residual_loss[key] = res.fun
    # fig, ax = plt.subplots()
    # ax.plot(pid, rid)
    # # ax.plot(pid, res.x[0] * pid ** res.x[1]) # Power
    # rid_est = np.zeros(len(pid))
    # rid_est[pid > res.x[0]] = (pid[pid > res.x[0]] - res.x[0]) * res.x[1]
    # ax.plot(pid, rid_est)
    # plt.show()

total = sum(residual_loss.values())
print(total)
