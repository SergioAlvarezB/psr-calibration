import numpy as np
import torch
import ternary

import matplotlib.pyplot as plt

from psrcal.psr import logcost, shift



p = torch.Tensor([0.2, 0.2, 0.6])
off = torch.Tensor([2, 0, 0])

def logS(q):
    c = 0.
    for l in range(3):
        c += p[l]*logcost(torch.Tensor(q).view([1, -1]), l)
    return torch.log(c).item()

shift_logcost = shift(logcost, off)

def slogS(q):
    c = 0.
    q = torch.Tensor(q).view([1, -1])
    qs = torch.softmax(torch.log(q) + off, dim=1)
    for l in range(3):
        c += p[l]*shift_logcost(qs, l)
    return torch.log(c).item()


scale=60

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

_, tax1 = ternary.figure(scale=scale, ax=ax1)
tax1.heatmapf(logS, boundary=False, style="hexagonal",
             cmap=plt.get_cmap('gnuplot'), colorbar=False)
tax1.boundary(linewidth=2.0)

tax1.scatter([p.numpy()*scale], marker='s', color='red', zorder=2, label="p={}".format(p.numpy()))

tax1.get_axes().axis('off')
ticks = np.linspace(0, 1, 10).tolist()
tax1.ticks(ticks=ticks, axis='rlb', linewidth=1, clockwise=True, offset=0.03, tick_formats="%0.1f")
tax1.clear_matplotlib_ticks()
tax1._redraw_labels()
tax1.set_title("log(Log Score)\n\n")
tax1.legend()



_, tax2 = ternary.figure(scale=scale, ax=ax2)
tax2.heatmapf(slogS, boundary=False, style="hexagonal",
             cmap=plt.get_cmap('gnuplot'), colorbar=False)
tax2.boundary(linewidth=2.0)

tax2.scatter([p.numpy()*scale], marker='s', color='red', zorder=2, label="p={}".format(p.numpy()))

tax2.get_axes().axis('off')
ticks = np.linspace(0, 1, 10).tolist()
tax2.ticks(ticks=ticks, axis='rlb', linewidth=1, clockwise=True, offset=0.03, tick_formats="%0.1f")
tax2.clear_matplotlib_ticks()
tax2._redraw_labels()
tax2.set_title("log(Shifted-Log Score)\n o={}\n\n".format(off.numpy()))
tax2.legend()


ternary.plt.show()


