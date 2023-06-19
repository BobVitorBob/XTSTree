import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def plot(vals, labels=None, detected_anomalies=[], continuous_anomalies=[], margins=None, title='', save=False, img_name=None, show=True, divisions=[], sec_plots=[], figsize=(4,2), dpi=360, color_gradient=None, color_pallete='viridis', backgrounds=[], show_axis: (bool, bool)=(True, True)):
  X=range(len(vals))
  max_y, min_y = max([max(vals), *[max(sec_val) for sec_val in sec_plots]]), min([min(vals), *[min(sec_val) for sec_val in sec_plots]])
  fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
  ax.set_title(title)
  
  ax.get_xaxis().set_visible(show_axis[0])
  ax.get_yaxis().set_visible(show_axis[1])
  
  for bg in backgrounds:
    plt.axvspan(bg['start'], bg['end'], facecolor=bg['color'], alpha=0.1)
  if color_gradient and len(color_gradient) > 0:
    points = np.array([X, vals]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(min(color_gradient), max(color_gradient))
    # print([[i, val] for i, val in enumerate(vals)])
    lc = LineCollection(segments, cmap=color_pallete, norm=norm)
    # Set the values used for colormapping
    lc.set_array(color_gradient)
    lc.set_linewidth(0.5)
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax)
  else:
    plt.plot(vals, linewidth=0.5)

 
  if labels:
    plt.xticks([label['x'] for label in labels], [label['label'] for label in labels])
  for subplot in sec_plots:
    plt.plot(subplot, linewidth=0.5, color='r')
  for i in detected_anomalies:
    plt.scatter(i, vals[i], s=800, marker="x", color="r", alpha=0.6)
  anom_dev = max_y - min_y
  for i in continuous_anomalies:
    anom_top_margins = []
    anom_bottom_margins = []
    for point in range(i[0], i[1]):
      anom_top_margins.append(vals[point]+anom_dev)
      anom_bottom_margins.append(vals[point]-anom_dev)
    plt.fill_between(range(i[0], i[1]), anom_top_margins, anom_bottom_margins, alpha = 0.7, color="r", interpolate=True)
  plt.fill_between(
    [i for i in range(len(vals))],
    [max_y for i in range(len(vals))],
    [min_y for i in range(len(vals))],
    where=[True if i in divisions else False for i in range(len(vals))],
    color='grey', linestyle='--', alpha=0.5
  )
  j = 0
  if margins:
    for margin in margins:
      plt.fill_between(range(len(vals)), margin[0], margin[1], alpha = 0.1, color="b")
  fig.tight_layout()
  if save:
    fig.savefig(img_name if img_name is not None else title, bbox_inches='tight')
  if show:	
     plt.show()
  else:
    plt.close(fig)