import matplotlib.pyplot as plt


def plot(vals, labels=None, detected_anomalies=[], continuous_anomalies=[], margins=None, title='', save=False, img_name=None, show=True, divisions=[], sec_plots=[]):
	X=range(len(vals))
	max_y, min_y = max([max(vals), *[max(sec_val) for sec_val in sec_plots]]), min([min(vals), *[min(sec_val) for sec_val in sec_plots]])
	fig, ax = plt.subplots(figsize=(20, 6), dpi=720)
	ax.set_title(title)
	plt.plot(vals)
	if labels:
		plt.xticks([label[0] for label in labels], [label[1] for label in labels], rotation=45)
	for subplot in sec_plots:
		plt.plot(subplot)
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
	