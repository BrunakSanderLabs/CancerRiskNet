#TODO optional : this function maybe could be moved out of here
def save_figure_and_subplots(figname, fig, **kwargs):
    """
        figname: name of the file to save
        fig: matplotlib.pyplot.figure object

        This function takes some figure and save its content. In addition it save al; its axes/subfigure
        for offline processing.
    """
    if 'format' not in kwargs:
        kwargs['format'] = 'png'
    for ax_num, ax in enumerate(fig.axes):
        extent = ax.get_tightbbox(renderer=fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig("{}_{}.{}".format(figname, ax_num, kwargs['format']), bbox_inches=extent, **kwargs)
    fig.savefig("{}.{}".format(figname, kwargs['format']), bbox_inches='tight', **kwargs)
