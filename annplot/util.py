# -*- coding: utf-8 -*-
'''
Just a few functions that are used everywhere.
'''
import matplotlib as mpl
import matplotlib.pyplot as plt

from math import sqrt


def show():
    '''Just a wrapper for the normal plt.show, also calls pyplot.tight_layout()

    '''
    set_sensible_sizes()
    plt.tight_layout()
    plt.show()


def save(fig, filename, width=396.0):
    '''Saves the plot to file. Note that you should absolutely include a file
    extension in the filename.
    Recommended formats are: png (fixed size) or eps (vector format suitable
    for LaTEX)
    Also calls pyplot.tight_layout()

    To make figures prettier in LaTEX, this makes the figure equal in width to
    the column width.
    Get the width you are using in LaTEX with "\showthe\columnwidth"
    '''
    set_sensible_sizes()
    latexprep(width)
    plt.tight_layout()
    fig.savefig(filename)


def set_sensible_sizes():
    '''Sets the font sizes to large'''
    mpl.rcParams['axes.labelsize'] = 'large'
    mpl.rcParams['xtick.labelsize'] = 'large'
    mpl.rcParams['ytick.labelsize'] = 'large'


def latexprep(width=396.0):
    '''To make figures prettier in LaTEX, this makes the figure equal in width
    to the column width.
    Get the width you are using in LaTEX with "\showthe\columnwidth"
    '''
    fig_width_pt = width  # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0 / 72.27               # Convert pt to inch
    golden_mean = (sqrt(5) - 1.0) / 2.0         # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = fig_width * golden_mean      # height in inches
    fig_size = [fig_width, fig_height]
    #Update settings
    plt.rcParams['figure.figsize'] = fig_size
