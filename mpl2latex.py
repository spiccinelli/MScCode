# largely based on https://github.com/mballarin97/Matplotlib2LaTeX
import matplotlib
import matplotlib.pyplot as plt

def mpl2latex():

    SIZE = 11
    
    packages = [ r'\usepackage[utf8]{inputenc}', r'\usepackage[T1]{fontenc}', ]

    # set LaTeX params
    matplotlib.rcParams.update({ 
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
            "pgf.preamble": "\n".join( packages ),
        })
    plt.rc('font', size=SIZE)           # controls default text sizes
    plt.rc('axes', titlesize=SIZE)      # fontsize of the axes title
    plt.rc('axes', labelsize=SIZE)      # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SIZE)     # fontsize of the tick labels
    plt.rc('ytick', labelsize=SIZE)     # fontsize of the tick labels
    plt.rc('legend', fontsize=SIZE)     # legend fontsize
    plt.rc('figure', titlesize=SIZE)    # fontsize of the figure title
        

def latex_figsize(wf=1, hf=(5.**0.5-1.0)/2.0, columnwidth=360.0):
    '''
    Get the correct figure size to be displayed in a latex report/publication
    
    Parameters
    ----------
    wf : float, optional
        width fraction in columnwidth units. Default to 0.5
    hf : float, optional
        height fraction in columnwidth units. Set by default to golden ratio.
    columnwidth: float 
        width of the column in latex. Get this from LaTeX using \the\columnwidth
                
    Returns
    -------
    fig_size: list of float
        fig_size [width, height] that should be given to matplotlib
    '''
    fig_width_pt = columnwidth*wf 
    inches_per_pt = 1.0/72.27               # convert pt to inch
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*hf               # height in inches
    return [fig_width, fig_height]