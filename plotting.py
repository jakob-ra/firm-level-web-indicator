import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
import numpy as np
from matplotlib.ticker import PercentFormatter, FuncFormatter
import seaborn as sns
import matplotlib as mpl

# latex plotting
# import os
# os.environ["PATH"] += os.pathsep + 'C:/texlive/2022/bin\win32'

# plotting functions
def plot_barh(df: pd.Series, title='', xlabel='', ylabel='', fontsize=plt.rcParams['font.size'],
              extend_x_axis=0.1, label_fmt='%.1f', color='tab:blue'):
    # other label format: '{:,.0f}'
    df.plot(kind='barh', figsize=(10, 7), fontsize=fontsize, color=color)
    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    ax = plt.gca()
    ax.bar_label(ax.containers[0], fmt=label_fmt, label_type='edge', padding=3, fontsize=fontsize)
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter(label_fmt))
    # extend x-axis to make labels visible
    ax.set_xlim(right=df.max() + df.max()*extend_x_axis)
    plt.gca().invert_yaxis()
    plt.show()

def fix_hist_step_vertical_line_at_end(ax):
    axpolygons = [poly for poly in ax.get_children() if isinstance(poly, mpl.patches.Polygon)]
    for poly in axpolygons:
        poly.set_xy(poly.get_xy()[:-1])

def plot_hist(df: pd.Series, title='', xlabel='', ylabel='', clip=None, n_bins=25, log_bins=False, logx=False,
              logy=False, relative_freq=False, cumulative_step=False, alpha=1.0, density=False, label=None, decimals=0):
    if clip:
        df = np.clip(df, a_min=None, a_max=clip)
    if log_bins:
        bins = np.logspace(np.log10(df.min()),np.log10(df.max()), n_bins, dtype=int)
        if cumulative_step:
            _, bins = np.histogram(np.log10(df), bins='auto')
            bins = np.power(10, bins)
    else:
        bins = n_bins
    if relative_freq:
        weights = np.ones_like(df)*100 / df.size
    else:
        weights = None
    if cumulative_step:
        density = True
        cumulative = True
        histtype = 'step'
    else:
        density = False
        cumulative = False
        histtype = 'bar'
    ax = df.plot.hist(logx=logx, logy=logy, bins=bins, weights=weights, density=density, cumulative=cumulative,
                 histtype=histtype, alpha=alpha, label=label)
    if cumulative_step:
        fix_hist_step_vertical_line_at_end(ax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if relative_freq:
        plt.gca().yaxis.set_major_formatter(PercentFormatter(decimals=decimals))
    # else:
    #     plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
    if clip:
        plt.gca().set_xticklabels([f'{x:,}' if x < clip else f'{x:,}+' for x in plt.gca().get_xticks()])

    return ax

def plot_correlation_heatmap(corr: pd.DataFrame, title='', vmin=-1, vmax=1, cmap='coolwarm', fmt='.2f',
                             annot=True, square=True, linewidths=.5,
                             cbar_kws={'shrink': .4, 'ticks': [-1, -.5, 0, 0.5, 1]}, figsize=(15, 10)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.heatmap(corr, annot=annot, ax=ax, fmt=fmt, cmap=cmap, vmin=vmin, vmax=vmax, square=square,
                linewidths=linewidths, cbar_kws=cbar_kws)
    plt.xticks(rotation=45, ha='right')
    ax.tick_params(axis=u'both', which=u'both', length=0)
    plt.title(title)

    return fig, ax

# set rcParams
def rcParams():
    # reset rcParams
    plt.rcParams.update(plt.rcParamsDefault)

    # plt.rcParams['figure.figsize'] = [12, 6]
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    # plt.rcParams['font.size'] = 12

    # set theme
    # plt.style.use('seaborn-darkgrid')
    # plt.style.use('ggplot')
    plt.style.use(['science', 'no-latex'])
    # plt.style.use(['science', 'ieee', 'no-latex'])

    # tight layout
    plt.rcParams['figure.autolayout'] = True

    # set font
    # plt.rcParams['font.family'] = 'sans-serif'
    # plt.rcParams['font.sans-serif'] = 'Tahoma'

    # set color palette
    # plt.rcParams['image.cmap'] = 'Set2'
    # plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set2.colors)

rcParams()