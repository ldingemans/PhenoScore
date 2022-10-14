import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as matplotlib
from PIL import Image, ImageDraw
#Table and figure module with all corresponding functions


def color_map_color(value, cmap_name='Wistia', vmin=0, vmax=1):
    """
    Convert number to color
    
    Parameters
    ----------
    value: float
        Number to obtain color for
    cmap_name: matplotlib colormap
        The colormap
    vmin: float
        The minimum
    vmax: float
        The maximum
    
    Returns
    -------
    color: matplotlib color
        The obtained corresponding color
    """
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)  # PiYG
    rgb = cmap(norm(abs(value)))[:3]  # will return rgba, we take only first 3 so we get rgb
    color = matplotlib.colors.rgb2hex(rgb)
    return color


def add_corners(im, rad, fill_value=None, w=224, h=224):
    """
    Add rounded edges to rectangular figure, purely for aesthetic reasons

    Parameters
    ----------
    im: numpy array
        Image to add rounded edges to
    rad: float
        Radius of the edges to round

    Returns
    -------
    im: numpy array
        Image with rounded edges
    """
    circle = Image.new('L', (rad * 2, rad * 2), 0)
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0, 0, rad * 2, rad * 2), fill=255)
    alpha = Image.new('L', (w, h), 255)
    alpha.paste(circle.crop((0, 0, rad, rad)), (0, 0))
    alpha.paste(circle.crop((0, rad, rad, rad * 2)), (0, h - rad))
    alpha.paste(circle.crop((rad, 0, rad * 2, rad)), (w - rad, 0))
    alpha.paste(circle.crop((rad, rad, rad * 2, rad * 2)), (w - rad, h - rad))
    alpha = np.array(alpha)
    if im.ndim == 3:
        im[:,:,0][alpha == 0] = 1
        im[:,:,1][alpha == 0] = 1
        im[:,:,2][alpha == 0] = 1
    else:
        im[alpha == 0] = fill_value
    return im


def get_top_HPO(list_of_hpo_terms, invert_negative_correlations):
    """
    Get the top HPO terms from a list of LIME explanations
    
    Parameters
    ----------
    list_of_hpo_terms: list
        List of all generated HPO LIME explanations
    invert_negative_correlations: bool
        Whether to invert the negative correlations (i.e. if something is strongly negatively correlated with class 1,
         it points to class 0). Not useful when investigating individual explanations, but for larger groups (genetic syndromes),
         it provides more information
    Returns
    -------
    df_summ_hpo: pandas DataFrame
        The top important features
    """                          
    df_exp_total = pd.DataFrame()
    
    for exp_hpo in list_of_hpo_terms:
        df_exp_here = pd.DataFrame(exp_hpo.as_list())
        df_exp_here = pd.concat([df_exp_here.iloc[:,0].str.split('=',expand=True), df_exp_here.iloc[:,1]],axis=1)
        df_exp_here.columns = ['hpo', 'positive', 'corr']
        df_exp_total = pd.concat([df_exp_total, df_exp_here]).reset_index(drop=True)
    
    #now, we can invert all the negative correlations (i.e. if something is strongly negatively correlated with class 1, it points to class 0), since it is a binary classification problem: therefore this will only work with two classes!
    if invert_negative_correlations == True:
        df_exp_total.loc[df_exp_total['positive'] == '0', 'corr'] = -df_exp_total.loc[df_exp_total['positive'] == '0', 'corr']
    df_exp_total['positive'] = df_exp_total['positive'].astype(int)
    
    if len(list_of_hpo_terms) > 4:
        #only take into account HPO terms that are present in at least 3 / 5 predictions, if there are at least 5
        # predictions, otherwise just take them all
        threshold = int(np.round(len(list_of_hpo_terms) * 0.6))

        df_exp_total = df_exp_total[df_exp_total.groupby("hpo")['hpo'].transform('size') > (threshold - 1)]
    
    df_summ_hpo = df_exp_total.groupby('hpo').mean()
    df_summ_hpo['hpo'] = list(df_summ_hpo.index)
    
    # if there are more than 10 important features, select the 10 with highest correlations
    df_summ_hpo = df_summ_hpo[df_summ_hpo['hpo'].isin(np.array(abs(df_summ_hpo['corr']).nlargest(10).index))]
    return df_summ_hpo


def get_heatmap_from_multiple(list_of_heatmaps, fig, ax, bg_image, alpha):
    """
    Generate one average heatmap over several iterations
    
    Parameters
    ----------
    list_of_heatmaps: list
        List of all generated heatmaps over all iterations
    fig: matplotlib fig object
        Fig to write figure to
    ax: matplotlib ax object
        Ax to write figure to
    bg_image: str
        Path to image to use as background for the facial heatmap
    alpha: float
        Transparance of the background image
    
    Returns
    -------
    fig: matplotlib fig object
        Figure
    """
    heatmaps = []
    for explanation_ in list_of_heatmaps:
        ind = explanation_.top_labels[0]
        dict_heatmap = dict(explanation_.local_exp[ind])
        try:
            if np.isnan(np.vectorize(dict_heatmap.get)(explanation_.segments)).mean() == 0:
                heatmaps.append(np.vectorize(dict_heatmap.get)(explanation_.segments))
        except:
            continue
    heatmap = np.mean(heatmaps, axis=0) 
    
    max_heatmap = heatmap.max()
    min_heatmap = heatmap.min()
    
    heatmap = add_corners(heatmap, 50)
    mean_face = add_corners(bg_image, 50, min_heatmap)
    
    sm = ax.imshow(heatmap, cmap = 'seismic_r',  vmin  = -max_heatmap, vmax = max_heatmap)
    #fig.colorbar(sm, ax=ax,fraction=0.046, pad=0.04)
    ax.imshow(mean_face, alpha=alpha)
    ax.axes.xaxis.set_visible(False)
    ax.set_yticks([])
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    return fig


def draw_heatmap(explanation, n_syndromes, ax=None):
    """
    Draw the heatmap of the LIME explanations for a single instance

    Parameters
    ----------
    explanation: LIME explanation
        The generated explanation instance
    n_syndromes: list
        List of syndromes, to be used to convert indices to syndrome names
    ax: matplotlib ax instance
        Axis to plot figure to
    """
    if type(explanation) == list:
        heatmaps = []
        for explanation_ in explanation:
            ind =  explanation_.top_labels[0]
            dict_heatmap = dict(explanation_.local_exp[ind])
            try:
                if np.isnan(np.vectorize(dict_heatmap.get)(explanation_.segments)).mean() == 0:
                    heatmaps.append(np.vectorize(dict_heatmap.get)(explanation_.segments))
            except:
                continue
        heatmap = np.mean(heatmaps, axis=0)
        explanation = explanation_
    else:
        ind =  explanation.top_labels[0]
        dict_heatmap = dict(explanation.local_exp[ind])
        heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    temp, mask = explanation.get_image_and_mask(ind)
    max_heatmap, min_heatmap = heatmap.max(), heatmap.min()
    temp = add_corners()(temp, 50, min_heatmap)
    heatmap = add_corners()(heatmap, 50)
    if ax == None:
        plt.title(n_syndromes[ind] + ', LIME score:' + str(np.round(explanation.score,2)) + ' Local pred: ' + str(np.round(explanation.local_pred[0],2)) )
        plt.imshow(heatmap, cmap = 'seismic_r',  vmin  = -max_heatmap, vmax = max_heatmap)
        plt.colorbar()
        plt.imshow(temp, alpha=0.5)
        plt.axis('off')
        plt.show()
    else:
        ax.set_title(n_syndromes[ind] + ', LIME score:' + str(np.round(explanation.score,2)) + ' Local pred: ' + str(np.round(explanation.local_pred[0],2)) )
        sm = ax.imshow(heatmap, cmap = 'seismic_r',  vmin  = -max_heatmap, vmax = max_heatmap)
        ax.imshow(temp, alpha=0.5)
        ax.axis('off')
    return