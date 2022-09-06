import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from explainability_lime.LIME import predict_hpo, predict_image, get_norm_image, random_mask, explain_prediction, draw_heatmap

import os
import seaborn as sns

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
    import matplotlib.cm as cm
    import matplotlib as matplotlib
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
    from PIL import Image, ImageDraw
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

def gen_vus_figure(exp_faces_all, exp_hpos_all, img, preds_face, preds_hpo, preds_both, filename='individual_lime_explanations.pdf'):
    """
    Generate the average heatmap and HPO explanation for an individual prediction, for a VUS for instance
    
    Parameters
    ----------
    exp_faces_all: list
        List of all generated heatmaps over all iterations
    exp_hpos_all: list
        List of all generated HPO LIME explanations
    img: str
        Path to original facial image
    preds_face: float
        Prediction score for face data
    preds_hpo: float
        Prediction score for HPO data
    preds_both: float
        Prediction score for both i.e. PhenoScore
    filename: str
        Filename of exported figure
    
    Returns
    -------
    fig: matplotlib fig object
        Figure
    """                          
    fig, axs = plt.subplots(1,2, figsize=(12,5))
    axs = axs.flatten()

    fig = get_heatmap_from_multiple(exp_faces_all, fig, axs[0], get_norm_image(img)[0], 0.6)
    axs[0].set_title('Face: ' + str(np.round(preds_face, 2)), fontsize=18, fontweight='bold')
  
    df_summ_hpo = get_top_HPO(exp_hpos_all, False)
    
    axs[1].set_title('HPO: ' + str(np.round(preds_hpo, 2)), fontsize=18, fontweight='bold')

    df_summ_hpo = df_summ_hpo.sort_values('corr', ascending=False)
    
    if len(df_summ_hpo) > 0:
        g = sns.barplot(x=df_summ_hpo['corr'], y=list(df_summ_hpo.index), color='blue', alpha=0.6, ax=axs[1])
        
        for bar in g.patches:
            if bar.get_width() < 0:
                bar.set_color('red')       
        df_summ_hpo = df_summ_hpo.reset_index(drop=True)
        for y in range(len(df_summ_hpo)):
            axs[1].text(0, y ,df_summ_hpo.loc[y,'hpo'] + ' = ' + str(int(df_summ_hpo.loc[y,'positive'])), fontsize=12, horizontalalignment='center', verticalalignment='center', fontweight='semibold')
        axs[1].set_xlabel('LIME regression coefficient')
            
        axs[1].set_xlim(-0.25, 0.25)
        axs[1].set_yticks([])
    else:
        axs[1].text(0.25, 0.5, 'No relevant features found')
    axs[1].axes.yaxis.set_visible(False)
    axs[1].spines['left'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    fig.suptitle('PhenoScore: '+ str(np.round(preds_both, 2)), fontsize=20, fontweight='bold')
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print("Figure saved as " + filename)
    return
    
def get_top_HPO(list_of_hpo_terms, invert_negative_correlations):
    """
    Get the top HPO terms from a list of LIME explanations
    
    Parameters
    ----------
    list_of_hpo_terms: list
        List of all generated HPO LIME explanations
    invert_negative_correlations: bool
        Whether to invert the negative correlations (i.e. if something is strongly negatively correlated with class 1, it points to class 0). Not usefull when investigating individual explanations, but for larger groups (genetic syndromes), it provides more information
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
    
    if len(list_of_hpo_terms) == 5:
        #only take into account HPO terms that are present in at least 3 of the 5 predictions, if there are 5 predictions, otherwise just take them all
        df_exp_total = df_exp_total[df_exp_total.groupby("hpo")['hpo'].transform('size') > 2]
    
    df_summ_hpo = df_exp_total.groupby('hpo').mean()
    df_summ_hpo['hpo'] = list(df_summ_hpo.index)
    
    #if there are more then 10 important features, select the 10 with highest correlations
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
    fig.colorbar(sm, ax=ax,fraction=0.046, pad=0.04)
    ax.imshow(mean_face, alpha=alpha)
    ax.axes.xaxis.set_visible(False)
    ax.set_yticks([])
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    return fig

def gen_lime_and_results_figure(df_results, gene_name, bg_image,  df_data, mode, filename='face_hpo_lime_explanations.pdf'):
    """
    Plot the LIME and results figure of the paper
    
    Parameters
    ----------
    df_results: pandas DataFrame
        Data containing the results to plot
    gene_name: str
        Gene to plot as title
    bg_image: str
        Path to image to use as background for the facial heatmap
    df_data: pandas DataFrame
        Original dataframe with raw data, used to plot the original prevalences of the HPO terms in the figure to check whether the recovered HPO terms make sense
    mode: str
        PhenoScore mode, either hpo/face/both depending on the data available and therefore which analysis to run.
    filename: str
        Filename of exported figure
    """
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mean_face_norm = get_norm_image(bg_image)[0]
    
    score_aroc_both = np.round(df_results.loc[0, 'roc_both_svm'], 2)
    score_aroc_face = np.round(df_results.loc[0, 'roc_face_svm'], 2)
    score_aroc_hpo = np.round(df_results.loc[0, 'roc_hpo_svm'], 2)

    fig, axs = plt.subplots(1,2, figsize=(12,5))
    axs = axs.flatten()
    
    df_exp = pd.DataFrame()
    df_exp['hpo_pred'] = np.array(df_results.loc[0, 'hpo_pred'])
    df_exp['face_pred'] = np.array(df_results.loc[0, 'face_pred'])
    for i in range(len(df_exp)):
        if type(df_exp.loc[i, 'face_pred']) == list:
            df_exp.loc[i,'face_pred'] = np.mean(df_exp.loc[i,'face_pred'])
    df_exp['svm_pred'] = np.array(df_results.loc[0, 'svm_preds'])
    df_exp['hpo_exp'] = np.array(df_results.loc[0, 'hpo_explanation'])
    df_exp['face_exp'] = np.array(df_results.loc[0, 'face_explanation'])
    df_exp['y_true'] = np.array(df_results.loc[0, 'y_real'])
    
    if mode != 'hpo':
        df_top = df_exp[df_exp['face_exp'] != ''].reset_index(drop=True)
    else:
        df_top = df_exp[df_exp['hpo_exp'] != ''].reset_index(drop=True)
        
    if mode != 'face':
        df_summ_hpo = get_top_HPO(df_top.loc[:, 'hpo_exp'].explode(), True)
    
    if mode != 'hpo':
        fig = get_heatmap_from_multiple(df_top.loc[:,'face_exp'].explode(), fig, axs[0], mean_face_norm, 0.5)
        axs[0].set_title('Face: ' + str(score_aroc_face), fontsize=18, fontweight='bold')
        axs[0].set_ylabel(gene_name, fontsize=18, fontweight='bold', style='italic')
        axs[0].annotate("n="+str(np.sum(df_results.loc[0, 'y_real'] == 1)), (95,240) , annotation_clip=False, fontsize=16, fontweight='bold')
    
    if mode != 'face':
        axs[1].set_title('HPO: ' + str(score_aroc_hpo), fontsize=18, fontweight='bold')
    
        df_summ_hpo = df_summ_hpo.sort_values('corr', ascending=False)
        
        for hpo_term in df_summ_hpo['hpo']:
            df_summ_hpo.loc[hpo_term, 'prev_0'] = df_data.loc[df_data['y_label'] == 0, 'hpo_name_inc_parents'].astype(str).str.contains(hpo_term).mean()
            df_summ_hpo.loc[hpo_term, 'prev_1'] = df_data.loc[df_data['y_label'] == 1, 'hpo_name_inc_parents'].astype(str).str.contains(hpo_term).mean()
        
        g = sns.barplot(x=df_summ_hpo['corr'], y=list(df_summ_hpo.index), color='blue', alpha=0.6, ax=axs[1])
        
        for bar in g.patches:
            if bar.get_width() < 0:
                bar.set_color('red')       
        axs[1].set_xlim(-0.25, 0.25)
        axs[1].set_yticks([])
        axs[1].axes.yaxis.set_visible(False)
        axs[1].spines['left'].set_visible(False)
        axs[1].spines['right'].set_visible(False)
        axs[1].spines['top'].set_visible(False)
        axs[1].set_xlabel('LIME regression coefficient')
        df_summ_hpo = df_summ_hpo.reset_index(drop=True)
        for y in range(len(df_summ_hpo)):
            axs[1].text(0, y ,df_summ_hpo.loc[y,'hpo'], fontsize=12, horizontalalignment='center', verticalalignment='center', fontweight='semibold')
            axs[1].text(-0.28, y, str(int(np.round(df_summ_hpo.loc[y,'prev_0']*100))) + '%', fontsize=12, horizontalalignment='left', verticalalignment='center')    
            axs[1].text(0.28, y, str(int(np.round(df_summ_hpo.loc[y, 'prev_1']*100))) + '%', fontsize=12, horizontalalignment='right', verticalalignment='center')    
        
    fig.suptitle('PhenoScore: '+ str(score_aroc_both), fontsize=20, fontweight='bold')
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print("Figure saved as " + filename)
    return

def plot_incremental(df_incremental):
    """
    Plot the figure of the incremental results of the paper
    
    Parameters
    ----------
    df_incremental: pandas DataFrame
        Data containing the results to plot
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_style("darkgrid")
    
    CUT_OFF = 20
    
    fig, axs = plt.subplots(1, 2, figsize=(15,5))
    y_axis_labels = ['Brier score', 'AROC']
    y_lim = [0.5, 1]     
    
    for i, y_plot in enumerate(['brier_both', 'roc_both']):
        sns.lineplot(data=df_incremental[df_incremental.n_patients < (CUT_OFF+1)].reset_index(drop=True), x='n_patients', estimator=np.median, y=y_plot, ax=axs[i], legend='auto', label='Median score over all syndromes')
        axs[i].set_xlim(2,CUT_OFF)
        axs[i].set_xticks(range(2,(CUT_OFF+1)))
        axs[i].set_xticklabels(range(2,(CUT_OFF+1)))
        axs[i].set_xlabel('Number of included patients in training')
        
        axs[i].set_ylim(0,y_lim[i])
        axs[i].set_ylabel(y_axis_labels[i])
    axs[0].set_yticks(np.array(range(11))/20)
    axs[1].set_yticks(np.array(range(11))/10)
    
    plt.legend(loc="lower center", bbox_to_anchor=(-0.1, -0.25), borderaxespad=0.)
    plt.savefig("fig_brier_aroc_incremental_median.png", dpi=300, bbox_inches='tight')
    print("Figure saved as fig_brier_aroc_incremental_median.png")
    plt.show()
    
    df_median_scores = df_incremental.groupby('n_patients').median()
    return   