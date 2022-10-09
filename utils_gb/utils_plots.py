import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection, HandlerLine2D
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss,auc, roc_auc_score, roc_curve, f1_score,confusion_matrix,classification_report
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_array, check_consistent_length


def plot_classification_stats(y, y_pred):
    y = check_array(y, ensure_2d=False, force_all_finite=True)
    y_pred = check_array(y_pred, ensure_2d=False, force_all_finite=True)
    check_consistent_length(y, y_pred)
    
    n_samples = y.shape[0]
    n_event = np.sum(y)
    n_nonevent = n_samples - n_event
    
    idx = y_pred.argsort()[::-1][:n_samples]
    yy = y[idx]

    p_event = np.append([0], np.cumsum(yy)) / n_event
    p_population = np.arange(0, n_samples + 1) / n_samples

    auroc = roc_auc_score(y, y_pred)
    fpr, tpr, _ = roc_curve(y, y_pred)
    gini = auroc * 2 - 1

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    axs[0][0].plot([0, 1], [0, 1], color='k', linestyle='--', label="Random Model")
    axs[0][0].plot([0, n_event / n_samples, 1], [0, 1, 1], color='grey',
             linestyle='--', label="Perfect Model")
    axs[0][0].plot(p_population, p_event, color="y",
             label="Model (Gini: {:.2%})".format(gini))
    axs[0][0].set_title("Cumulative Accuracy Profile (CAP)")
    axs[0][0].set_xlabel("Fraction of all population")
    axs[0][0].set_ylabel("Fraction of event population")
    axs[0][0].set_xlim(0, 1)
    axs[0][0].set_ylim(0, 1)     
    axs[0][0].legend(loc='lower right')  
    
    axs[0][1].plot(fpr, fpr, linestyle="--", color="k", label="Random Model")
    axs[0][1].plot(fpr, tpr, color="c", label="Model (AUC: {:.2%})".format(auroc))
    axs[0][1].set_title("ROC curve")
    axs[0][1].set_xlabel("False Positive Rate")
    axs[0][1].set_ylabel("True Positive Rate")
    axs[0][1].set_xlim(0 ,1)
    axs[0][1].set_ylim(0, 1)    
    axs[0][1].legend(loc='lower right')
    
    idx2 = y_pred.argsort()
    yy2 = y[idx2]
    pp = y_pred[idx2]

    cum_event = np.cumsum(yy2)
    cum_population = np.arange(0, n_samples)
    cum_nonevent = cum_population - cum_event

    p_event = cum_event / n_event
    p_nonevent = cum_nonevent / n_nonevent

    p_diff = p_nonevent - p_event
    ks_score = np.max(p_diff)
    ks_max_idx = np.argmax(p_diff)


    axs[1][0].plot(pp, p_event, color="firebrick", label="Cumulative events")
    axs[1][0].plot(pp, p_nonevent, color="lightgreen", label="Cumulative non-events")
    axs[1][0].vlines(pp[ks_max_idx], ymin=p_event[ks_max_idx],
               ymax=p_nonevent[ks_max_idx], color="k", linestyles="--")

    # Set KS value inside plot
    pos_x = pp[ks_max_idx] + 0.02
    pos_y = 0.5 * (p_nonevent[ks_max_idx] + p_event[ks_max_idx])
    text = "KS: {:.2%} at {:.2f}".format(ks_score, pp[ks_max_idx])
    axs[1][0].text(pos_x, pos_y, text, fontsize=12, rotation_mode="anchor")
    axs[1][0].set_title("Kolmogorov-Smirnov")
    axs[1][0].set_xlabel("Threshold")
    axs[1][0].set_ylabel("Cumulative probability")
    axs[1][0].set_xlim(0, 1)
    axs[1][0].set_ylim(0, 1)    
    axs[1][0].legend(loc='lower right')    

    axs[1][1].hist(y_pred[y==0], label="non-event", color="lightgreen", alpha=0.4)
    axs[1][1].hist(y_pred[y==1], label="event", color="r", alpha=0.3)
    axs[1][1].set_title("PD init distribution")
    axs[1][1].set_xlabel("PD init")
    axs[1][1].set_ylabel("# population")    
    axs[1][1].legend(loc='upper right')    
    
    return gini, auroc, ks_score ,pp[ks_max_idx]

def plot_raw_calibration(y_test, test_proba,title):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot([0, 1], [0, 1], "b--", label="Perfectly calibrated")
    ax.set_ylabel("Fraction of positives")
    ax.set_xlabel("Mean predicted value")
    ax.set_title(title+': Calibration plot (reliability curve)')
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, test_proba, n_bins=10)
    model_Brier_score = brier_score_loss(y_test, test_proba, pos_label=y_test.max())
#    ax.plot(mean_predicted_value, fraction_of_positives, "s-",label="%s (Brier: %0.3f)" % ('model', model_Brier_score) )
    ax.plot(mean_predicted_value, fraction_of_positives, "s-",label=f"model (Brier: {model_Brier_score:.2%})")   
    ax.plot(test_proba, y_test, 'o', alpha=.005, label="True values")
    ax.legend(loc="best",framealpha=.5)


    def update(handle, orig):
        handle.update_from(orig)
        handle.set_alpha(1)

    plt.legend(handler_map={PathCollection : HandlerPathCollection(update_func= update), plt.Line2D : HandlerLine2D(update_func = update)})

    plt.show()
    print(len(y_test))

def plot_final_calibration(df,ile_kubelkow, bar_width=25):
    df_agg=df.groupby([pd.cut(df['score'], ile_kubelkow, labels=False)]
                        ).agg({'dflt_ind': ['sum'],'nondflt_ind': ['sum'],'score': ['mean'],
                               'PD_init': ['mean'],'PD_final': ['mean']})
    df_agg.columns=['_'.join(col).rstrip('_') for col in df_agg.columns.values]
    #df.columns = df.columns.to_flat_index()
    df_agg.score_mean=np.round(df_agg.score_mean,0).astype('int')
    df_agg['odr']=df_agg['dflt_ind_sum']/(df_agg['dflt_ind_sum']+df_agg['nondflt_ind_sum'])
    df_agg['odds']=df_agg['nondflt_ind_sum']/df_agg['dflt_ind_sum']
    df_agg['log_odds']=np.log(df_agg['odds'])

    print(f"\nFin")

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.plot(df_agg.score_mean,df_agg.odr , "-",color='r',label="ODR" )
    ax.plot(df_agg.score_mean,df_agg.PD_init_mean , ".-",label="PD init" )
    ax.plot(df_agg.score_mean,df_agg.PD_final_mean ,".-",color='k',label="PD final" )
    ax2 = ax.twinx()
    ax2.bar(df_agg.score_mean,df_agg.dflt_ind_sum ,label="bad [rhs]",color='lightcoral',align='center', width=bar_width,alpha=0.2 )
    ax2.bar(df_agg.score_mean,df_agg.nondflt_ind_sum ,label="good [rhs]",color='lightgreen',align='center', width=bar_width,alpha=0.2 )
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)
    plt.title('Calibration')
    plt.show()
    display(df_agg)    

def plot_strategy_curve(df):
    """
    # Precision -  jak trafię to z jaką skutecznością dobrze
    # Recall - ile trafiam z celu (targetu)
    # Accuracy - ile mam trafień z całości (target i dopełnienie)
    # FPR - jaka część całości non-target stanowią złe predykcje target
    # FOR - jaką część przewidzianych non-target stanowią target'y
    """
    #https://campus.datacamp.com/courses/credit-risk-modeling-in-r/chapter-4-evaluating-a-credit-risk-model?ex=1
    df_sorted=df.sort_values(by=['score','dflt_ind'],ascending=[False, True])
    #df_sorted=df.sort_values(by=['score'],ascending=False)
    df_sorted['dflt_ind_cumsum']=df_sorted['dflt_ind'].cumsum()
    df_sorted['nondflt_ind_cumsum']=df_sorted['nondflt_ind'].cumsum()
    df_sorted['pct_population']=(df_sorted['dflt_ind_cumsum']+df_sorted['nondflt_ind_cumsum'])/df_sorted.shape[0]
    df_sorted['pct_event']=df_sorted['dflt_ind_cumsum']/(df_sorted['dflt_ind_cumsum']+df_sorted['nondflt_ind_cumsum'])
    df_sorted['TP']=df_sorted['nondflt_ind_cumsum']
    df_sorted['FP']=df_sorted['dflt_ind_cumsum']
    df_sorted['FN']=df_sorted['nondflt_ind_cumsum'].iloc[-1]-df_sorted['nondflt_ind_cumsum']
    df_sorted['TN']=df_sorted['dflt_ind_cumsum'].iloc[-1]-df_sorted['dflt_ind_cumsum']
    df_sorted['FPR']=df_sorted['FP']/(df_sorted['FP']+df_sorted['TN'])
    df_sorted['FOR']=df_sorted['FN']/(df_sorted['FN']+df_sorted['TN'])    
    df_sorted['accuracy']=(df_sorted.TP+df_sorted.TN)/(df_sorted.TP+df_sorted.FP+df_sorted.FN+df_sorted.TN)
    df_sorted['precision']=df_sorted.TP/(df_sorted.TP+df_sorted.FP)
    df_sorted['recall']=df_sorted.TP/(df_sorted.TP+df_sorted.FN)
    df_sorted['F1_score']=2*df_sorted.TP/(2*df_sorted.TP+df_sorted.FP+df_sorted.FN)
    df_sorted['PD_final_toPoint']=df_sorted['PD_final'].cumsum()/(df_sorted['dflt_ind_cumsum']+df_sorted['nondflt_ind_cumsum'])
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.plot(df_sorted['pct_population'],df_sorted['pct_event'] , "-",label="% Bads" )
    ax.plot(df_sorted['pct_population'],df_sorted['PD_final'] , "-",label="PD" )
    ax.plot(df_sorted['pct_population'],df_sorted['accuracy'] , "-",label="accuracy" )
    ax.plot(df_sorted['pct_population'],df_sorted['precision'] , "--",label="precision" )
    ax.plot(df_sorted['pct_population'],df_sorted['recall'] , "--",label="recall" )
    ax.plot(df_sorted['pct_population'],df_sorted['FPR'] , ":",label="FPR" )
    ax.plot(df_sorted['pct_population'],df_sorted['FOR'] , ":",label="FOR" )   
    ax.plot(df_sorted['pct_population'],df_sorted['F1_score'] , "-",label="F1_score" )
    ax.plot(df_sorted['pct_population'],df_sorted['PD_final_toPoint'] , "-",label="avgPD_toPoint" )
    #ax.plot(df_sorted['pct_population'],df_sorted['pct_population'] , "-",label="% population" )
    ax.legend(loc="best",framealpha=.5)
    plt.title('Strategy curve')
    plt.ylabel('% accumulated observed events')
    plt.xlabel('% population sorted by PD')
    plt.show()
    
def plot_strategy_curve_plotly(df):
    """
    # Precision -  jak trafię to z jaką skutecznością dobrze
    # Recall - ile trafiam z celu (targetu)
    # Accuracy - ile mam trafień z całości (target i dopełnienie)
    # FPR - jaka część całości non-target stanowią złe predykcje target
    # FOR - jaką część przewidzianych non-target stanowią target'y
    """    
    #https://campus.datacamp.com/courses/credit-risk-modeling-in-r/chapter-4-evaluating-a-credit-risk-model?ex=1
    pd.options.plotting.backend = "plotly"
    df_sorted=df.sort_values(by=['score','dflt_ind'],ascending=[False, True])
    #df_sorted=df.sort_values(by=['score'],ascending=False)
    df_sorted['dflt_ind_cumsum']=df_sorted['dflt_ind'].cumsum()
    df_sorted['nondflt_ind_cumsum']=df_sorted['nondflt_ind'].cumsum()
    df_sorted['pct_population']=(df_sorted['dflt_ind_cumsum']+df_sorted['nondflt_ind_cumsum'])/df_sorted.shape[0]
    df_sorted['pct_event']=df_sorted['dflt_ind_cumsum']/(df_sorted['dflt_ind_cumsum']+df_sorted['nondflt_ind_cumsum'])
    df_sorted['TP']=df_sorted['nondflt_ind_cumsum']
    df_sorted['FP']=df_sorted['dflt_ind_cumsum']
    df_sorted['FN']=df_sorted['nondflt_ind_cumsum'].iloc[-1]-df_sorted['nondflt_ind_cumsum']
    df_sorted['TN']=df_sorted['dflt_ind_cumsum'].iloc[-1]-df_sorted['dflt_ind_cumsum']
    df_sorted['FPR']=df_sorted['FP']/(df_sorted['FP']+df_sorted['TN'])
    df_sorted['FOR']=df_sorted['FN']/(df_sorted['FN']+df_sorted['TN'])
    df_sorted['accuracy']=(df_sorted.TP+df_sorted.TN)/(df_sorted.TP+df_sorted.FP+df_sorted.FN+df_sorted.TN)
    df_sorted['precision']=df_sorted.TP/(df_sorted.TP+df_sorted.FP)
    df_sorted['recall']=df_sorted.TP/(df_sorted.TP+df_sorted.FN)
    df_sorted['F1_score']=2*df_sorted.TP/(2*df_sorted.TP+df_sorted.FP+df_sorted.FN)
    df_sorted['avgPD_toPoint']=df_sorted['PD_final'].cumsum()/(df_sorted['dflt_ind_cumsum']+df_sorted['nondflt_ind_cumsum'])
    df_sorted['PD']=df_sorted['PD_final']
    Y=['precision','recall','F1_score','FPR','FOR','accuracy','PD','avgPD_toPoint','pct_event','PD']
    fig = df_sorted.plot(x='pct_population'\
                   , y=Y\
                   ,labels={
                        "variable": "Legend",
                        "pct_population": "% population sorted by PD",
                        "value": "% of metric",
                        "pct_event": "% Bad",    
                    }
                    ,title="Strategy curve"
                  )
    fig.data[Y.index('pct_event')].name="% Bads"
    fig.data[Y.index('PD')].line.color = "Black"
    fig.data[Y.index('PD')].line.width = 4
    fig.data[Y.index('precision')].line.dash = 'dash'
    fig.data[Y.index('recall')].line.dash = 'dash'
    fig.data[Y.index('F1_score')].line.dash = 'dash'
    fig.data[Y.index('accuracy')].line.dash = 'dashdot' 
    #fig.data[Y.index('accuracy')].line.color = "Green"    
    fig.data[Y.index('FPR')].line.dash = 'dot'
    fig.data[Y.index('FOR')].line.dash = 'dot'
    fig.data[Y.index('PD')].line.dash = 'solid'
    
    fig.update_layout( plot_bgcolor="rgba(0,0,0,0)") #per_bgcolor="rgba(0,0,0,0)", 
    fig.update_xaxes(showline=True, linewidth=.1, linecolor='black', gridcolor='LightGrey')
    fig.update_yaxes(showline=True, linewidth=.1, linecolor='black', gridcolor='LightGrey')
    fig.layout.xaxis.tickformat = '.0%'
    fig.layout.yaxis.tickformat = '.0%'
    fig.layout.xaxis.showspikes=True
    fig.layout.xaxis.spikesnap="cursor"
    fig.layout.yaxis.spikesnap="cursor"
    fig.layout.xaxis.spikemode="across"
    fig.layout.xaxis.spikecolor="Black"
    fig.layout.yaxis.spikecolor="Black"
    fig.layout.xaxis.spikethickness=1
    fig.layout.xaxis.spikedash="solid"
    fig.layout.yaxis.showspikes=True
    fig.layout.yaxis.spikedash="solid"
    fig.layout.yaxis.spikethickness=1
    fig.layout.hovermode="x"
    fig.data[Y.index('precision')].hovertemplate='%{y:.2%}'
    fig.data[Y.index('recall')].hovertemplate='%{y:.2%}'
    fig.data[Y.index('F1_score')].hovertemplate='%{y:.2%}'
    fig.data[Y.index('FPR')].hovertemplate='%{y:.2%}'
    fig.data[Y.index('FOR')].hovertemplate='%{y:.2%}'
    fig.data[Y.index('accuracy')].hovertemplate='%{y:.2%}'
    fig.data[Y.index('PD')].hovertemplate='%{y:.2%}'
    fig.data[Y.index('avgPD_toPoint')].hovertemplate='%{y:.2%}'
    fig.data[Y.index('pct_event')].hovertemplate='%{y:.2%}'
    fig.layout.width=1000
    fig.layout.height=800
    fig.show()
    pd.options.plotting.backend = 'matplotlib'