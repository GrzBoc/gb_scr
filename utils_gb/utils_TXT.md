### init'y

import utils_gb.utils_data
import utils_gb.utils_plots
import utils_gb.utils_stats


### utils_data

import numpy as np
import pandas as pd

def describe_df(df):
    w=pd.DataFrame(df.dtypes, columns=['dtype'])
    w['NA cnt']=df.isna().sum()
    w['NA pct']=100*w['NA cnt']/df.shape[0]
    w['Uniq cnt']=df.nunique()
    w['Uniq pct']=100*w['Uniq cnt']/df.shape[0]
    w['Uniq_nonNA pct']=np.where( w['NA pct']<100.0 ,100*w['Uniq cnt']/(df.shape[0]-w['NA cnt']),np.nan)
    w['Is numeric']=w['dtype'].astype(str).isin(['int64','int32','int16', 'int8', 'float64', 'float32', 'float16', 'float8', 'bool'])*1
    w['Not numeric']= np.where(w['Is numeric']==1,0,1)
    w['Potential Cat.']=np.where(w['Uniq_nonNA pct']<=10,1,0)

    w['Top 1 - value']=pd.DataFrame(df[w[(w['Uniq pct']<=25)&(w['Uniq cnt']>0)].index].apply(lambda x:pd.DataFrame(x.value_counts()).index[0])) 
    w['Top 1 - count']=pd.DataFrame(df[w[(w['Uniq pct']<=25)&(w['Uniq cnt']>0)].index].apply(lambda x: x.value_counts().iloc[0] ))
    w['Top 2 - value']=pd.DataFrame(df[w[(w['Uniq pct']<=25)&(w['Uniq cnt']>1)].index].apply(lambda x:pd.DataFrame(x.value_counts()).index[1])) 
    w['Top 2 - count']=pd.DataFrame(df[w[(w['Uniq pct']<=25)&(w['Uniq cnt']>1)].index].apply(lambda x: x.value_counts().iloc[1] ))
    w['Top 3 - value']=pd.DataFrame(df[w[(w['Uniq pct']<=25)&(w['Uniq cnt']>2)].index].apply(lambda x:pd.DataFrame(x.value_counts()).index[2])) 
    w['Top 3 - count']=pd.DataFrame(df[w[(w['Uniq pct']<=25)&(w['Uniq cnt']>2)].index].apply(lambda x: x.value_counts().iloc[2] ))

    w['mean - value']=df[w[w['Is numeric']==1].index].mean()
    w['std - value']=df[w[w['Is numeric']==1].index].std()
    w['min - value']=df[w[w['Is numeric']==1].index].min()
    w['Q 5% - value']=df[w[w['Is numeric']==1].index].quantile(0.05, numeric_only=False)
    w['Q 25% - value']=df[w[w['Is numeric']==1].index].quantile(0.25, numeric_only=False)
    w['Q 50% - value']=df[w[w['Is numeric']==1].index].quantile(0.5, numeric_only=False)
    w['Q 75% - value']=df[w[w['Is numeric']==1].index].quantile(0.75, numeric_only=False)
    w['Q 95% - value']=df[w[w['Is numeric']==1].index].quantile(0.95, numeric_only=False)
    w['max - value']=df[w[w['Is numeric']==1].index].max()
    
    return w

def reduce_df_size(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage of dataframe is {start_mem:.2f} MB.')
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            tmp_col = pd.to_numeric(df[col], errors='coerce')
            if tmp_col.isna().all():
                non_null_series = df[col].dropna()  # Create a series without NaN/Null
                total_rows = len(non_null_series)
                unique_count = non_null_series.nunique()
                percentage_unique = (unique_count / total_rows) if total_rows > 0 else 0 
                if percentage_unique <= 0.2 and percentage_unique>0.0:
                    df[col] = df[col].astype('category')
                    df[col]=df[col].cat.add_categories('n/d')
                elif non_null_series.str.contains('%').all():
                    df[col] = df[col].str.replace('%', '').astype(float) / 100

                else: 
                    df[col] = df[col].astype('str')
                    #df[col] = df[col].astype(pd.StringDtype())
            else:
                df[col]=tmp_col

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage after optimization is equal to {end_mem:.2f} MB.')
    print(f'Decreased by {(start_mem - end_mem) / start_mem:.1%}.')
    ## function optimizing dataframe size:
    ## taken from ---->  https://www.kaggle.com/davidcairuz/feature-engineering-lightgbm      
    return df  
    

### utils_plots

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

### utils_stats

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection, HandlerLine2D
import seaborn as sns
from sklearn.utils import check_array, check_consistent_length
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss,auc, roc_auc_score, roc_curve, f1_score,confusion_matrix,classification_report
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
import shap
from optbinning import BinningProcess,OptimalBinning,Scorecard  # kombajn do bucket'owania i scorecard

def create_measures(y,y_pred, proba_cutoff=0.5): 
    score_test = roc_auc_score(y, y_pred)
    auc = score_test
    gini = 2*score_test - 1

    y_pred_class=np.where(y_pred>proba_cutoff,1,0)
    cm = confusion_matrix(y, y_pred_class)    
    TP, FN, FP, TN =cm.ravel()
    print('Proba cut-off:', str(proba_cutoff))
    P=TP+FN #actual positive
    N=FP+TN # acutal negative
    PP=TP+FP #predicted positive
    PN=FN+TN #predicted negative
    
    recall = TP/P #recall,sensitivity, hit rate
    precision = TP/PP #precision
    
    f1_score=2*TP/(2*TP+FP+FN)
    accuracy=(TP+TN)/(P+N)
    balanced_accuracy=(TP/P+TN/N)/2
    fPositiveR=FP/N
    fOmissionR=FN/PN
    d = {'AUC': [round(score_test,4)]
         , 'GINI': [round(gini,4)]
         , 'Accuracy': [round(accuracy,4)]
         , 'Balanced Accuracy': [round(balanced_accuracy,4)]  
        , 'F1 score': [round(f1_score,4)]        
        , 'Recall': [round(recall,4)]      
        , 'Precision': [round(precision,4)]      
        , 'FPR': [round(fPositiveR,4)]      
        , 'FOR': [round(fOmissionR,4)]      
        }  
    d = pd.DataFrame.from_dict(d)
    return d

def calculating_metrics(X_train, X_test, X_oot, y_train, y_test, y_oot,model):
    train = create_measures(y_train,model.predict_proba(X_train)[:, 1])
    test = create_measures(y_test,model.predict_proba(X_test)[:, 1])
    oot = create_measures(y_oot,model.predict_proba(X_oot)[:, 1]) 
 
    measures =  pd.concat([train,test,oot]).set_index([pd.Index(['TRAIN', 'TEST', 'OOT'])]) 
     
    return measures

def pd_to_score(df, PD_start, offset_start=700, PDO=50 ):
    """ Tworzy technicznego score'a.
    """
    df['PD_init']=np.where(df['PD_init']<0.9999,df['PD_init'],0.9999)
    df['PD_init']=np.where(df['PD_init']>0.0001,df['PD_init'],0.0001)
    df['PD_init']=np.round(df['PD_init'],4)
    
    odds=(1-PD_start)/PD_start
    log_odds=np.log(odds)
    factor=PDO/np.log(2)  # z '-' wtedy najgorszy ma najwiekszego skora
    offset=offset_start-factor*log_odds

    df['score']=np.round(np.log((1/df['PD_init']-1))*factor+offset ,0).astype(int)
    score_min=round(offset_start-factor*np.log(odds)+factor*np.log(1/9999),0)
    score_max=round(offset_start-factor*np.log(odds)+factor*np.log(9999),0)
    print('score min: {0}, score max: {1}'.format(score_min,score_max))
    return score_min, score_max

def score_to_PD_calibration(df, alpha, beta):
    """ konwertuje score na PD dla zadanych parametrów alpa i beta.
    Tworzy kolumne PD_final i wymaga do tego columny score.
    """ 
    df['PD_final']=1/(1+np.exp(alpha*df['score']+beta))

def PD_calibration_from_score(df,ile_grup, pd_ref=0.01, score_offset_start=700,  PDO=50):
    s_min, s_max=pd_to_score(df, pd_ref, score_offset_start, PDO )
    df_agg=df.groupby([pd.qcut(df['score'], ile_grup, labels=False)]
                        ).agg({'dflt_ind': ['sum'],'nondflt_ind': ['sum'],'score': ['mean'],'PD_init': ['mean']})
    df_agg.columns=['_'.join(col).rstrip('_') for col in df_agg.columns.values]
    #df.columns = df.columns.to_flat_index()
    df_agg.score_mean=np.round(df_agg.score_mean,0).astype('int')
    df_agg['odr']=df_agg['dflt_ind_sum']/(df_agg['dflt_ind_sum']+df_agg['nondflt_ind_sum'])
    df_agg['odds']=df_agg['nondflt_ind_sum']/df_agg['dflt_ind_sum']
    df_agg['log_odds']=np.log(df_agg['odds'])

    #kalibracja
    x=np.array(df_agg['score_mean']).reshape((-1, 1))
    y=np.array(df_agg['log_odds'])
    lg = LinearRegression()
    lg.fit(x, y)
    r_sqr = lg.score(x, y)
    alpha = lg.coef_[0]
    beta = lg.intercept_
    
    df['PD_final']=1/(1+np.exp(alpha*df['score']+beta))
    df['log_odds_final']=alpha*df['score']+beta
    df_agg['PD_final']=1/(1+np.exp(alpha*df_agg['score_mean']+beta))
    
    return df_agg, alpha, beta, r_sqr
  
def score_to_rating_bucketing(df, ile_ratingow):
    x=df['score'].values
    y=df['dflt_ind'].values
    optb = OptimalBinning(name='score' 
                          ,dtype="numerical" 
                          ,solver="cp"
                          ,min_n_bins=ile_ratingow
                          ,max_n_bins=ile_ratingow
                          #,max_pvalue =0.2 #mocno wpływa - doczytać dokumentację
                          ,prebinning_method ='cart' # cart / mdlp / quantile / uniform
                          ,min_prebin_size =0.02
                          ,min_bin_size=0.05
                          #,max_bin_size=0.3
                          #,monotonic_trend ='auto_asc_desc'
                          ,gamma=0.051 #ogranicznie koncentracji bucket'ów [regularyzacja]
                          ,outlier_detector ='zscore'
                         )
    optb.fit(x, y)
    display("Wynik optymalnego bucketow'ania:"+ optb.status)

    if optb.status=='OPTIMAL':
        binning_table = optb.binning_table
        btable=binning_table.build()
        binning_table.plot(metric="woe")
        #binning_table.plot(metric="event_rate")
        display(btable)
        opt_splits=list(optb.splits)
        opt_splits.insert(0, -np.inf)
        opt_splits.append(np.inf)
        display("Optymalne podziały: ", optb.splits)
        optb.binning_table.analysis()    

        df['PD_rating']= (optb.transform(df['score'].values, metric="indices").astype(str) )
        df['PD_rating']='R'+df['PD_rating']
        df_agg=df.groupby(df['PD_rating']).agg({'score': ['min','max','mean'],'PD_final': ['min','max','mean']})
        display(df_agg)
        return opt_splits
    else:
        return None    
    
def score_to_rating_assign(df, score_buckets, stats=False):
    df['PD_rating']='R'+pd.cut(df['score'], score_buckets, labels=False).astype(str)
    if stats==True:
        df_agg=df.groupby(df['PD_rating']).agg({'score': ['min','max','mean'],'PD_final': ['min','max','mean']})       
        display(df_agg)
    
def importances_calc(xgbModel,X_test, y_test, permutation_repeats=60):

    importance_metrics=['gain','total_gain','weight', 'cover', 'total_cover']
    xgbBestImportance=pd.DataFrame(index=X_test.columns)

    for importance in importance_metrics:

        feature_importance=xgbModel.get_booster().get_score(importance_type=importance)
        f_importance=pd.DataFrame.from_dict(feature_importance, orient='index',columns=[importance])
        f_importance=f_importance/f_importance.sum()
        xgbBestImportance.loc[f_importance.index,importance]=f_importance.iloc[:,0]

    perm_importance = permutation_importance(xgbModel, X_test, y_test
                                             , n_repeats=permutation_repeats
                                             , random_state=0)
    xgbBestImportance['permutation']=perm_importance.importances_mean 

    explainer = shap.Explainer(xgbModel,feature_names=np.array(X_test.columns))
    shap_values = explainer(np.ascontiguousarray(X_test))
    shap_importance = shap_values.abs.mean(0).values
    xgbBestImportance['shap_import']=shap_importance

    xgbBestImportance=xgbBestImportance[['permutation','shap_import','gain','total_gain','weight', 'cover','total_cover']]    
    xgbBestImportanceRanks=xgbBestImportance.rank(axis='index', ascending=False)#.astype('int8')

    xgbBestImportanceRanks.sort_values(by='permutation',ascending=True,inplace=True) 
    xgbBestImportanceRanks['Average gain/shap']=xgbBestImportanceRanks.loc[:,['gain','shap_import']].mean(axis=1).astype(int)
    xgbBestImportanceRanks=xgbBestImportanceRanks[['permutation','Average gain/shap','shap_import','gain','total_gain','weight', 'cover','total_cover']]    
     
    shap.plots.beeswarm(shap_values, max_display=45,plot_size=None)
    sns.heatmap(xgbBestImportanceRanks, annot=True)
    
    for col in xgbBestImportance.columns:
        xgbBestImportance.sort_values(by=col,ascending=True,inplace=True)
        fig, ax = plt.subplots()
        list(xgbBestImportance.index)
        ax.stem( list(xgbBestImportance.index),xgbBestImportance.loc[:,col],orientation='horizontal',linefmt=':')
        ax.set_title("Feature Importance: "+col)

    xgbBestImportance.sort_values(by='permutation',ascending=False,inplace=True)
    
    #display(xgbBestImportance)
    
    return xgbBestImportance,xgbBestImportanceRanks
    
    
def make_confusion_matrix(y_test, y_test_pred, proba_cutoff, title=None):
    
    y_pred=np.where(y_test_pred>proba_cutoff,1,0)
    display("proba cut-off (confusion matrix):" + f"{proba_cutoff:.2%}")
    #print(classification_report(y_test, y_pred))
    cf = confusion_matrix(y_test, y_pred)
    group_labels = ['TRUE POSITIVE (TP) \nhit\n',
                    'FALSE NEGATIVE (FN) \n[type 2 error] \nmiss \nunderestimation\n',
                    'FALSE POSITIVE (FP) \n[type 1 error] \nfalse alarm \noverstimation\n',
                    'TRUE NEGATIVE (TN) \ncorrect rejection\n']
    xcategories = ['Positive (PP)', 'Negative (PN)']
    ycategories = ['Positive (P)', 'Negative (N)']
    group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    
    TP, FN, FP, TN =cf.ravel()
    P=TP+FN #actual positive
    N=FP+TN # acutal negative
    PP=TP+FP #predicted positive
    PN=FN+TN #predicted negative
    
    TPR=TP/P #recall,sensitivity, hit rate
    FNR=FN/P #miss rate
    FPR=FP/N #false alarm proba, fall-out
    TNR=TN/N #specifity, selevtivity
    
    PPV=TP/PP #precision
    FOR=FN/PN 
    FDR=FP/PP
    NPV=TN/PN

    F1_score=2*TP/(2*TP+FP+FN)
    Accuracy=(TP+TN)/(P+N)
    Balanced_Accuracy=(TPR+TNR)/2
    


    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])
    
    stats_text = """\nSample: """+title+"""\nAccuracy: {:23.2%}\nBalanced Accuracy: {:7.2%}\n\nF1 Score: {:23.2%}\nPrecision: {:22.2%}\nRecall: {:27.2%}""".format(Accuracy,Balanced_Accuracy,F1_score,PPV,TPR)

    group_labels2 = ['True positive rate (TPR) \nRECALL, sensitivity\nTPR = TP / P\n',
                     'False negative rate (FNR) \nmiss rate \nFNR = FN / P\n',
                     'False positive rate (FPR) \nFALLOUT, false alarm proba\nFPR = FP / N\n',
                     'True negative rate (TNR) \nspecificity\nTNR = TN / N\n']
    group_counts2 = ["{0:0.2%}\n".format(value) for value in [TPR,FNR,FPR,TNR]]
    box_labels2 = [f"{v1}{v2}".strip() for v1, v2 in zip(group_labels2,group_counts2)]
    box_labels2 = np.asarray(box_labels2).reshape(cf.shape[0],cf.shape[1])    
    
    group_labels3 = ['Positive predictive value (PPV) \nPRECISION \nPPV = TP / PP\n',
                     'False omission rate (FOR) \nFOR = FN / PN\n',
                     'False discovery rate (FDR) \nFDR = FP / PP\n',
                     'Negative predictive value (NPV)  \nNPV = TN / PN\n']
    group_counts3 = ["{0:0.2%}\n".format(value) for value in [PPV,FOR,FDR,NPV]]
    box_labels3 = [f"{v1}{v2}".strip() for v1, v2 in zip(group_labels3,group_counts3)]
    box_labels3 = np.asarray(box_labels3).reshape(cf.shape[0],cf.shape[1])              

    df = pd.DataFrame({'Straight': [i for i in range(10)],'Square': [i * i for i in range(10)]})    
    plt.rcParams["figure.figsize"] = [11, 7]
    plt.rcParams["figure.autolayout"] = True
    
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    g1=sns.heatmap(cf,annot=box_labels,fmt="",cmap='Blues',cbar=True,xticklabels=xcategories,yticklabels=ycategories,ax=ax1)
    g1.set_ylabel('ACTUAL')
    g1.set_xlabel('PREDICTED')
    g1.set_title('Confiusion matrix') 
    ax2 = fig.add_subplot(2,2,2)
    ax2.text(0.25, 0.4, stats_text, fontsize=16)
    ax2.axis('off')

    ax3 = fig.add_subplot(2,2,3)  
    metrics1=pd.DataFrame({'col1': [10,0], 'col2': [0,10]})
    g3=sns.heatmap(metrics1,annot=box_labels2,fmt="",cmap='RdYlGn',cbar=False,xticklabels=xcategories,yticklabels=ycategories,ax=ax3)
    g3.set_ylabel('ACTUAL')
    g3.set_xlabel('PREDICTED')    
    g3.set_title('Metrics') 
    ax4 = fig.add_subplot(2,2,4)  
    metrics1=pd.DataFrame({'col1': [10,0], 'col2': [0,10]})
    g4=sns.heatmap(metrics1,annot=box_labels3,fmt="",cmap='RdYlGn',cbar=False,xticklabels=xcategories,yticklabels=ycategories,ax=ax4)
    g4.set_ylabel('ACTUAL')
    g4.set_xlabel('PREDICTED')       
    plt.show()
