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