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
    
