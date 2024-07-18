import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from scipy import stats
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,MinMaxScaler,OneHotEncoder,OrdinalEncoder,LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix , recall_score , precision_score , f1_score, precision_recall_curve, auc

c=(0.48366628618847957, 0.1286467902201389, 0.31317188565991266)

import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('C:\\Users\\abhin\\Downloads\\archive (1)\\weatherAUS.csv')
print(df.info())

sort=df.isna().sum().sort_values(ascending=False)
per=sort*100/df.shape[0]
unique=df.nunique()
note=pd.concat([sort,per,unique,df.dtypes],axis=1)
note.columns=['Null_count','Null_percentage','Unique_values','Data_Type']
print(note)

plt.figure(figsize=(10,6))
sns.barplot(x=note.index,y=note['Null_percentage'],color=c)
plt.xticks(rotation=45)
print(plt.show())

def impute_missing(df):
    loc_unique = df['Location'].unique()
    num_col = df.select_dtypes(exclude='object').columns
    cat_col = df.select_dtypes(include='object').columns

    for col in num_col:
        for loc in loc_unique:
            filt = df['Location'].isin([loc])
            med = df[filt][col].median()
            df.loc[filt, col] = df[filt][col].fillna(med)
    
    for col in cat_col:
        for loc in loc_unique:
            filt = df['Location'].isin([loc])
            if df[filt][col].empty:
                continue  
            mode = df[filt][col].mode()
            if not mode.empty:
                med = mode[0]
                df.loc[filt, col] = df[filt][col].fillna(med)
    
    return df

df=impute_missing(df)
print(df)

sort=df.isna().sum().sort_values(ascending=False)
per=sort*100/df.shape[0]
unique=df.nunique()
note=pd.concat([sort,per,unique,df.dtypes],axis=1)
note.columns=['Null_count','Null_percentage','Unique_values','Data_Type']
print(note)

remaining_nulls=df.isnull().sum().sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=remaining_nulls.index,y=remaining_nulls.values,color=c)
plt.xticks(rotation=45)
plt.show()

df.dropna(subset=['WindGustDir' , 'WindGustSpeed' , 'WindDir9am', 'WindDir3pm' , 'Pressure9am' , 'Pressure3pm' , 'RainToday' ,  'RainTomorrow',
                  'Evaporation','Sunshine', 'Cloud9am' , 'Cloud3pm']
                    , inplace=True  , axis= 0)

df['Date'] = pd.to_datetime(df['Date'] )
df['Day']=df['Date'].dt.day
df['Month']=df['Date'].dt.month
df['year']=df['Date'].dt.year
df.drop('Date',axis=1,inplace=True)

num_col = df.select_dtypes(exclude='object').columns
cat_col = df.select_dtypes(include='object').columns

plt.figure(figsize=(10,6))
co=df['RainTomorrow'].value_counts()/df['RainTomorrow'].count()
sns.barplot(x=co.index,y=co.values,color=c)
print(plt.show())

fig,ax=plt.subplots(5,3,figsize=(20,35))
idx=0
for i in range(5):
    for j in range(3):
        sns.boxplot(ax=ax[i, j], x=df[num_col[idx]],color=c)
        ax[i, j].set_title(num_col[idx])
        idx=idx+1
print(plt.show())


def handle_outliers(df,impute_strategy='median'):
    num_col = df.select_dtypes(exclude='object').columns
    for col in num_col:
        z_scores = np.abs(stats.zscore(df[col]))
        outliers = np.where(z_scores > 2)[0]  

        if len(outliers) == 0:
            continue 
            
        if impute_strategy == 'median':
            imputed_value = df[col].median()
        elif impute_strategy == 'mean':
            imputed_value = df[col].mean()
            
        df.loc[outliers, col] = imputed_value

    return df

fig,ax=plt.subplots(5,3,figsize=(20,35))
idx=0
for i in range(5):
    for j in range(3):
        sns.boxplot(ax=ax[i, j], x=df[num_col[idx]],color=c)
        ax[i, j].set_title(num_col[idx])
        idx=idx+1
print(plt.show())

def handle_outlires_IQR(df):
    num_col = df.select_dtypes(exclude='object').columns
    for col in num_col:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        TQR=1.5*IQR
        outliers = df[ ( df[col] < (Q1 -IQR)) | (df[col] > (Q3 +IQR) ) ][col]
        med_value=df[col].median()
        df[df[col].isin([outliers])][col]=med_value
    return df
df=handle_outlires_IQR(df)

fig,ax=plt.subplots(5,3,figsize=(20,35))
idx=0
for i in range(5):
    for j in range(3):
        sns.kdeplot(ax=ax[i, j], x=df[num_col[idx]],color=c,alpha=0.4,fill=True)
        ax[i, j].set_title(num_col[idx])
        idx=idx+1
print(plt.show())

num_pipeline=Pipeline(steps=[
    ('impute',SimpleImputer(strategy='mean')),
    ('scale',StandardScaler())
]
        
)
num_pipeline
cat_pipeline=Pipeline( steps=[
    ('impute',SimpleImputer(strategy='most_frequent')),
    ('encoder',OrdinalEncoder())
])
cat_pipeline
features=df.drop('RainTomorrow',axis=1)
labels=df['RainTomorrow']
num_col = features.select_dtypes(exclude='object').columns
cat_col = features.select_dtypes(include='object').columns
x_train,x_test,y_train,y_test =train_test_split(features,labels,test_size=0.30,random_state=42)
col_transformer=ColumnTransformer(
    transformers=[('num_pipeline',num_pipeline,num_col)
                ,('cat_pipeline',cat_pipeline,cat_col)
                ]
    , remainder='passthrough',n_jobs=-1

)
col_transformer
rf = RandomForestClassifier( random_state=42)
pipefinal=make_pipeline(col_transformer,rf)
print(pipefinal)

pipefinal.fit(x_train,y_train)
pred=pipefinal.predict(x_test)
print('Accuracy Score :', accuracy_score(y_test, pred) , '\n')
print('Classification Report :', '\n',classification_report(y_test, pred))

