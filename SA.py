# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 12:19:42 2021

@author: cbn978
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing, linear_model
import scipy.io as sio
import os
import seaborn as sns               


# =============================================================================
# Note: training of the neural network meta-models was carried out based on a 
# modified code from https://github.com/gsi-lab/easyGSA 
# 
# Al, R., Behera, C.R., Zubov, A., Gernaey, K.V., Sin, G., 2019. Meta-modeling
# based efficient global sensitivity analysis for wastewater treatment plants 
#  An application to the BSM2 model. Computers & Chemical Engineering 127, 
# 233246. https://doi.org/10.1016/j.compchemeng.2019.05.015
# =============================================================================

# =============================================================================
# import features and outputs. final table with parameters and outputs (
# =============================================================================
path = r'...'

N=4939    
X={'CA':{},
   'CT':{}}
for nsyst, syst in enumerate(['CT','CA']):#
    for col in ['O','I']:# 
        for where in ['field','drain']:
            X['%s_%s_%s'%(syst,where,col)]=pd.read_csv(path+'\\%s\\'%syst+'UA_SA_parameters_outputs_%d_%s_%s_%s_waterbalance.csv'%(N,syst,where,col))

# =============================================================================
# linear regression for SRCs computation
# =============================================================================
for nsyst, syst in enumerate(['CT','CA']):#
    for col in ['O','I']:# 
        for where in ['drain']:
            X['%s_%s_%s'%(syst,where,col)]['leaching']=(X['%s_%s_%s'%(syst,where,col)]['Out-pest_total_0_30']/500*100+
                                                        X['%s_%s_%s'%(syst,where,col)]['drain_pest_biopore'])
          
parameters=['koc', 'dt50', 'Dp', 'Kmp', 'ddensity', 'Kmm', 'Si', 'f_exch',
       'alpha', 'w_capacity', 'k', 'BD1', 'BD2', 'BD3', 'Ks1', 'Ks2', 
       'Ks3', 'precipitation_total','soil_WC_WW1', 'soil_WC_SB1', 'soil_WC_WW2', 
       'soil_WC_SB2', 'soil_WC_WR', 'LAI_SB1', 'LAI_SB2']
                
SRCs={'pest_decompose_0_3half':{}}    
 
SRC_df=pd.DataFrame() 
for nsyst, syst in enumerate(['CA','CT']):#
    for col in ['O','I']:# 
        for where in ['drain']:
            SA_parameters=X['%s_%s_%s'%(syst,where,col)][parameters]
            scaler = preprocessing.StandardScaler().fit(SA_parameters.values)
            SA_parameters_stdz = scaler.transform(SA_parameters.values)
            
            for y in ['pest_decompose_0_3half']: #'leaching',
                reg=linear_model.LinearRegression().fit(SA_parameters_stdz, X['%s_%s_%s'%(syst,where,col)][y])
                R2=reg.score(SA_parameters_stdz, X['%s_%s_%s'%(syst,where,col)][y])
                from scipy.stats import pearsonr
                pearsonr(np.dot(SA_parameters_stdz, reg.coef_)+reg.intercept_, X['%s_%s_%s'%(syst,where,col)][y].values)[0]**2
                SRC=reg.coef_*np.std(SA_parameters_stdz, axis=0)/np.std(X['%s_%s_%s'%(syst,where,col)][y].values)
                
                print(syst,col,where)
                print(np.array(parameters)[np.argsort(np.abs(SRC))])
                
                SRCs[y]['%s_%s_%s'%(syst,where,col)]=SRC
                syst_list=[syst for i in range(len(SA_parameters.columns))]
                col_list=[col for i in range(len(SA_parameters.columns))]
                SRC_df=pd.concat([SRC_df,pd.DataFrame({'Parameter':SA_parameters.columns,
                             'System': syst_list,
                             'Column': col_list,
                             'SRC': SRC})], axis=0)
                
g=sns.catplot(x='Parameter', y='SRC', hue="System",data=SRC_df, kind="bar", orient="v", 
              height=10, legend=False, palette=sns.color_palette(['mediumseagreen', 'dodgerblue']), ci=None)
g.set_xticklabels(fontsize=20,rotation=90)
g.set_ylabels(label='SRC',fontsize=20)
g.set_xlabels(label='',fontsize=5)
ylab=[-0.6,-0.4,-0.2,0, 0.2,0.4,0.6]
plt.ylim(bottom=-0.7,top=0.6)
plt.yticks(ylab,fontsize=20)
g.set_yticklabels(ylab,fontsize=20)
plt.tight_layout()
plt.legend(loc='upper right')
plt.legend(fontsize=20,  title_fontsize=20)
plt.savefig(path+'\\SA\\'+'Coeff_linear_model_%s.png'%(y))
plt.savefig(path+'\\SA\\'+'Coeff_linear_model_%s.pdf'%(y))

# =============================================================================
# Prepare matrix for SA in MatLab
# =============================================================================
parameters_si=['koc', 'dt50', 'Dp', 'Kmp', 'ddensity', 'Kmm', 'Si', 'f_exch',
       'alpha', 'w_capacity', 'k', 'BD1', 'BD2', 'BD3', 'pred_FC1', 'Ks1', 'pred_FC2', 'Ks2', 
       'pred_FC3', 'Ks3', 'precipitation_total','soil_WC_WW1', 'soil_WC_SB1', 'soil_WC_WW2', 
       'soil_WC_SB2', 'soil_WC_WR', 'LAI_WW1', 'LAI_SB1', 'LAI_WW2', 'LAI_SB2', 'LAI_WR','pest_decompose_0_3half', 'leaching'] 

for nsyst, syst in enumerate(['CT','CA']):#
    for col in ['O','I']:# 
        for where in ['field','drain']:
            correlations=X['%s_%s_%s'%(syst,where,col)][parameters_si].corr()   
            correlations.to_csv(path+'\\'+'corr_matrix_parameters_%s_%s_%s.csv'%(syst,where,col)) 
## BD in layers 1 and 2 highly correlated to pred FC: choose BD because measured

parameters_si=['koc', 'dt50', 'Dp', 'Kmp', 'ddensity', 'Kmm', 'Si', 'f_exch',
       'alpha', 'w_capacity', 'k', 'BD1', 'BD2', 'BD3', 'Ks1', 'Ks2', 
       'Ks3', 'precipitation_total','soil_WC_WW1', 'soil_WC_SB1', 'soil_WC_WW2', 
       'soil_WC_SB2', 'soil_WC_WR', 'LAI_SB1', 'LAI_SB2', 'pest_decompose_0_3half', 'leaching'] 
## removed 'Thetas1','Ks1', 'Thetas2','Ks2', 'Thetas3','Ks3','predfc1' to 3 due to correlations

for nsyst, syst in enumerate(['CT','CA']):#
    for col in ['O' ,'I']:#
        for where in ['drain']: #'field',
            dataset_matlab=X['%s_%s_%s'%(syst,where,col)][parameters_si]
            sio.savemat(os.path.join(path,'%s_%s_%s.mat'%(syst,where,col)), {name: col.values for name, col in dataset_matlab.items()})
            X['%s_%s_%s'%(syst,where,col)][parameters_si].to_csv(path+'\\%s\\'%syst+'%s_%s_%s.csv'%(syst,where,col),
            index=False, index_label=None)
            
# =============================================================================
# plot Sis and STis
# =============================================================================
sti=pd.read_csv((path+'\\SA\\'+'Si_STi.csv'), sep=';')

### Si or Sti for leaching
for index in ['STi', 'Si']:
    g=sns.catplot(x='Parameter', y=index.lower(), hue="System",data=sti, kind="bar", orient="v", col='Column',
                  height=10, legend=False, palette=sns.color_palette(['mediumseagreen', 'dodgerblue']))
    g.set_xticklabels(fontsize=20,rotation=90)
    g.set_ylabels(label='%s'%index,fontsize=20)
    g.set_xlabels(label='',fontsize=5)
    if index == 'Si':
        maxx=0.6
        ylab=[0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    else:
        maxx=0.9
        ylab=[0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    plt.ylim(bottom=0,top=maxx)
    plt.yticks(ylab,fontsize=20)
    g.set_yticklabels(ylab,fontsize=20)
    g.set_titles("{col_name}",size=20)
    plt.tight_layout()
    plt.legend(loc='upper left')
    plt.legend(fontsize=20,  title_fontsize=20)
    plt.savefig((path+'\\SA\\'+'SA_%s_leaching.png'%index))