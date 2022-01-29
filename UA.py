# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:10:51 2020

@author: cbn978
"""
import pandas as pd
import numpy as np
import sys
from matplotlib import pyplot as plt
import scipy.optimize as spo
from scipy.stats import norm, uniform, randint
from pandas.plotting import scatter_matrix
from collections import defaultdict
import os
from os import path

from Daisy import DaisyDlf, DaisyModel, run_many
from hydraulic import pF2h, h2pF, M_vG, HYPRES

from pyDOE import lhs


outputs_folder_path = r'...'
N=5050 

# =============================================================================
# run Rosetta with texture, BD and OM content inputs (N=5050)
# =============================================================================
P_norm=defaultdict()
for syst in ['CA','CT']:
    for col in ['O','I']:
        P_norm['%s_%s'%(syst,col)]=pd.read_csv(outputs_folder_path+'\\%s\\'%syst+'UA_parameters_%d_%s_%s.csv'%(N,syst,col))


rosetta_inputs={'CA_O':P_norm['CA_O'][['BD1','OM1','BD2','OM2','BD3','OM3','Clay','Silt_USDA','Sand_USDA']],
                'CA_I':P_norm['CA_I'][['BD1','OM1','BD2','OM2','BD3','OM3','Clay','Silt_USDA','Sand_USDA']],
                'CT_O':P_norm['CT_O'][['BD1','OM1','BD2','OM2','BD3','OM3','Clay','Silt_USDA','Sand_USDA']],
                'CT_I':P_norm['CT_I'][['BD1','OM1','BD2','OM2','BD3','OM3','Clay','Silt_USDA','Sand_USDA']]}
 
depths={'CA':{1:[0,3.5],2:[3.5,25],3:[25,30]},
        'CT':{1:[0,5],2:[5,25],3:[25,30]}}

rosetta_outputs_path=r'...'

for syst in ['CA','CT']:
    for col in ['O','I']:
        for i in [1,2,3]:      
            inputs=open(rosetta_outputs_path+'\inputs_%s_%s_%d.txt'%(syst,col,i), mode='w') 
            inputs.write('sam_id\tped_id\thz_tp\thz_bt\tsand_tot\tsilt_tot\tclay_tot\tdb_13b\tOC\tw3cld\tw15ad\n') 
            inputs.write('sam_id\tped_id\thz_tp\thz_bt\tsand_tot\tsilt_tot\tclay_tot\tdb_13b\tOC\tw3cld\tw15ad\n')
            for n_sample in range(N):
                inputs.write('%d\t%s%d%d\t%.1f\t%.1f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t-9.9\t-9.9\n'%(
                        n_sample,syst,i,n_sample,depths[syst][i][0],depths[syst][i][1],
                        np.round(rosetta_inputs['%s_%s'%(syst,col)]['Sand_USDA'].values[n_sample],2),
                        np.round(rosetta_inputs['%s_%s'%(syst,col)]['Silt_USDA'].values[n_sample],2),
                        np.round(100.00-np.round(rosetta_inputs['%s_%s'%(syst,col)]['Sand_USDA'].values[n_sample],2)-np.round(rosetta_inputs['%s_%s'%(syst,col)]['Silt_USDA'].values[n_sample],2),2), #rosetta_inputs['%s_%s'%(syst,col)]['Clay'].values[n_sample],
                        np.round(rosetta_inputs['%s_%s'%(syst,col)]['BD%d'%i].values[n_sample],2),
                        rosetta_inputs['%s_%s'%(syst,col)]['OM%d'%i].values[n_sample]/1.72,
                        ))
            inputs.close()

# from Rosetta
for syst in ['CA','CT']:#
    for col in ['O','I']:
        for i in [1,2,3]: 
            os.system('cmd /C "... &"'+
                      '"Rosetta WEIGHTS 3 inputs_%s_%s_%d.txt outputs_%s_%s_%d.txt"'%(syst,col,i,syst,col,i))             


# =============================================================================
# common parameters to CA and CT
# =============================================================================
unif_parameters_names={'pesticide':["koc", "dt50", 'Dp','Kmp'], 
                 'biopore':['ddensity'],
                 'weather':['year'],
                 'soil':['Kmm'], 
                 'mulch':['w_capacity','k','Si','f_exch','alpha']} 

unif_parameters={'pesticide':{1:[np.log10(10), 0.5, 0, 0], 
                              2:[np.log10(10**4), 300, 1.5, 350]}, 
                 'biopore':{1:[0.1],
                            2:[14]},
                 'weather':{1:[1962],
                            2:[4954]},
                  'soil':{1:[0], 
                          2:[350]}, 
                 'mulch':{1:[5.59, 0.097, 1, 0, 0.25], 
                          2:[1.10, 0.027, 99, 1, 2]}}     

# =============================================================================
# sample x times in the common parameter spaces
# =============================================================================
pp=(len(unif_parameters_names['pesticide'])+len(unif_parameters_names['biopore'])+len(unif_parameters_names['weather'])+
   len(unif_parameters_names['soil'])+len(unif_parameters_names['mulch'])) 

Xp=lhs(pp, samples=N) # sample with Latin Hypercube Sampling method
X_unif=np.zeros([1,N])
other_parameters=np.zeros([1,N])
other_parameters_names, X_unif_names=[],[]
for k in unif_parameters_names.keys():
    for p, pname in enumerate(unif_parameters_names[k]):
        if pname == 'year':
            X_unif_names.append(unif_parameters_names[k][p]) 
            i=len(X_unif_names)-1
            X_unif = np.vstack([X_unif, randint.ppf(q=Xp[:,i],low=unif_parameters[k][1][p], high=unif_parameters[k][2][p]+1)]) 
        elif pname=="k" or pname== "w_capacity":
            other_parameters_names.append(unif_parameters_names[k][p])
            other_parameters = np.vstack([other_parameters,np.random.normal(loc=unif_parameters[k][1][p], scale=unif_parameters[k][2][p], size=N)])   
        else:
            X_unif_names.append(unif_parameters_names[k][p]) 
            i=len(X_unif_names)-1
            X_unif = np.vstack([X_unif, uniform.ppf(q=Xp[:,i],loc=unif_parameters[k][1][p],scale=unif_parameters[k][2][p]-unif_parameters[k][1][p])]) # [loc, loc + scale]

other_parameters=np.delete(other_parameters, (0), axis=0)        
X_unif=np.delete(X_unif, (0), axis=0)        

P_unif=pd.DataFrame(data=X_unif.T, columns=X_unif_names)
P_n=pd.DataFrame(data=other_parameters.T, columns=other_parameters_names)
P=pd.concat([P_unif,P_n],axis=1)
P['koc']=pow(10,P['koc'].values)
P.to_csv(outputs_folder_path+'\\'+'Common_parameters_distr_%d.csv'%N)

# =============================================================================
# make final datasets with specific and common soil parameters 
# =============================================================================
outputs_folder_path=r'...'
X_final={'CA':{},
   'CT':{}}

for nsyst, syst in enumerate(['CT','CA']):#
    for col in ['O','I']:#   
        X_final[syst][col]=pd.concat([P,P_norm['%s_%s'%(syst,col)][['BD1','BD2','BD3',
        'pred_FC1', 'Thetas1', 'Thetar1', 'Alpha1', 'N1', 'L1', 'Ks1',
        'pred_FC2', 'Thetas2', 'Thetar2', 'Alpha2', 'N2', 'L2', 'Ks2',
        'pred_FC3', 'Thetas3', 'Thetar3', 'Alpha3', 'N3', 'L3', 'Ks3']]], axis=1)
        X_final[syst][col].to_csv(outputs_folder_path+'\\%s\\'%syst+'UA_parameters_%d_%s_%s_final.csv'%(N,syst,col), index=False, index_label=None) 
        X_final[syst][col]=pd.read_csv(outputs_folder_path+'\\%s\\'%syst+'UA_parameters_%d_%s_%s_final.csv'%(N,syst,col), engine='python')

# =============================================================================
# parameterize Daisy             
# =============================================================================                     
for n_sample in np.arange(0,N): 
    for nsyst, syst in enumerate(['CA','CT']):#
        # create a directory
        os.mkdir(outputs_folder_path+'\\%s'%syst+'\\sim'+'\\%s_%d'%(syst,n_sample))
        soil_param=DaisyModel(outputs_folder_path+'\\%s'%syst+'\\sim'+'\\soil.dai')
        for col in ['O','I']:# 
            # pesticide parameters
            pest_param=DaisyModel(outputs_folder_path+'\\%s'%syst+'\\sim'+'\\pesticides.dai')
            new_value_dt50=X_final[syst][col]['dt50'].values[n_sample]
            pest_param.Input['defchemical'][8].setvalue(np.round(new_value_dt50,1)) # DT50 soil
            pest_param.Input['defchemical'][9].setvalue(np.round(new_value_dt50,1)) # DT50 surface
            pest_param.Input['defchemical'][10].setvalue(np.round(new_value_dt50,1)) # DT50 mulch

            new_value_kfoc=X_final[syst][col]['koc'].values[n_sample]
            pest_param.Input['defchemical']['adsorption'][0].setvalue(np.round(new_value_kfoc,1)) # Kfoc
            # new_value_m=X_final[syst][col]['m'].values[n_sample]
            # pest_param.Input['defchemical']['adsorption'][1].setvalue(np.round(new_value_m,2)) # m
            
            new_value_dp=X_final[syst][col]['Dp'].values[n_sample]
            pest_param.Input['defchemical'][1].setvalue(np.round(new_value_dp,2)) # Dp
            
            new_value_kmp=X_final[syst][col]['Kmp'].values[n_sample]*X_final[syst][col]['BD1'].values[n_sample]*10**-6
            pest_param.Input['defchemical'][4].setvalue(np.round(new_value_kmp,8)) # Kmp
            pest_param.Input['defchemical'][5].setvalue(np.round(new_value_kmp,8))
            
            pest_param.save_as(outputs_folder_path+'\\%s'%syst+'\\sim'+'\\%s_%d'%(syst,n_sample)+'\\pesticides_%d.dai'%(n_sample))
            
            for layer in [1,2,3]:
                # soil hydraulic parameters
                new_thetas=X_final[syst][col]['Thetas%d'%layer].values[n_sample]*100 # to %
                new_thetar=X_final[syst][col]['Thetar%d'%layer].values[n_sample]*100
                new_alpha=X_final[syst][col]['Alpha%d'%layer].values[n_sample]
                new_n=X_final[syst][col]['N%d'%layer].values[n_sample]
                new_Ks=X_final[syst][col]['Ks%d'%layer].values[n_sample]/24*10 # to mm/h
                new_l=X_final[syst][col]['L%d'%layer].values[n_sample]
                new_value_rho_b=X_final[syst][col]['BD%d'%layer].values[n_sample]            

                lay=layer
                if syst == 'CT':
                    lay+=6
                if col == 'I':
                   lay+=3 

                soil_param.Input['defhorizon'][lay-1]['dry_bulk_density'].setvalue(np.round(new_value_rho_b,2))
                soil_param.Input['defhorizon'][lay-1]['hydraulic'][0].setvalue(np.round(new_thetas,2))
                soil_param.Input['defhorizon'][lay-1]['hydraulic'][1].setvalue(np.round(new_thetar,2))
                soil_param.Input['defhorizon'][lay-1]['hydraulic'][2].setvalue(np.round(new_alpha,4))
                soil_param.Input['defhorizon'][lay-1]['hydraulic'][3].setvalue(np.round(new_n,4))
                soil_param.Input['defhorizon'][lay-1]['hydraulic'][4].setvalue(np.round(new_Ks,2))
                soil_param.Input['defhorizon'][lay-1]['hydraulic'][5].setvalue(np.round(new_l,4))
                
        # soil biopore density parameter
        new_value_biop=X_final[syst][col]['ddensity'].values[n_sample]
        if syst == 'CT':
            place1, place2 = 6, 8
            new_value_biop_ct=new_value_biop/2     
            soil_param.Input['defbiopore'][place1]['drain']['density'].setvalue(np.round(new_value_biop_ct,2))
            soil_param.Input['defbiopore'][place2]['drain']['density'].setvalue(np.round(new_value_biop_ct,2))
        
        else:
            place=7
            soil_param.Input['defbiopore'][place]['drain']['density'].setvalue(np.round(new_value_biop,2))
                
        soil_param.save_as(outputs_folder_path+'\\%s'%syst+'\\sim''\\%s_%d'%(syst,n_sample)+'\\soil_%d.dai'%(n_sample))
        
        # weather parameters
        random_start_year=int(X_final[syst][col]['year'].values[n_sample])    
        set_up_file=DaisyModel(outputs_folder_path+'\\%s'%syst+'\\sim'+'\\setup_%s.dai'%(syst))
        set_up_file.Input['input'][0].setvalue('file "soil_%d.dai"'%(n_sample))
        set_up_file.Input['input'][1].setvalue('file "pesticides_%d.dai"'%(n_sample))
        set_up_file.Input['defprogram'][1].setvalue(random_start_year)
        set_up_file.Input['defprogram'][2][0].setvalue(random_start_year+3)
        set_up_file.Input['defprogram'][3].setvalue(random_start_year+8)
        
        # mulch parameters 
        new_w_capacity=X_final[syst][col]['w_capacity'].values[n_sample]
        set_up_file.Input['deflitter']['water_capacity'].setvalue(np.round(new_w_capacity,2)) # w_capacity
        
        new_k=X_final[syst][col]['k'].values[n_sample]
        set_up_file.Input['deflitter']['retention']['exp']['k'].setvalue(np.round(new_k,3)) # k

        new_si=X_final[syst][col]['Si'].values[n_sample]
        set_up_file.Input['deflitter']['Si'].setvalue(np.round(new_si,0)) # Si
        
        new_fe=X_final[syst][col]['f_exch'].values[n_sample]
        set_up_file.Input['deflitter']['factor_exch'].setvalue(np.round(new_fe,4)) # f_exch
        
        new_alpha=X_final[syst][col]['alpha'].values[n_sample]
        set_up_file.Input['deflitter']['alpha'].setvalue(np.round(new_alpha,2)) # alpha
        
        new_value_kmm=X_final[syst][col]['Kmm'].values[n_sample]*X_final[syst][col]['BD1'].values[n_sample]*10**-6
        set_up_file.Input['deflitter']['decompose_SMB_KM'].setvalue(np.round(new_value_kmm,8)) # Kmm
        set_up_file.Input['deflitter']['SMB_ref'].setvalue(np.round(new_value_kmm,8))
        
        set_up_file.save_as(outputs_folder_path+'\\%s'%syst+'\\sim'+'\\%s_%d'%(syst,n_sample)+'\\setup_%s_%d.dai'%(syst,n_sample))
    
        
# =============================================================================
# run Daisy in parallel
# =============================================================================
weather_series=DaisyDlf(outputs_folder_path+'\\CA\\'+'Control_hourly_daisy.dwf', 
              FixedTimeStep=True).Data

daisy_models = []
for n_sample in np.arange(0,N):
    for syst in ['CA','CT' ]:#
        daisy_models.append(outputs_folder_path+'\\%s\\sim\\%s_%d'%(syst,syst,n_sample)+'\\setup_%s_%d.dai'%(syst,n_sample))

DaisyModel.path_to_daisy_executable =  r'C:\Program Files\Daisy 6.27\bin\Daisy.exe'
run_state=run_many(daisy_models, NumberOfProcesses=12)


# =============================================================================
# check for crashed simulations, save pesticide load, decomposition, LAI and SWC
# =============================================================================
crashed_simulations=defaultdict(list)
N=5050
apps=['WW1','SB1','WW2','SB2','WR']
inputs_folder_path=r'...'
outputs_folder_path=r'C:\Users\cbn978\OneDrive - University of Copenhagen\PhD\GMSR\Daisy\UA_SA'
for syst in ['CA','CT']: #
    for col in ['drain']:
        for where in ['O','I']:
            print(syst,col,where)
            pest_load=open(outputs_folder_path+'\\%s\\'%syst+'sim_endpoints_factors_waterbalance_%s_%s.txt'%(col,where), mode='w')
            pest_load.write('n_sample\tdrain_pest_matrix\tdrain_pest_biopore\tprecipitation_total\tdrain_water_matrix\tdrain_water_biopore\t'+
                            'potential_evapotranspiration\tactual_evapotranspiration\tsoil_evaporation\tponding_evaporation\tmulch_evaporation\t'+
                            'potential_transpiration\tactual_transpiration\tinfiltration_total\tmatrix_infiltration\tbiopore_infiltration\t'+
                            'percolation_total_0_30\tmatrix_percolation_0_30\tbiopore_percolation_0_30\tpercolation_total_0_200\t'+
                            'matrix_to_biopore_0_3half\tmatrix_to_biopore_3half_30\ts_ponding\t'+
                            'pest_content_0_3half\tpest_content_3half_30\tIn-pest_total\tIn-pest_matrix\tIn-pest_biopore\t'+
                            'Out-pest_total_0_30\tOut-pest_matrix_0_30\tOut-pest_biopore_0_30\t'+
                            'pest_matrix_to_biopore_0_3half\tpest_matrix_to_biopore_3half_30\t'+
                            'pest_decompose_0_3half\tpest_decompose_3half_30\t'+
                            'soil_WC_WW1\tsoil_WC_SB1\tsoil_WC_WW2\tsoil_WC_SB2\tsoil_WC_WR\t'+
                            'LAI_WW1\tLAI_SB1\tLAI_WW2\tLAI_SB2\tLAI_WR\tsurface_T\twater_potential_05\t'+
                            'water_potential_125\twater_potential_245\n')
            
            for n_sample in np.arange(0,N):
                # print(n_sample)
                if not path.exists(inputs_folder_path+'\\%s\\%s_%d'%(syst,syst,n_sample)+'\\UA_%s-harvest.dlf'%syst):
                    crashed_simulations[syst+'_%s'%col+'_%s'%where].append(n_sample)
                else:
                    if path.exists(inputs_folder_path+'\\%s\\%s_%d'%(syst,syst,n_sample)+'\\UA_%s-%s_%s_%s_drain_pest_test.dlf'%(syst,syst,col,where)):
                        sim=DaisyDlf(inputs_folder_path+'\\%s\\%s_%d'%(syst,syst,n_sample)+'\\UA_%s-%s_%s_%s_drain_pest_test.dlf'%(syst,syst,col,where))
                    harvest=DaisyDlf(inputs_folder_path+'\\%s\\%s_%d'%(syst,syst,n_sample)+'\\UA_%s-harvest.dlf'%syst).Data
                    if harvest.size == 0:
                        crashed_simulations[syst+'_%s'%col+'_%s'%where].append(n_sample)
                    elif len(harvest) < 5:
                        crashed_simulations[syst+'_%s'%col+'_%s'%where].append(n_sample)
                    elif not len(sim.Data) == 60:
                        crashed_simulations[syst+'_%s'%col+'_%s'%where].append(n_sample)
                    elif len(sim.Data) == 60:
                        
                        precipitations=sim.Data['Precipitation'].sum()
                                                
                            # evapotranspiration
                        surface=DaisyDlf(inputs_folder_path+'\\%s\\%s_%d'%(syst,syst,n_sample)+'\\UA_%s-%s_%s_%s_surface_water_balance.dlf'%(syst,syst,col,where))
                        evappot=surface.Data['Potential evapotranspiration'].sum()
                        evap=surface.Data['Actual Evapotranspiration'].sum()
                        evap_sw=surface.Data["Evaporation of soil water"].sum()
                        evap_pw=surface.Data["Evaporation from ponded water"].sum()
                        evap_mw=surface.Data["Evaporation from litter"].sum()
                        trsppot=surface.Data["Potential transpiration"].sum()
                        trsp=surface.Data["Actual transpiration"].sum()
                        
                            # ponding, infiltration, percolation, potential and T
                        sw0_35=DaisyDlf(inputs_folder_path+'\\%s\\%s_%d'%(syst,syst,n_sample)+'\\UA_%s-%s_%s_%s_soil-water-0-3half.dlf'%(syst,syst,col,where))    
                        sw35_30=DaisyDlf(inputs_folder_path+'\\%s\\%s_%d'%(syst,syst,n_sample)+'\\UA_%s-%s_%s_%s_soil-water-3half-30.dlf'%(syst,syst,col,where))
                        sw0_30=DaisyDlf(inputs_folder_path+'\\%s\\%s_%d'%(syst,syst,n_sample)+'\\UA_%s-%s_%s_%s_soil-water-0-30.dlf'%(syst,syst,col,where))
                        sw0_200=DaisyDlf(inputs_folder_path+'\\%s\\%s_%d'%(syst,syst,n_sample)+'\\UA_%s-%s_%s_%s_soil-water-0-200.dlf'%(syst,syst,col,where))
                        sw_pot=DaisyDlf(inputs_folder_path+'\\%s\\%s_%d'%(syst,syst,n_sample)+'\\UA_%s-%s_%s_%s_soil_water_potential.dlf'%(syst,syst,col,where))
                        inf_total = sw0_35.Data['Matrix infiltration'].sum() + sw0_35.Data['Biopore infiltration'].sum()
                        soil_inf, biop_inf = sw0_35.Data['Matrix infiltration'].sum(), sw0_35.Data['Biopore infiltration'].sum()
                        drain_biopw0_30 = sw0_30.Data['Biopore drain flow'].sum()
                        drain_soilw0_30 = sw0_30.Data['Matrix direct drain flow'].sum() + sw0_30.Data['Matrix indirect drain flow'].sum()
                        perco_total0_30 = sw0_30.Data["Matrix percolation"].sum() + sw0_30.Data["Biopore percolation"].sum()
                        perco_soil0_30, perco_biop0_30 = sw0_30.Data["Matrix percolation"].sum(), sw0_30.Data["Biopore percolation"].sum()
                        perco_total0_200 = sw0_200.Data["Matrix percolation"].sum() + sw0_200.Data["Biopore percolation"].sum()
                        soil_biop0_35 = sw0_35.Data["Matrix to biopores"].sum()
                        soil_biop35_30 = sw35_30.Data["Matrix to biopores"].sum()                        
                        ponding=surface.Data['Surface ponding'].sum()
                        sw_pot05=np.average(sw_pot.Data['h @ -0.5'].values)
                        sw_pot125=np.average(sw_pot.Data['h @ -12.5'].values)
                        sw_pot245=np.average(sw_pot.Data['h @ -24.5'].values)
                        
                            # soil chemical in, content, biopore and percolation
                        schem0_35=DaisyDlf(inputs_folder_path+'\\%s\\%s_%d'%(syst,syst,n_sample)+'\\UA_%s-%s_%s_%s_soil-pest-0-3half.dlf'%(syst,syst,col,where))    
                        schem35_30=DaisyDlf(inputs_folder_path+'\\%s\\%s_%d'%(syst,syst,n_sample)+'\\UA_%s-%s_%s_%s_soil-pest-3half-30.dlf'%(syst,syst,col,where))    
                        schem0_30=DaisyDlf(inputs_folder_path+'\\%s\\%s_%d'%(syst,syst,n_sample)+'\\UA_%s-%s_%s_%s_soil-pest-0-30.dlf'%(syst,syst,col,where))
                        chem_content0_35, chem_content35_30 = schem0_35.Data["Content"][-1], schem35_30.Data["Content"][-1]
                        in_chem_total0_35 = schem0_35.Data["In-Matrix"].sum() + schem0_35.Data["In-Biopores"].sum()
                        in_chem_soil0_35, in_chem_biop0_35 = schem0_35.Data["In-Matrix"].sum(), schem0_35.Data["In-Biopores"].sum()
                        drain_soilpest0_30 = schem0_30.Data['Drain-Soil'].sum()
                        drain_bioppest0_30 = schem0_30.Data['Drain-Biopores'].sum() + schem0_30.Data['Drain-Biopores-Indirect'].sum()
                        out_chem_total0_30 = schem0_30.Data["Leak-Matrix"].sum() + schem0_30.Data["Leak-Biopores"].sum()
                        out_chem_soil0_30, out_chem_biop0_30 = schem0_30.Data["Leak-Matrix"].sum(), schem0_30.Data["Leak-Biopores"].sum()
                        chem_soil_biop0_35 = schem0_35.Data["Matrix to biopores"].sum()
                        chem_soil_biop35_30 = schem35_30.Data["Matrix to biopores"].sum()
                        decompose35_30 = schem35_30.Data["Decompose"].sum()
                        
                        # decomposition
                        decompose=DaisyDlf(inputs_folder_path+'\\%s\\%s_%d'%(syst,syst,n_sample)+'\\UA_%s-%s_%s_%s_field-pest_test.dlf'%(syst,syst,col,where))
                        decompose0_35=(decompose.Data["Soil Decompose 0-3half"].sum()+ # total pesticide decomposition from mulch, soil surface and first soil layer (0-5 cm))
                                     decompose.Data['Litter Decompose'].sum()+
                                     decompose.Data['Surface Decompose'].sum()) # g/ha
                        
                        # spraying rate
                        sprays=DaisyDlf(inputs_folder_path+'\\%s\\%s_%d\\'%(syst,syst,n_sample)+'UA_%s-%s_%s_%s_pesticides_spray.dlf'%(syst,syst,col,where)).Data  
                        total_sprayed=sprays['pest_test'].sum() # g/ha
                        temp0_30=np.average(sprays['Temp 0-30'].values)
                        idx_sprays=sprays.index[sprays['pest_test'] > 0]
                        water_content, LAI = defaultdict(), defaultdict()
#                        print('this: ',len(idx_sprays), ': should be 5!')
                        for i,idx in enumerate(idx_sprays):
                            water_content[apps[i]]=sprays['Soil matrix water 0-5'][idx]
                            LAI[apps[i]]=sprays['LAI'][idx]

                        pest_load.write('%d\t%.6f\t%.6f\t%.2f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'%(
                                        n_sample, drain_soilpest0_30/total_sprayed*100,
                                        drain_bioppest0_30/total_sprayed*100, precipitations,
                                        drain_soilw0_30, drain_biopw0_30, evappot, evap, evap_sw, evap_pw, evap_mw)+
                                        '\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'%(
                                        trsppot, trsp, inf_total, soil_inf, biop_inf, 
                                        perco_total0_30, perco_soil0_30, perco_biop0_30, perco_total0_200,
                                        soil_biop0_35)+
                                        '\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'%(
                                        soil_biop35_30, ponding, chem_content0_35, 
                                        chem_content35_30, in_chem_total0_35,
                                        in_chem_soil0_35, in_chem_biop0_35, 
                                        out_chem_total0_30, out_chem_soil0_30, out_chem_biop0_30)+
                                        '\t%.3f\t%.3f\t%.3f\t%.3f'%(
                                        # endpoints to be compared between CA and CT 
                                        chem_soil_biop0_35, chem_soil_biop35_30, 
                                        decompose0_35/total_sprayed*100, decompose35_30/total_sprayed*100)+
                                        '\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f'%(
                                                            # factors of endpoints (to add to list of parameters)
                                                            water_content['WW1'], water_content['SB1'],
                                                            water_content['WW2'], water_content['SB2'],
                                                            water_content['WR'], LAI['WW1'], LAI['SB1'],
                                                            LAI['WW2'], LAI['SB2'], LAI['WR'])+
                                        '\t%.2f\t%.2f\t%.2f\t%.2f\n'%(temp0_30,sw_pot05,sw_pot125,sw_pot245))            

            pest_load.close()


## same set of parameters that crashed for each system

crashed_samples_CA=pd.read_csv(r'...'+'\\CA\\'+'crashed_samples.csv', engine='python')
crashed_samples_CT=pd.read_csv(r'...'+'\\CT\\'+'crashed_samples.csv', engine='python')

crashed_simulations=defaultdict()
crashed_simulations['CA']=list(crashed_samples_CA['CA'].values)
crashed_simulations['CT']=list(crashed_samples_CT['CT'].values)

# =============================================================================
# common run samples
# =============================================================================
crashed_CA_for_CT=crashed_samples_CA.loc[~crashed_samples_CA['CA'].isin(crashed_samples_CT['CT'].values)]  
crashed_CT=crashed_simulations['CT']+list(crashed_CA_for_CT['CA'].values)

crashed_CT_for_CA=crashed_samples_CT.loc[~crashed_samples_CT['CT'].isin(crashed_samples_CA['CA'].values)]  
crashed_CA=crashed_simulations['CA']+list(crashed_CT_for_CA['CT'].values)
crashed=pd.DataFrame({'CA':np.sort(crashed_CA),
                       'CT':np.sort(crashed_CT)})

# =============================================================================
# correct sampling matrices
# =============================================================================
X_pest={'CA':{},
   'CT':{}}

for nsyst, syst in enumerate(['CT','CA']):#
    for col in ['O','I']:#   
          X_pest[syst][col]=pd.read_csv(outputs_folder_path+'\\%s\\'%syst+'UA_parameters_%d_%s_%s_final.csv'%(N,syst,col), engine='python')
          X_pest[syst][col]=X_pest[syst][col].loc[~X_pest[syst][col].index.isin(crashed[syst].values)]
          X_pest[syst][col].index=np.arange(0,len(X_pest[syst][col]))
          X_pest[syst][col].to_csv(outputs_folder_path+'\\%s\\'%syst+'UA_parameters_%d_%s_%s_final_corrected.csv'%(len(X_pest[syst][col]),
                                                                                                                   syst,col),
                  index=False, index_label=None)


 
            
# =============================================================================
# Correct outputs
# =============================================================================
X_outputs=defaultdict()
for nsyst, syst in enumerate(['CT','CA']):#
    for where in ['O','I']:# 
        for col in ['field','drain']:
            X_outputs['%s_%s_%s'%(syst,col,where)]=pd.read_csv(outputs_folder_path+'\\%s\\'%syst+'sim_endpoints_factors_waterbalance_%s_%s.txt'%(col,where), engine='python', sep='\t')

for nsyst, syst in enumerate(['CT','CA']):#
    for where in ['O','I']:# 
        for col in ['field','drain']:
            X_outputs['%s_%s_%s'%(syst,col,where)]=X_outputs['%s_%s_%s'%(syst,col,where)].loc[~X_outputs['%s_%s_%s'%(syst,col,where)]['n_sample'].isin(crashed[syst])]
            X_outputs['%s_%s_%s'%(syst,col,where)].index=np.arange(0,len(X_outputs['%s_%s_%s'%(syst,col,where)]))
            

outputs_folder_path_os=r'C:\Users\cbn978\OneDrive - Københavns Universitet\PhD\GMSR\\Daisy\UA_SA'
X_outputs=defaultdict()
for nsyst, syst in enumerate(['CT','CA']):#
    for where in ['O','I']:# 
        for col in ['field','drain']:
            X_outputs['%s_%s_%s'%(syst,col,where)]['Syst']=['%s'%syst for i in range(len(X_outputs['%s_%s_%s'%(syst,col,where)]))]
            X_outputs['%s_%s_%s'%(syst,col,where)].to_csv(outputs_folder_path_os+'\\%s\\'%syst+'%s_%d_outputs_%s_%s_waterbalance.csv'%(syst,len(X_outputs['%s_%s_%s'%(syst,col,where)]),col,where), index=False, index_label=None)  
            X_outputs['%s_%s_%s'%(syst,col,where)]=pd.read_csv(outputs_folder_path_os+'\\%s\\'%syst+'%s_4939_outputs_%s_%s_waterbalance.csv'%(syst,col,where))  
