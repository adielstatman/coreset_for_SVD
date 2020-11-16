# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 10:31:45 2019
אני:
תוצאות מעניינות בקורסט חלש נגד סוהלר: במימד קטן 1-3 מנצח אחרי זה מפסיד ואחרי זה שוב מנצח.
מנצח את CNW בקורסט חזק בחלק מהמימדים. קורע אותו בזמן.
מנצח את עלא בקורסט חזק , מפסיד לו בזמן בקורסטים גדולים מאוד.

עלא מנצח יוניפ בקורסט חזק בd גדול וj קרוב לd
@author: statman2
"""


    #return C,delta_C,real_cs
from scipy import io    

import general_SVD_algs as gsc 
import scipy
import scipy.sparse as ssp
from scipy.sparse import linalg
from scipy.sparse import coo_matrix as CM
from scipy.sparse import csr_matrix as SM

#from scipy.stats import unitary_group

from scipy.sparse import dia_matrix

from scipy.sparse import hstack,vstack
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
num_of_css=5

is_pca=0 #0 for SVD, 1 for PCA 
is_sparse=1
Data=np.random.randn(100,11)
if is_sparse==0:

    mean_data=np.mean(Data,0)
    mean_data=np.reshape(mean_data,(1,len(mean_data)))

else:    
    Data1=SM(Data)
    mean_data=Data1.mean(0)
Datam=Data-np.dot(np.ones((Data.shape[0],1)),mean_data)
Data1=SM(Data)

n=Data.shape[0]
d=Data.shape[1]
opt_error=1
r=1
is_to_save=0
is_save=1

is_big_data=1
alg_list=[0,3,5]
t0=1
if datum==2:
    coreset_space=5
else:
    coreset_space=1
coreset_space=10
jd=[5,10]
if opt_error==1:
    U,D,Vt=ssp.linalg.svds(Datam,max(jd))
    vlen=Vt.shape[0]
    Vt=Vt[vlen-1-np.arange(vlen),:] #flip V rows
    V=np.transpose(Vt)
    coreset_space=max(jd)+1
coreset_space=100
num_of_j=len(jd)
num_of_algs=len(alg_list)


weak_error=np.zeros((num_of_j*num_of_algs,num_of_css+1))
weak_error1=np.zeros((num_of_j*num_of_algs,num_of_css+1))

times=np.zeros((num_of_j*num_of_algs,num_of_css+1))
real_coreset_sizes=np.zeros((num_of_j*num_of_algs,num_of_css+1))
siz_con=10
coreset_sizes=np.zeros((num_of_css+1))
is_part=0
j_step=1
is_single=0
is_weak=1
is_kline=1
w=np.ones((Data.shape[0],1))
labs=['Uniform','Iterative Sens','CNW','Projetion Sens','Statman']
num_of_coresets=5
num_of_exp=1
error0=np.zeros((len(jd)*num_of_exp,len(alg_list),num_of_coresets))
time0=np.zeros((len(jd)*num_of_exp,len(alg_list),num_of_coresets))
coreset_sizes=np.ones(num_of_coresets)
is_time_mes=1
is_exp=1
gamma1=0.01
spar=0

mes_prob_beg=time.time()
Data=SM(Data)

if is_exp==1:
    for j0 in range (0,num_of_j):     #different number of dimentions   
    
        j=jd[j0]
                
        V=np.transpose(Vt[:jd[j0],:])
        if is_pca==1:
            DataV=SM.dot(Datam,V)
        else:
            DataV=SM.dot(Data,V)


        DataV_norm=np.linalg.norm(DataV,ord='fro')**2

        print('j=',j)    
        for o in range(num_of_exp):
            print('trial=',o)
            

            for alg1 in range(0,num_of_algs):
                alg=alg_list[alg1]
                print('alg=',alg)
                for g in range (num_of_coresets):#calculates coreset    
                    coreset_sizes[g]=int(((g+1)/(num_of_coresets)*0.8*Data.shape[0]))
                    #coreset_sizes[g]=(g+1)*200
                    eps=j/(coreset_sizes[g]-j+1)
                    if alg==0:
                        mes0=time.time()
    
                        prob=np.ones(Data.shape[0])/Data.shape[0]
                        DMM_ind=np.random.choice(Data.shape[0],int(coreset_sizes[g]), replace=False, p=prob)
                        time0[j0+o,alg1,g]=time.time()-mes0

                    if alg==2:
                       mes2=time.time()
                       if is_pca==1:
                           prob=prob2+81*eps
                       else:
                           prob=prob2                           
                       if g==0:
                           DMM_ind=np.random.choice(Data.shape[0],int(coreset_sizes[0]), replace=False, p=prob)
                       else:
                           print('HHHHHHHHHHHHHHHHHHHHHHHHH')
                           DMM_ind=np.concatenate((DMM_ind,np.random.choice(Data.shape[0],int(coreset_sizes[0]), replace=True, p=prob)))
                       time0[j0+o,alg1,g]=time.time()-mes2+mes_prob2

                    if alg==4:

                       mes4=time.time()
                       if is_pca==1:
                           prob=prob4+81*eps
                       else:
                           prob=prob4
                       if g==0:
                           DMM_ind=np.random.choice(Data.shape[0],int(coreset_sizes[0]), replace=False, p=prob)
                       else:
                           print('HHHHHHHHHHHHHHHHHHHHHHHHH')
                           DMM_ind=np.concatenate((DMM_ind,np.random.choice(Data.shape[0],int(coreset_sizes[0]), replace=True, p=prob)))

                       time0[j0+o,alg1,g]=time.time()-mes4+mes_prob4
                    print('DMM ind',DMM_ind.shape[0])
                    u=np.divide(np.ones(Data.shape[0]),prob[0:Data.shape[0]])/coreset_sizes[g]
                    u1=np.reshape(u[DMM_ind],(len(DMM_ind),1))
                    
                    X=Data[DMM_ind,:].multiply(np.sqrt(u1))
                    X=X.toarray()
                    mean_X=np.mean(X,0)
                    mean_X=np.reshape(mean_X,(1,len(mean_X)))


                    if alg==3:
                        mes3=time.time()
                        print('Data shape',Data.shape)
                        X,_=gsc.SCNW_classic(Data.toarray(),j,int(coreset_sizes[g]),0)                                                
                        time0[j0+o,alg1,g]=time.time()-mes3
    
                    if alg==5:
                        mes5=time.time()
                        if is_sparse==1:
                            Data1=SM(Data)
                        else:
                            Data1=np.copy(Data)

                        X=gsc.my_coreset(Data1,j,0,1,int(coreset_sizes[g]),3,is_sparse,is_kline=1)
                        time0[j0+o,alg1,g]=time.time()-mes5

                    if is_pca==1:                    
                        Xm=X-np.dot(np.ones((X.shape[0],1)),mean_X)
                        U_c,D_c,V_ct=ssp.linalg.svds(Xm,jd[j0])
                    else:
                        print('X.shape',X.shape)
                        U_c,D_c,V_ct=ssp.linalg.svds(X,jd[j0])
    
                    vclen=V_ct.shape[0]
                    V_ct=V_ct[vclen-1-np.arange(vclen),:]
                    V_c=np.transpose(V_ct)
                    #DataV_c=np.dot(Data-np.dot(np.ones((Data.shape[0],1)),mean_X),V_c)
                    if is_pca==1:
                            DataV_c=np.dot(Data-np.dot(np.ones((Data.shape[0],1)),mean_X),V_c)
                      
                    else:
                            print(Data.shape)
                            print(V_c.shape)

                            DataV_c=SM.dot(Data,V_c)


    
                    error0[num_of_exp*j0+o,alg1,g]=np.abs(DataV_norm-np.linalg.norm(DataV_c,ord='fro')**2)/DataV_norm

         

mean_error0=np.zeros((len(jd),len(alg_list),error0.shape[2]))
std_error0=np.zeros((len(jd),len(alg_list),error0.shape[2]))
mean_time0=np.zeros((len(jd),len(alg_list),time0.shape[2]))
std_time0=np.zeros((len(jd),len(alg_list),time0.shape[2]))
#import random
for j0 in range (len(jd)):
    for alg1 in range (len(alg_list)):
        mean_error0[j0,alg1,:] = np.mean(error0[num_of_exp*j0:num_of_exp*(j0+1),alg1,:],0)
        std_error0[j0,alg1,:]= np.std(error0[num_of_exp*j0:num_of_exp*(j0+1),alg1,:],0)
        mean_time0[j0,alg1,:] = np.mean(time0[num_of_exp*j0:num_of_exp*(j0+1),alg1,:],0)
        std_time0[j0,alg1,:]= np.std(time0[num_of_exp*j0:num_of_exp*(j0+1),alg1,:],0)
    
labs=['Uniform sampling','sohler','Tight sensitivity bound','CNW','Existing current bound','k-line+CNW']
fig_num=5  
colors='brg'
for j0 in range(num_of_j):            
        plt.figure(fig_num*j0-3)

        for alg1 in range(len(alg_list)):
            plt.title("k="+str(jd[j0])+", Dataset "+name)
            plt.plot(coreset_sizes[:],mean_error0[0,alg1,:],marker='^',label=labs[alg_list[alg1]])   
            plt.legend(loc=1)
            plt.xlabel("# Sampled points")
            plt.ylabel("Approximation error")
            #plt.title('dim='+str(j))
            plt.xlim((np.min(coreset_sizes[:]),np.max(coreset_sizes[:])))
            
for j0 in range(num_of_j):            
        plt.figure(fig_num*j0)

        for alg1 in range(len(alg_list)):
            plt.title("k="+str(jd[j0])+", Dataset "+name)       
            plt.plot(coreset_sizes[:],mean_time0[j0,alg1,:],marker='^',label=labs[alg_list[alg1]])   
            plt.legend(loc=0)
            plt.xlabel("# Sampled points")
            plt.ylabel("time")
            #plt.title('dim='+str(j))
            plt.xlim((np.min(coreset_sizes[:]),np.max(coreset_sizes[:])))

