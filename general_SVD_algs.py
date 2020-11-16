# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 11:53:09 2019

@author: Adiel
"""
import numpy as np
import scipy.sparse as ssp
from scipy.sparse import linalg
from scipy.sparse import csr_matrix as SM
from scipy.sparse import coo_matrix as CM
#from scipy.stats import unitary_group
import scipy 

from scipy.sparse import dia_matrix

from scipy.sparse import hstack,vstack
import matplotlib.pyplot as plt
import pandas as pd
import time
gamma1=0.00000001
is_jl=1
def calc_error(A,SA,k):  
        AtA=np.dot(np.transpose(A),A)
        SAtSA=np.dot(np.transpose(SA),SA)    
        ro=100
        d=AtA.shape[0]
        X=np.random.rand(d,d-k)            
        V,R=np.linalg.qr(X)
        Vt=np.transpose(V)
        VtAtAV=np.dot(Vt,np.dot(AtA,V))
        VtSAtSAV=np.dot(Vt,np.dot(SAtSA,V))
        newro=np.abs(np.trace(VtAtAV)/np.trace(VtSAtSAV))
        j=0
        while np.logical_and(max(ro/newro,newro/ro)>1.01,j<10):
            j=j+1
            ro=newro
            G=AtA-ro*SAtSA
            w,v=np.linalg.eig(G)
            V=v[:,0:d-k]
            Vt=np.transpose(V)
            VtAtAV=np.dot(Vt,np.dot(AtA,V))
            VtSAtSAV=np.dot(Vt,np.dot(SAtSA,V))
            newro=np.abs(np.trace(VtAtAV)/np.trace(VtSAtSAV))
        roS=100
        d=AtA.shape[0]
        X=np.random.rand(d,d-k)            
        V,R=np.linalg.qr(X)
        Vt=np.transpose(V)
        VtAtAV=np.dot(Vt,np.dot(AtA,V))
        VtSAtSAV=np.dot(Vt,np.dot(SAtSA,V))
        newroS=np.abs(np.trace(VtSAtSAV)/np.trace(VtAtAV))
        j=0
        while np.logical_and(max(roS/newroS,newroS/roS)>1.01,j<10):
            j=j+1
            roS=newroS
            G=SAtSA-roS*AtA
            w,v=np.linalg.eig(G)
            V=v[:,0:d-k]
            Vt=np.transpose(V)
            VtAtAV=np.dot(Vt,np.dot(AtA,V))
            VtSAtSAV=np.dot(Vt,np.dot(SAtSA,V))
            newroS=np.abs(np.trace(VtSAtSAV)/np.trace(VtAtAV))
        return max(np.abs(newroS-1),np.abs(newro-1),np.abs(1-newro),np.abs(1-newroS))

def Nonuniform_Alaa(AA0,k,is_pca,eps,spar): 
        d=AA0.shape[1]
        if is_pca==1:
                k=k+1
                AA0=PCA_to_SVD(AA0,eps,spar)
        if is_jl==1:
            dex=int(k*np.log(AA0.shape[0]))
            ran=np.random.randn(AA0.shape[1],dex)
            if spar==1:
                AA=SM.dot(AA0,ran)
            else:
                AA=np.dot(AA0,ran)
        else:
            AA=AA0

        size_of_coreset=int(k+k/eps-1) 
        U,D,VT=ssp.linalg.svds(AA,k)       
        V = np.transpose(VT)
        AAV = np.dot(AA, V)
        del V
        del VT    
        x = np.sum(np.power(AA, 2), 1)
        y = np.sum(np.power(AAV, 2), 1)
        P = np.abs(x - y)
        AAV=np.concatenate((AAV,np.zeros((AAV.shape[0],1))),1)
        Ua, _, _ = ssp.linalg.svds(AAV,k)
        U = np.sum(np.power(Ua, 2), 1)
        pro = 2 * P / np.sum(P) + 8 * U
        if is_pca==1:
            pro=pro+81*eps
        pro0 = pro / sum(pro)
        w=np.ones(AA.shape[0])
        u=np.divide(w,pro0)/size_of_coreset
        DMM_ind=np.random.choice(AA.shape[0],size_of_coreset, p=pro0)
        u1=np.reshape(u[DMM_ind],(len(DMM_ind),1))
        if spar==1:
            SA0=SM(AA0)[DMM_ind,:d].multiply(np.sqrt(u1))

        else:
            SA0=np.multiply(np.sqrt(u1),AA0[DMM_ind,:d])
        return pro0,SA0,0,size_of_coreset#,#AA.shape[0]*SA0/size_of_coreset   
def sorted_eig(A):
	eig_vals, eig_vecs =scipy.linalg.eigh(A)  # np.linalg.eig(A)	
	eig_vals_sorted = np.sort(eig_vals)[::-1]
	eig_vecs = eig_vecs.T
	eig_vecs_sorted = eig_vecs[eig_vals.argsort()][::-1]
	return eig_vals_sorted,eig_vecs_sorted
def get_unitary_matrix(n, m):
	a = np.random.random(size=(n, m))
	q, _ = np.linalg.qr(a)
	return q
	
def get_gamma(A_tag,l,d):
	vals , _ = sorted_eig(A_tag)
	sum_up = 0;sum_down = 0
	for i in range (l) : 
		sum_up += vals[d-i -1]
		sum_down += vals[i]
	return (sum_up/sum_down)
def calc_sens1(A,p,j,eps):
	
	d=A.shape[1]; l = d-j;
	A_tag = np.dot(A.T , A) ; 
	p = np.reshape(p, (p.shape[0], 1)).T ; 
	p_tag = np.dot(p.T,p) ;
	s_old = -float("inf")
	x = get_unitary_matrix(d, l)
	step = 0  ; stop = False
	gama = get_gamma(A_tag,l,d);
	stop_rule = (gama*eps)/(1-gama)
	
	s_l = []
	s_old = 0 
	s_new = np.inf
	while  step <20000:	
		s_new =  np.trace( np.dot (np.dot(x.T,p_tag) ,x))  / np.trace( np.dot(np.dot(x.T,A_tag) , x  ))
	
		s_l.append(s_new)
		G = p_tag - s_new*A_tag
		_ , ev = sorted_eig(G)
		x = ev[:l].T
		if s_new - stop_rule < s_old :
			return max(s_l)
		s_old = s_new 
		step+=1
	return max(s_l)
def calc_sens(A,p,j,eps):
	
    d=A.shape[1]; l = d-j;
    A_tag = np.dot(A.T , A) ; 
    p = np.reshape(p, (p.shape[0], 1)).T ; 
    p_tag = np.dot(p.T,p) ;
    s_old = -float("inf")
    x = get_unitary_matrix(d, l)
    step = 0  ; stop = False
    gama = get_gamma(A_tag,l,d);
    stop_rule = (gama*eps)/(1-gama)

    	
    s_l = []
    s_old = 0 
    while  step <20000:	
        s_new =  np.trace( np.dot (np.dot(x.T,p_tag) ,x))  / np.trace( np.dot(np.dot(x.T,A_tag) , x  ))
        	
        s_l.append(s_new)
        G = p_tag - s_new*A_tag
        _ , ev = sorted_eig(G)
        x = ev[:l].T
        if s_new - stop_rule < s_old :                
            return max(s_l)
        s_old = s_new 
        step+=1
    #print('step',step)
    return max(s_l)	
def PCA_to_SVD(P,epsi,is_spar):
    if is_spar==0:
        r=1+2*np.max(np.sum(np.power(P,2),1))/epsi**4
        P=np.concatenate((P,r*np.ones((P.shape[0],1))),1)
    else:
        P1=SM.copy(P)
        P1.data=P1.data**2
        r=1+2*np.max(np.sum(P1,1))/epsi**4
        P=hstack((P,r*np.ones((P.shape[0],1))))
    return P
def alaa_coreset(wiki0,j,eps,coreset_size,w,is_pca,spar):    
    dex=int(j*np.log(wiki0.shape[0]))
    d=wiki0.shape[1]
    if is_pca==1:
        j=j+1
        wiki0=PCA_to_SVD(wiki0,eps,spar)
    if is_jl==1:
        ran=np.random.randn(wiki0.shape[1],dex)
        if spar==1:
            wiki=SM.dot(wiki0,ran)	
        else:
            wiki=np.dot(wiki0,ran)	
    else:
        wiki=wiki0
    w=w/wiki.shape[0]
    print('w sum',np.sum(w))

    sensetivities=[]
    jd=j
    w1=np.reshape(w,(len(w),1))
    wiki1=np.multiply(np.sqrt(w1),wiki)
    k=0
    for i,p in enumerate(wiki1) :
        k=k+1
        sensetivities.append(calc_sens(wiki1,p,jd,eps))
    
    p0=np.asarray(sensetivities)
    if is_pca==1:
        p0=p0+81*eps
    indec=np.random.choice(np.arange(wiki.shape[0]),int(coreset_size),p=p0/np.sum(p0)) #sampling according to the sensitivity
    p=p0/np.sum(p0) #normalizing sensitivies
    w=np.ones(wiki.shape[0])

    u=np.divide(np.sqrt(w),p)/coreset_size #caculating new weights
    u1=u[indec]
    #u1=u1/np.mean(u1)
    u1=np.reshape(u1,(len(u1),1))
    squ=np.sqrt(u1)   
    if spar==1:        
        C=SM(wiki0)[indec,:d].multiply(squ)
    else:

        C=np.multiply(squ,wiki0[indec,:d])
    return p,C,0,u[indec],coreset_size#,wiki.shape[0]*wiki[indec,:]/coreset_size#
	
def alaa_coreset1(wiki,j,eps,w):	
    coreset_size=j+int(j/eps)-1
    print('w sum',np.sum(w))

    sensetivities=[]
    jd=j
    #jd=j

    #coreset_size=j+int(j/eps)-1
    w1=np.reshape(w,(len(w),1))
    wiki1=np.multiply(np.sqrt(w1),wiki)
    for i,p in enumerate(wiki1) :
        sensetivities.append(calc_sens(wiki1,p,jd,eps))
    p0=np.asarray(sensetivities)
    print('sens sum',np.sum(p0))
    indec=np.random.choice(np.arange(wiki.shape[0]),coreset_size,p=p0/np.sum(p0)) #sampling according to the sensitivity
    p=p0/np.sum(p0) #normalizing sensitivies
    w=np.ravel(w)

    u=np.divide(np.sqrt(w),p)/coreset_size #caculating new weights
    u=np.ravel(u)
    print('p sum',p.shape)

    print('u sum',u.shape)

    u1=np.power(u[indec],2)
    print('w sum',w.shape)

    #u=1.1*u/np.sum(u)
    print('u sum',u.shape)
    u1=np.reshape(u1,(len(u1),1))
    squ=np.sqrt(u1)
    return C,0,u[indec],coreset_size#,wiki.shape[0]*wiki[indec,:]/coreset_size#
def sohler1(A,j,eps,is_sparse=0):
    m=np.min((np.min(A.shape)-1,j+int(j/eps)-1)) #the coreset size, and also the SVD degree
    print('coreset size',m)
    U,D,Vt=ssp.linalg.svds(A,m) #SVD 
    S=np.multiply(np.reshape(D,(len(D),1)),Vt)#/np.power(np.linalg.norm(A,ord='fro'),2)
    Am=np.dot(U,S)
    Delta=np.power(np.linalg.norm(A-Am,ord='fro'),2)#/np.power(np.linalg.norm(A,ord='fro'),2)
    return SM(S),Delta,m

def unif_sam(A,j,eps,is_sparse=0):
    m=j+int(j/eps)-1
    print('coreset size',A.shape)
    print('coreset size',type(A))

    S=A[np.random.choice(A.shape[0],size=m,replace='false'),:]
    Delta=0
    return S,Delta,m
def initializing_data(Q,k):
    print('Q',Q.shape)
    #U, D, VT = ssp.linalg.svds(np.transpose(Q),2*k)
    U, D, VT = np.linalg.svd(np.transpose(Q))

    V=VT.T
    D=np.diag(D)
    Z=V[:,0:2*k]
    V_k=V[:,0:k] 
    print(D.shape)
    DVk=np.dot(D[0:k,0:k],np.transpose(V_k))
    Q_k=np.dot(U[:,0:k],DVk)
    ZZt=np.dot(Z,np.transpose(Z))
    Qt=np.transpose(Q)
    A_2t=(np.sqrt(k)/np.linalg.norm(Q-np.transpose(Q_k),'fro'))*(Qt-np.dot(Qt,ZZt))
    A_2=np.transpose(A_2t)
    An=np.concatenate((Z,A_2),1)

    return An,A_2

    
def single_CNW_iteration_classic( A,At,delta_u,delta_l,X_u,X_l,Z):
     #AtA=np.dot(At,A)
     M_u = np.linalg.inv(X_u-Z)
     M_l = np.linalg.inv(Z-X_l)
     betha_diff=np.zeros((A.shape[0],2))
     L=np.dot(A,np.dot(M_l,At))
     L2=np.dot(L,L)
     U=np.dot(A,np.dot(M_u,At))
     U2=np.dot(U,U)
     betha_l0=L2/(delta_l*np.trace(L2))-L
     betha_u0=U2/(delta_u*np.trace(U2))+U
     betha_l=np.diag(betha_l0)
     betha_u=np.diag(betha_u0)  

     betha_diff=betha_l-betha_u
     #print(np.max(betha_diff))
     betha_diff2=np.argmax(betha_diff)         
     jj=betha_diff2
     t=(1/betha_l[jj])/2+(1/betha_u[jj])/2 #should be between (1/betha_l[jj]) to (1/betha_u[jj])
     aj=np.zeros((1,A.shape[1]))
     aj[0,:]=A[jj,:]      
     ajtaj=np.dot(np.transpose(aj),aj)
     Z=Z+np.power(t,1)*ajtaj


     return Z,jj,t
 
def SCNW_classic(A2,k,coreset_size,is_jl):
    coreset_size=int(coreset_size)

    """
    This function operates the CNW algorithm, exactly as elaborated in Feldman & Ras

    inputs:
    A: data matrix, n points, each of dimension d.
    k: an algorithm parameter which determines the normalization neededand the error given the coreset size.
    coreset_size: the maximal coreset size (number of lines inequal to zero) demanded for input.
    output:
    error: The error between the original data to the CNW coreset.        
    duration: the duration this CNW operation lasted
    """
    if is_jl==1:
        dex=int(k*np.log(A2.shape[0]))
    
        ran=np.random.randn(A2.shape[1],dex)
        A1=SM.dot(A2,ran)	
    else:
        A1=np.copy(A2)
    print('A1.shape',A1.shape)
    epsi=np.sqrt(k/coreset_size)    #
    A,A3=initializing_data(A1,k)
    print('A.shape',A.shape)
    At=np.transpose(A)
    AtA=np.dot(At,A)
    num_of_channels = A.shape[1]
    ww = np.zeros((int(coreset_size)))
    Z = np.zeros((num_of_channels,num_of_channels))
    X_u = k*np.diag(np.ones(num_of_channels))
    X_l =-k*np.diag(np.ones(num_of_channels))
    delta_u = epsi+2*np.power(epsi, 2)
    delta_l = epsi-2*np.power(epsi, 2)
    ind=np.zeros(int(coreset_size), dtype=np.int)             

    for j in range(coreset_size):
         if j%50==1:
             print('j=',j)
         X_u=X_u+delta_u*AtA
         X_l=X_l+delta_l*AtA                  
         Z,jj,t=single_CNW_iteration_classic(A,At,delta_u,delta_l,X_u,X_l,Z)
         ww[j]=t
         ind[j]=jj             
    sqrt_ww=np.sqrt(epsi*ww/k)
    sqrt_ww=np.reshape(sqrt_ww,(len(sqrt_ww),1))
    if is_jl==1:
        SA0=SM(A2)[ind,:].multiply(sqrt_ww)
    else:
        SA0=np.multiply(A2[ind,:],sqrt_ww)
    return SA0,ind


    
def nor_data(A):
  
        A1=np.power(A,2)
        Anormssq=np.sqrt(np.sum(A1,1))
        Anormssq[Anormssq==0]=1
        we=1/Anormssq
        A=np.multiply(np.reshape(we,(len(we),1)),A) 
        return A,Anormssq
def nor_data1(A):
        A1=SM.copy(A)
        A1.data=np.power(A.data,2)
        Anormssq=np.sqrt(np.sum(A1,1))
        Anormssq[Anormssq==0]=1
        we=1/Anormssq
        A=A.multiply(np.reshape(we,(len(we),1))) 
        return SM(A),Anormssq
def make_P_dense(M):
    d=M.shape[1]
    P=np.zeros((M.shape[0],d+2))
    p=np.sum(np.power(M, 2), 1)
   # print('p.shape',p.shape)
   # print('p.shape',P[:,1:2].shape)

    P[:,1:2]=np.reshape(p,(len(p),1)) #P defined just as in the algorithm you sent me
    P[:,0]=1                        
    P[:,2:d+2]=-2*M
    return P    
def squaredis_dense(P,Cent,to_pert=0):
    #print(Cent.shape)
    if len(Cent.shape)==3:
        Cent=np.reshape(Cent,(Cent.shape[0],Cent.shape[2]))    
    d=Cent.shape[1]
    C=np.zeros((Cent.shape[0],d+2))    
    C[:,1]=1      #C is defined just as in the algorithm you sent me.
 #   if C.shape[0]==1:
  #      C[0:1,0:1] =np.sum(np.power(Cent, 2))
   # else:
    cent1=np.copy(Cent)
    print('Cent',type(Cent))
    cent1=np.power(Cent,2)
    c=np.sum(cent1, 1)
    C[:,0:1] =np.reshape(c,(len(c),1))
    C[:,2:d+2]=Cent
    D=np.dot(P,np.transpose(C))
    D[D<0]=0
    if to_pert>0:
        D=D+to_pert*np.random.rand(D.shape[0],D.shape[1])

    Tags=D.argmin(1)  #finding the most close centroid for each point 
    dists=D.min(1)
    y=D.argmin(0)
    y=np.reshape(y,(len(y),1))
    return dists,Tags,y

def make_P(M):
    n=M.shape[0]
    M1=SM.copy(M)
    M1.data=M.data**2
    M_norms=M1.sum(1)
    M=hstack((np.ones((n,1)),M_norms,-2*M))
    return M

def squaredis(P,Cent):    
    d=Cent.shape[1]
    C=SM((Cent.shape[0],d+2))    
    C[:,1]=1      #C is defined just as in the algorithm you sent me.
    C[:,0] =SM.sum(SM.power(Cent, 2), 1)
    C[:,2:d+2]=Cent
    D=SM.dot(P,C.T)
    D=D.toarray()
    Tags=D.argmin(1)#finding the most close centroid for each point 
    #print('Tags sha',Tags.shape)
    if min(D.shape)>1:

        dists=D.min(1)
    else:
        dists=np.ravel(D)
    y=D.argmin(0)
    #y=np.reshape(y,(len(y),1))

    return dists,Tags,y 

def kmeans_plspls1(A,w,eps,V,clus_num,we,alfa_app,is_sparse,is_jl):
        """
        This funtion operates the kmeans++ initialization algorithm. each point chosed under the Sinus probability.
        Input:
            A: data matrix, n points, each on a sphere of dimension d.
            k: number of required points to find.
        Output:
            Cents: K initial centroids, each of a dimension d.
        """
        if is_sparse==1:
            A=SM(A)
        if is_jl==1:
            dex=int(clus_num*np.log(A.shape[0]))
    
            ran=np.random.randn(A.shape[1],dex)
            A=SM.dot(A,ran)
            is_sparse=0      #A=np.multiply(w1,A)
        num_of_samples = A.shape[0]
        if any(np.isnan(np.ravel(w)))+any(np.isinf(np.ravel(w))):
            Cents= A[np.random.choice(num_of_samples,size=1),:]   #choosing arbitrary point as the first               
        else:  
            print('wsh',w.shape)              
            Cents= A[np.random.choice(num_of_samples,size=1,p=np.ravel(w)/np.sum(np.ravel(w))),:] #choosing arbitrary point as the first               
        if is_sparse==1:
            PA=make_P(A)
        else:
            PA=make_P_dense(A)
        fcost=alfa_app*1.1
        h1=1
        inds=[]
        print('cond 1',Cents.shape[0]<clus_num)
        print('cond 2',alfa_app<fcost)
#        print('Vs',np.dot(V.T,V).shape)

        while (Cents.shape[0]<clus_num+1):
        #for h1 in range(1, int(clus_num)): #h1 points of k have been chosed by far
            Cents2=Cents[h1-1:h1,:] 
            if is_sparse==1:
                Pmina,tags,_=squaredis(PA,Cents2)  
            else:
                Pmina,tags,_=squaredis_dense(PA,Cents2)  
            if h1==1:
                Pmin=Pmina
            else:
                Pmin=np.minimum(Pmin,Pmina)
                Pmin[np.asarray(inds)]=0
            #Pmin11=np.reshape(Pmin,(len(Pmin),1))
            #fcost=np.sqrt(np.sum(np.multiply(we,np.power(Pmin11,1)))) 
            Pmin[Pmin<0]=0
            Pmin00=np.multiply(w,Pmin)
            Pmin0=Pmin00/np.sum(Pmin00)
            if any(np.isnan(np.ravel(Pmin0)))+any(np.isinf(np.ravel(Pmin0))):
                ind=np.random.choice(Pmin.shape[0],1)
            else:
                Pmin0[Pmin0<0]=0
                ind=np.random.choice(Pmin.shape[0],1, p=Pmin0)
            if is_sparse==1:
                Cents=vstack((Cents,A[ind,:]))
            else:
                Cents=np.concatenate((Cents,A[ind,:]),0)

            inds.append(ind)
            h1=h1+1
        if eps>0:
            V_cnorm=np.linalg.norm(Cents-SM.dot(Cents,np.dot(V.T,V)),'fro')
            V_norm=np.linalg.norm(A-SM.dot(A,np.dot(V.T,V)),'fro')
        
            while (1+alfa_app*eps)*V_norm<V_cnorm:
            #for h1 in range(1, int(clus_num)): #h1 points of k have been chosed by far
                Cents2=Cents[h1-1:h1,:] 
                if is_sparse==1:
                    Pmina,tags,_=squaredis(PA,Cents2)  
                else:
                    Pmina,tags,_=squaredis_dense(PA,Cents2)  
                if h1==1:
                    Pmin=Pmina
                else:
                    Pmin=np.minimum(Pmin,Pmina)
                    Pmin[np.asarray(inds)]=0
                #Pmin11=np.reshape(Pmin,(len(Pmin),1))
                #fcost=np.sqrt(np.sum(np.multiply(we,np.power(Pmin11,1)))) 
                Pmin[Pmin<0]=0
                Pmin00=np.multiply(w,Pmin)
                Pmin0=Pmin00/np.sum(Pmin00)
                if any(np.isnan(np.ravel(Pmin0)))+any(np.isinf(np.ravel(Pmin0))):
                    ind=np.random.choice(Pmin.shape[0],1)
                else:
                    Pmin0[Pmin0<0]=0
                    ind=np.random.choice(Pmin.shape[0],1, p=Pmin0)
                if is_sparse==1:
                    Cents=vstack((Cents,A[ind,:]))
                else:
                    Cents=np.concatenate((Cents,A[ind,:]),0)
                V_cnorm=np.linalg.norm(Cents-SM.dot(Cents,np.dot(V.T,V)),'fro')
    
                inds.append(ind)
                h1=h1+1
        return Cents,inds


def Lloyd_iteration_dense2( A, P, w ,Q):
    dists,Tags,ta=squaredis_dense(P,Q)
    Qjl=np.zeros((Q.shape[0],A.shape[1]))
    wq=np.zeros((Q.shape[0],1))
    #y=0
    i=0
    w=np.reshape(w,(len(w),1))
    for q,wwq in zip(Qjl,wq):
        print(i)
        inds=np.where(Tags==i)[0]
        q=np.sum(np.multiply(A[inds,:],w[inds,:]),0)
        wwq=np.sum(w[inds,:],0)
        i=i+1
        print(i)
    wq[wq==0]=1
    wqi=1/wq
    Qjl=np.multiply(wqi,Qjl)
    return Qjl  
# 
def k_means_clustering_dense( A,  w ,K, iter_num,ind=[],is_sparse=0,is_kline=0): 
    if is_kline==1:
        A,Anormssq=nor_data(A)
        w=np.multiply(w,np.ravel(Anormssq))
    if is_sparse==1:
        ran=np.random.randn(A.shape[1],8*int(np.log(A.shape[0])))   
        Ajl=SM.dot(A,ran)
    else:
        Ajl=A
    if ind==[]:    
        ind=np.random.permutation(len(w))[0:K]
    Qnew=Ajl[ind,:]
    P=make_P_dense(Ajl)
    dists1=0
    if (iter_num>=1)+(iter_num==0):
        for i in range(0,iter_num):
            Qnew=Lloyd_iteration_dense2(Ajl,P,w,Qnew) 
            dists0=dists1
            dists1,Tags1,tagss=squaredis_dense(P,Qnew) 
            conv=np.abs(np.sum(np.multiply(w,dists0))-np.sum(np.multiply(w,dists1)))/np.sum(np.multiply(w,dists1))
            print('conv',conv)

    else:     
        Qjl=np.zeros(Qnew.shape)   
        dists0=0
        dists1,Tags1,tagss=squaredis_dense(P,Qnew)    
        i=0        
        conv=np.abs(np.sum(np.multiply(w,dists0))-np.sum(np.multiply(w,dists1)))/np.sum(np.multiply(w,dists1))
        while conv>iter_num:
            Qjl=Qnew
            Qnew=Lloyd_iteration_dense2(Ajl,P,w,Qjl)    
            i=i+1      
            dists0=dists1
            dists1,Tags1,tagss=squaredis_dense(P,Qnew)
            print(np.sum(np.multiply(w,dists1))/500)
            conv=np.abs(np.sum(np.multiply(w,dists0))-np.sum(np.multiply(w,dists1)))/np.sum(np.multiply(w,dists1))
            print('conv',conv)

    Q=A[tagss,:]
    return Q,w 

def old_clustering( A,w,alfa_app,eps,V, K,is_sparse,is_plspls=0,is_klinemeans=0):
        
        """
     
        inputs:
            A: data matrix, n points, each of dimension d.
            K: number of centroids demanded for the Kmeans.
            is_sparse: the  output SA0 will be: '0' the accurate cantroids, '1' the points that are the most close to the centroids.
            is_plspls: '1' to initialize with the kmeans++ algorithm which bounds the error, '0' random initialization.
            is_klinemeans:  '1' calculates klinemeans, '0' calculates Lloyd's kmeans.
        
        output:
            SA0: "ready coreset": a matrix of size K*d: coreset points multiplies by weights.
            GW1: weights
            Tags1: Data indices of the points chosen to coreset.
    
    """ 
        #sensitivity=0.01
        num_of_samples = A.shape[0]
        
        if is_klinemeans==1:
            if is_sparse==0:
                A1,weights1=nor_data(A)
            else:
                A1,weights1=nor_data1(A)
            weights1=np.reshape(weights1,(len(weights1),1))
            weights=np.multiply(w,weights1)
        else:
            if is_sparse==0:
                A1=np.copy(A)
            else:
                A1=SM.copy(A)
            weights=w
        print('A1',type(A1))
        print('A1',type(A1.shape[0]))
        print('A1',type(A1.shape[1]))

        num_of_samples = A1.shape[0]
        num_of_channels = A1.shape[1]
        K=int(K)
        if is_sparse==0:
            P=make_P_dense(A1)       
            Cent=np.zeros((2*K,num_of_channels))
        else:
            P=make_P(A1)       
            Centt=SM((2*K,num_of_channels))
        if is_plspls==1:
            Centt,per=kmeans_plspls1(A1,np.ravel(np.power(weights,2)),eps,V,K,np.power(weights,2),alfa_app,is_sparse,is_jl=0)            
        else:
            per=np.random.permutation(num_of_samples)
            #Cent[0:K,:]=A1[per[0:K],:]
        if is_sparse==0:
            #Cent=A1[np.ravel(per[0:K]),:]
            print('****per****',len(np.unique(per)))
            Cent=np.concatenate((A1[np.ravel(per[0:K]),:],-A1[np.ravel(per[0:K]),:]),0)
        else:
            Cent=vstack((A1[np.ravel(per[0:K]),:],-A1[np.ravel(per[0:K]),:]))
            #Cent=A1[np.ravel(per[0:K]),:]
            print('****per****',len(np.unique(per)))
        K1=Cent.shape[0]
    
        
        iter=0
        Cost=50 #should be just !=0
        old_Cost=2*Cost
    
        Tags=np.zeros((num_of_samples,1)) # a vector stores the cluster of each point
        print('c0s',Cent.shape)

        #while np.logical_and(min(Cost/old_Cost,old_Cost/Cost)<sensitivity,Cost>0.000001): #the corrent cost indeed resuces relating the previous one, 
        for i in range(1):
                            #however the loop continues until the reduction is not significantly and their ratio is close to one, and exceeds the parameter "sensitivity"    
            group_weights=np.zeros((K1,1))
            iter=iter+1 #counting the iterations. only for control
            old_Cost=Cost #the last calculated Cost becomes the old_Cost, and a new Cost is going to be calculated.
            if is_sparse==0:            
                Cent1=np.copy(Cent)
                Dmin,Tags,Tags1=squaredis_dense(P,Cent1)
            else:
                Cent1=SM.copy(Cent)
                Dmin,Tags,Tags1=squaredis(P,Cent1)
            #print('Tags',Tags)
            Cost=np.sum(Dmin) #the cost is the summation of all of the minimal distances
            for kk in range (1,K1+1):
                wheres=np.where(Tags==kk-1)  #finding the indeces of cluster k
                #print('wheres',weights[wheres[0]])
                weights2=np.power(weights[wheres[0]],2)  #finding the weights of cluster k
                group_weights[kk-1,:]=np.sum(weights2)
              
                        
        GW1=np.power(group_weights,0.5)
        print('***GW1***',len(np.where(GW1>0)[0]))
        F=Cent
        if is_sparse==0:
            
            SA0=np.multiply(GW1,F) #We may weight each group with its overall weight in ordet to compare it to the original data.   
        else:
            SA0=F.multiply(GW1)
         
        return SA0,GW1,Tags1    

    
    
def old_clustering1( A,alfa_app,eps,V, K,is_sparse,is_plspls=0,is_klinemeans=0):
        
        """
     
        inputs:
            A: data matrix, n points, each of dimension d.
            K: number of centroids demanded for the Kmeans.
            is_sparse: the  output SA0 will be: '0' the accurate cantroids, '1' the points that are the most close to the centroids.
            is_plspls: '1' to initialize with the kmeans++ algorithm which bounds the error, '0' random initialization.
            is_klinemeans:  '1' calculates klinemeans, '0' calculates Lloyd's kmeans.
        
        output:
            SA0: "ready coreset": a matrix of size K*d: coreset points multiplies by weights.
            GW1: weights
            Tags1: Data indices of the points chosen to coreset.
    
    """ 
        #sensitivity=0.01
        num_of_samples = A.shape[0]
        weights = np.ones((num_of_samples, 1))
        if is_klinemeans==1:
            if is_sparse==0:
                A1,weights=nor_data(A)
            else:
                A1,weights=nor_data1(A)
            weights=np.reshape(weights,(len(weights),1))
        else:
            if is_sparse==0:
                A1=np.copy(A)
            else:
                A1=SM.copy(A)
        print('A1',type(A1))
        print('A1',type(A1.shape[0]))
        print('A1',type(A1.shape[1]))

        num_of_samples = A1.shape[0]
        num_of_channels = A1.shape[1]
        K=int(K)
        if is_sparse==0:
            P=make_P_dense(A1)       
            Cent=np.zeros((2*K,num_of_channels))
        else:
            P=make_P(A1)       
            Centt=SM((2*K,num_of_channels))
        if is_plspls==1:
            Centt,per=kmeans_plspls1(A1,np.ravel(np.power(weights,2)),eps,V,K,np.power(weights,2),alfa_app,is_sparse,is_jl=0)            
        else:
            per=np.random.permutation(num_of_samples)
            #Cent[0:K,:]=A1[per[0:K],:]
        if is_sparse==0:
            #Cent=A1[np.ravel(per[0:K]),:]
            print('****per****',len(np.unique(per)))
            Cent=np.concatenate((A1[np.ravel(per[0:K]),:],-A1[np.ravel(per[0:K]),:]),0)
        else:
            Cent=vstack((A1[np.ravel(per[0:K]),:],-A1[np.ravel(per[0:K]),:]))
            #Cent=A1[np.ravel(per[0:K]),:]
            print('****per****',len(np.unique(per)))
        K1=Cent.shape[0]
    
        
        iter=0
        Cost=50 #should be just !=0
        old_Cost=2*Cost
    
        Tags=np.zeros((num_of_samples,1)) # a vector stores the cluster of each point
        print('c0s',Cent.shape)

        #while np.logical_and(min(Cost/old_Cost,old_Cost/Cost)<sensitivity,Cost>0.000001): #the corrent cost indeed resuces relating the previous one, 
        for i in range(1):
                            #however the loop continues until the reduction is not significantly and their ratio is close to one, and exceeds the parameter "sensitivity"    
            group_weights=np.zeros((K1,1))
            iter=iter+1 #counting the iterations. only for control
            old_Cost=Cost #the last calculated Cost becomes the old_Cost, and a new Cost is going to be calculated.
            if is_sparse==0:            
                Cent1=np.copy(Cent)
                Dmin,Tags,Tags1=squaredis_dense(P,Cent1)
            else:
                Cent1=SM.copy(Cent)
                Dmin,Tags,Tags1=squaredis(P,Cent1)
            #print('Tags',Tags)
            Cost=np.sum(Dmin) #the cost is the summation of all of the minimal distances
            for kk in range (1,K1+1):
                wheres=np.where(Tags==kk-1)  #finding the indeces of cluster k
                #print('wheres',weights[wheres[0]])
                weights2=np.power(weights[wheres[0]],2)  #finding the weights of cluster k
                group_weights[kk-1,:]=np.sum(weights2)
              
                        
        GW1=np.power(group_weights,0.5)
        print('***GW1***',len(np.where(GW1>0)[0]))
        F=Cent
        if is_sparse==0:
            
            SA0=np.multiply(GW1,F) #We may weight each group with its overall weight in ordet to compare it to the original data.   
        else:
            SA0=F.multiply(GW1)
         
        return SA0,GW1,Tags1    

def my_coreset(Ad1,j,is_jl,alfa_app,coreset_size,alg,is_sparse,is_kline):
    """
    This funtion operates the kmeans++ initialization algorithm. 
    Input:
        A: data matrix, n points, each on a sphere of dimension d.
        w: n weights
        k: number of required points to find.
    Output:
        Cents: K initial centroids, each of a dimension d.
    Ad=Data1
    we=Anormssq
    alfa_app=np.linalg.norm(Data,ord='fro')
    """    
    eps=j/(coreset_size-j+1)
    n=Ad1.shape[0]
    w=np.ones((n,1))
    _,_,V=ssp.linalg.svds(Ad1,j)
    if is_jl==1:
        ran=np.random.randn(Ad1.shape[1],int(j*np.log(Ad1.shape[0])))
        if is_sparse==1:
            Ad2=SM.dot(Ad1,ran)
        else:
            Ad2=np.dot(Ad1,ran)
    else:
        if is_sparse==1:
            Ad2=SM.copy(Ad1)
        else:
            Ad2=np.copy(Ad1)




    
    #if is_jl==1:
    if is_sparse==1:
        Ad2=SM(Ad2)
    C,GW1,Tags1=old_clustering1(Ad2,alfa_app,eps,V,coreset_size,is_sparse,1,is_kline)
    if is_jl==1:
        if is_sparse==1:
                Ad1,weights=nor_data1(Ad1)
                C =SM(Ad1)[Tags1,:].multiply(GW1)
        else:
                C =np.multiply(GW1,Ad1[Tags1,:])

    #C =Ad[Tags1,:].multiply(GW1)

    #else:
     #   Cents0,GW1,Tags1=old_clustering(Ad2,w,we,alfa_app,coreset_size,is_sparse-is_jl,1,is_kline)
    #print('GW1',GW1)
    print('GW1',np.sum(GW1))
    
    print('Csha',C.shape[0])
    if C.shape[0]>2*coreset_size:
        C=C.toarray()+0.01*np.random.rand(C.shape[0],C.shape[1])
        C,ind=SCNW_classic(C,j,min(coreset_size,C.shape[0]),0)
        
        #C1=Ad1[ind,:].multiply(GW1[ind])
#        print('C',C)
#    if alg==4:
#        prob,C,delta_C,real_cs=Nonuniform_Alaa(Cents0,j,0,eps,0)
##    else:
#        print('yes, else')
#
#        C=SM((coreset_size,Cents1.shape[1]))            
#        C[0:Cents1.shape[0],:]=SM(Cents1)
    return C#,delta_C,real_cs



def cal_opssp_j(A,j):
    #print('A=',A)
    U,D,Vt=ssp.linalg.svds(A,j)          
    V=np.transpose(Vt)
    AVVt=np.dot(A,np.dot(V,Vt)) 
    return np.linalg.norm((A-AVVt),ord='fro')**2
    
def SVD_streaming(path,name,Data,j,is_jl,alg,h,spar,trial=None,datum=None,gamma1=0.000000001):
    """
    alg=0 unif sampling
    alg=1 Sohler
    alg=2 CNW
    alg=3 Alaa
    """
    alfa=0.0000000001
    alg_inmine=3
    real_cs1=0
    prob=0
    coreset_size=Data.shape[0]//(2**(h+1))
    gamma=j/(coreset_size-j+1)
    k=0
    T_h= [0] * (h+1)
    DeltaT_h= np.zeros(h+1)
    if alg==2: #where there are weights
        u_h=[0]* (h+1) 
    leaf_ind=np.zeros(h+1)
    last_time=time.time()
    for jj in range(np.power(2,h)): #over all of the leaves
        Q=Data[k:k+2*coreset_size,:]
    
    
        k=k+2*coreset_size
        if alg==0:
            T,DeltaT,real_cs1=unif_sam(Q,j,gamma) #making a coreset of the leaf
        if alg==1:
            T,DeltaT,real_cs1=sohler1(Q,j,gamma)
        if alg==2:
            prob,T,DeltaT,u,real_cs1=alaa_coreset(Q,j,gamma1,coreset_size,np.ones(Q.shape[0])/Q.shape[0],is_pca,spar)
        if alg==3:
            T,_=SCNW_classic(Q,j,coreset_size,is_jl)
            print('type T',type(T))
        if alg==4:
            prob,T,DeltaT,real_cs1=Nonuniform_Alaa(Q,j,is_pca,gamma,spar)    
        if alg==5:
            if spar==1:
                Q=SM(Q)
            T=my_coreset(Q,j,is_jl,alfa,coreset_size,alg_inmine,spar,1)
        print('leaf num',jj)
        print('j=',j)
        i=0
                        

        while (i<h)*(type(T_h[i])!=int): #every time the leaf has a neighbor leaf it should merged and reduced
           if spar==0:
               print('T',T)
               print('Th',T_h)

               totT=np.concatenate((T,np.asarray(T_h[i])),0)
           else:
               print(type(T))
               print(T.shape)

               print(type(T_h[i]))
               print(T_h[i].shape)

               totT=vstack((T,T_h[i]))

                          #T=T*len(totT)/2*real_cs1

           if alg==0:
                T,DeltaT,real_cs1=unif_sam(totT,j,gamma)
                prob=np.ones(T.shape[0])/T.shape[0]
           if alg==1:
                T,DeltaT,real_cs1=sohler1(totT,j,gamma)
           if alg==2:
               
                prob,T,DeltaT,u,real_cs1=alaa_coreset(totT,j,gamma1,coreset_size,np.ones(totT.shape[0])/totT.shape[0],is_pca,spar)       
           if alg==3:
                if spar==1:
                    T,_=SCNW_classic(vstack((SM(T),T_h[i])),j,coreset_size,is_jl)
                else:
                    T,_=SCNW_classic(np.concatenate((T,T_h[i]),0),j,coreset_size,is_jl)

           if alg==4:


               prob,T,DeltaT,real_cs1=Nonuniform_Alaa(totT,j,is_pca,gamma,spar)
               print(type(T))
           if alg==5:
                T=my_coreset(totT,j,is_jl,alfa,coreset_size,alg_inmine,spar,1)              

           print('i=',i)
           print('j=',j)
           if alg==2:
               u_h[i]=0
    
           DeltaT=0 #zeroing leaf which reduced
           T_h[i]=0
           leaf_ind[i]=leaf_ind[i]+1
           if DeltaT_h[i]==0:
               DeltaT_h[i]=time.time()-last_time

           i=i+1
           
        T_h[i]=T
        print('TTTTTTTTTTTTT',T.shape)
        if spar==0:            
                np.save(path+name+',j='+str(j)+',alg='+str(alg)+',floor='+str(i)+',leaf='+str(leaf_ind[i])+'1.npy',T)
        
        else:
                ssp.save_npz(path+'trial='+str(trial)+',j='+str(j)+',alg='+str(alg)+',floor='+str(i)+',leaf='+str(leaf_ind[i])+'1.npz',T)

        if alg==2:
            u_h[i]=u #keeping the leaf which left
        

        Q=[]
        #print(T_h)
        
    if type(T_h[h])==int: #should be remained only the upper one. if not:
        all_levels=[]
        for g in range (h+1): #collecting all leaves which remained on tree.
            if type(T_h[g])!=int:
                if all_levels==[]:
                   all_levels=np.asarray(T_h[g])
                else:
                    all_levels=np.concatenate((all_levels,np.asarray(T_h[g])),0)
        DeltaT_hs=sum(DeltaT_h[h]) #summing its delta
    else:
        all_levels=T_h[h] 
        DeltaT_hs=DeltaT_h[h]
    np.save('C:/Users/Adiel/Dropbox/All_experimental/Clus/name='+str(name)+',j='+str(j)+',alg='+str(alg)+',floors='+str(h)+'.npy',DeltaT_h)

    return prob,all_levels,DeltaT_hs,real_cs1
 
def Lloyd_iteration2( A,P, w ,Q):
    dists,Tags,_=squaredis(P,Q)
    print('finish squaredis')
    Qjl=SM((Q.shape[0],A.shape[1]))
    wq=np.zeros((Q.shape[0],1))
    w=np.reshape(w,(len(w),1))
    for i in range (Qjl.shape[0]):
            print(i)
            inds=np.where(Tags==i)[0]
            Qjl[i,:]=(SM(A)[inds,:].multiply(w[inds,:])).sum(0)
            wq[i,:]=np.sum(w[inds,:],0)
    wq[wq==0]=1
    wqi=1/wq
    Qjl=Qjl.multiply(wqi)
    return SM(Qjl)
def kmeans_plspls_FBL(Ad,w,clus_num):
    """
    This funtion operates the kmeans++ initialization algorithm. 
    Input:
        A: data matrix, n points, each on a sphere of dimension d.
        w: n weights
        k: number of required points to find.
    Output:
        Cents: K initial centroids, each of a dimension d.
    """
    print('Ad',Ad.shape)
    print('Ad',type(Ad))

    num_of_samples = Ad.shape[0] #=n
    print('num_of_samples')
    num_of_channels = Ad.shape[1] #=d
    Cents=SM((clus_num,num_of_channels))
    print('a shape',np.arange(num_of_samples).shape)
    print('p shape',np.ravel(w).shape)


    w[w<0]=0
    #print('w',w)
    ind0=np.random.choice(np.arange(num_of_samples),1,p=np.ravel(w)/np.sum(w))
    
    inds=ind0
   # print('ind0',ind0)
#    print(type(Ad[ind0[0]:(ind0[0]+1),:]))
 #   print(type(Cents[0:1,:]))
    Ad=SM(Ad)
    Cents[0:1,:] = Ad[ind0[0]:(ind0[0]+1),:]  #choosing arbitrary point as the first             
    P=make_P(Ad)        
    for h1 in range(1, int(clus_num)): #h1 points of k have been chosed by far
        Cents2=Cents[h1-1:h1,:]
        #print('Cents2',Cents2.shape)
        Pmina,tags,_=squaredis(P,Cents2) 
        
        Pmina[Pmina<0]=0        

        if h1==1:
            Pmin=Pmina
        else:
            Pmin=np.minimum(Pmin,Pmina)
        Pmin[np.asarray(inds)]=0
        Pmin1=np.multiply(w.T,Pmin)
        #print('Pmin',Pmin1)
        Pmin0=Pmin1.T/np.sum(Pmin1)
        #print(Pmin0.shape)
        ind=np.random.choice(np.arange(len(Pmin0)),1, p=np.ravel(Pmin0))
        Cents[h1,:]=Ad[ind,:] #adding the new point to 
        inds=np.append(inds,ind)
    return Cents,inds 
                        
def k_means_clustering( A,  w ,K, iter_num,ind=[],is_sparse=0,is_kline=0): 
    if is_kline==1:
        A,Anormssq=nor_data1(A)
        w=np.multiply(w,np.ravel(Anormssq))
    if ind==[]:    
        ind=np.random.permutation(len(w))[0:K]
    Qnew=A[ind,:]
    P=make_P(A)
    dists1=0
    if (iter_num>=1)+(iter_num==0):
        for i in range(0,iter_num):
            Qnew=Lloyd_iteration2(A,P,w,Qnew) 
            dists0=dists1
            dists1,Tags1,tagss=squaredis(P,Qnew) 
            conv=np.abs(np.sum(np.multiply(w,dists0))-np.sum(np.multiply(w,dists1)))/np.sum(np.multiply(w,dists1))
            print('conv',conv)

    else:     
        Qjl=np.zeros(Qnew.shape)   
        dists0=0
        dists1,Tags1,tagss=squaredis(P,Qnew)    
        i=0        
        conv=np.abs(np.sum(np.multiply(w,dists0))-np.sum(np.multiply(w,dists1)))/np.sum(np.multiply(w,dists1))
        while conv>iter_num:
            Qjl=Qnew
            Qnew=Lloyd_iteration2(A,P,w,Qjl)    
            i=i+1      
            dists0=dists1
            dists1,Tags1,tagss=squaredis(P,Qnew)
            print(np.sum(np.multiply(w,dists1))/500)
            conv=np.abs(np.sum(np.multiply(w,dists0))-np.sum(np.multiply(w,dists1)))/np.sum(np.multiply(w,dists1))
            print('conv',conv)
    print('&&&&&&',len(np.unique(tagss)))
    Q=SM(A)[tagss,:]
    return Qnew,w 

def myminibatch(Data0,weights,cent_num,num_of_iterations,mini_batch_size):
    """
    Implements the Kmeans minibatch algorithm, calculates a very fast clustering by working on Data sampling.
    input: Data: n*d matrix of points, each line in a point in a dim d.
           cent_num: k, The desired size of the clustering
           num_of_iterations,mini_batch_size: an internal algorithm's parameters, smaller will be faster but might lead a less accurate result.
    output : Cent: k*d a matrix of centroids, each line in a point in a dim d.
             nor_Cent : normalized centroid, each line of Cent multiplied by its cluster size. 
    
    """
    num_of_samples=Data0.shape[0]
    cent_num=int(cent_num)
    per=np.random.permutation(num_of_samples)
    Cent=Data0[per[0:cent_num],:]
    vol=np.zeros(cent_num)
    for i in range (num_of_iterations):
        per1=np.random.permutation(num_of_samples)
        M=Data0[per1[0:mini_batch_size],:]
        if is_sparse==1:
            dists,Tags,t=squaredis(make_P(M),Cent)
        else:
            dists,Tags,t=squaredis_dense(make_P(M),Cent)

        for j in range (mini_batch_size):
            c=Tags[j]
            vol[c]=vol[c]+weights[per1[j]]
            eta=1/vol[c]
            Cent[c,:]=(1-eta)*Cent[c,:]+eta*M[j,:]
    return Cent,vol

def Coreset_FBL(P,w,B,is_sparse=0):
    w=np.ravel(w)
    #print('wwwwww',w.shape)
    """
    Input:
    P: Data matrix n*d
    w: n weights
    Bsize: size of beta approximation sampling
    m: size of coreset
    alg: algorithm to operate: 0- Benchmark 1-ours 2-ransac
    Output:
    S: coreset matrix m*d
    S_ind: m indeices of rows chosen
    u: sensitivity of every point  
    """
    Bsize=B.shape[0]
    partition=[]

    if is_sparse==1:
        P2=make_P(P)#.multiply(w1)
        dists,Tags,_=squaredis(P2,B)
    else:
        P2=make_P_dense(P)#.multiply(w1)

        dists,Tags,t=squaredis_dense(P2,B)
    dists[dists<0]=0
    sum_weights_cluster=np.zeros(Bsize)
    for t in range (0,Bsize):
        sum_weights_cluster[t]=np.sum(w[np.where(Tags==t)[0]])
        partition.append(np.where(Tags==t)[0])

    sumall=2*np.sum(np.multiply(w,dists))
    sumwei=2*Bsize*sum_weights_cluster[Tags]        
    A=np.multiply(w,dists)/sumall
    print('A',A.shape)
    AA=np.divide(w,sumwei)
    print('AA',AA.shape)

    Prob=AA+A
    return Prob,partition,sum_weights_cluster

def FBL_median(Prob,P,w,Q,B,partition,sum_weights_cluster,coreset_size1,posi,is_sparse):
    coreset_size=coreset_size1-B.shape[0]
    S_ind=np.random.choice(len(Prob),coreset_size,p=Prob)
    print('11111',w.shape)
    print('11111',Prob.shape)

    uw=np.divide(np.ravel(w),Prob)/coreset_size
    if posi==1:
        uw=FBL_positive(P,uw,B,Q,0.1,is_sparse)
    u0=uw[S_ind]
    u2=np.zeros(len(partition))
    for i in range(len(partition)):
        u2[i]=np.sum(w[np.intersect1d(S_ind,partition[i])])
    u1=sum_weights_cluster-u2
    #print('len u',len(np.concatenate((u0,u1))))
    #print('len S',len(S_ind))
    print('u0=',u0.shape)
    print('u1=',u1.shape)

    return S_ind,np.concatenate((u0,u1),0)

def FBL_positive(P,u,B,Q,epsit,is_sparse=1):
     if is_sparse==0:    
         BB=make_P_dense(B)
         d1,_,_=squaredis_dense(BB,P)
         d2,_,_=squaredis_dense(BB,Q)
     else:
         BB=make_P(B)
         d1,_,_=squaredis(BB,P)
         d2,_,_=squaredis(BB,Q)
     d=d1-d2/epsit
     u[np.where(d<0)[0]]=0
     return u
def FBL(P0,P,Prob,partition,sum_weights_cluster,w,indsB,Q,coreset_size,is_not_sparse,full_sampling,posi):
    Prob=Prob/np.sum(Prob)
    if is_not_sparse==0:
        P0=SM(P0)
#    indsB=indsB.astype(int)
    if full_sampling==1:
        ind=np.random.choice(np.arange(len(Prob)),coreset_size,p=np.ravel(Prob)) 
       # print('@@@@@@@@@@@@@@@@@@@',len(Prob))
       # print('@@@@@@@@@@@@@@@@@@@',coreset_size)
       # print('@@@@@@@@@@@@@@@@@@@',ind)
       # print('@@@@@@@@@@@@@@@@@@@',len(np.unique(ind)))

        u=np.divide(np.ravel(w),Prob)/coreset_size    
        if posi==1:
            print('is_not_sparse',is_not_sparse)
            if is_not_sparse==0:
                u=FBL_positive(P0,u,P0[indsB,:],Q,0.1,1-is_not_sparse)
            else:
                u=FBL_positive(P,u,P[indsB,:],Q,0.1,1-is_not_sparse)
        u1=u[ind]
        print('u1',u1.shape)
        u1=np.reshape(u1,(u1.shape[0],1))  
        
    else:
        if is_not_sparse==0:

            ind,u1=FBL_median(Prob,P0,w,Q,P0[indsB,:],partition,sum_weights_cluster,coreset_size,posi,1-is_not_sparse)
        else:
            ind,u1=FBL_median(Prob,P,w,Q,P[indsB,:],partition,sum_weights_cluster,coreset_size,posi,1-is_not_sparse)

    ind=ind.astype(int)
    u1=np.reshape(u1,(len(u1),1))
        
    if full_sampling==0:
        if is_not_sparse==0:
            #if len(indsB)==2:
              #  indsB=indsB[1]
            print('indsB shape',indsB.shape)                        
            print('ind shape',len(ind))


            print(indsB)
            X=vstack((P0[ind,:],P0[indsB,:]))
            #print('B shape',B.shape)
        else:
            X=np.concatenate((P0[ind,:],P0[indsB,:]),0)

    else:
            X=P0[ind,:]

    if is_not_sparse==0:
           print('u1s',u1.shape)
           print('Xs',X.shape)

           C=X.multiply(u1)
    else:
           C=np.multiply(u1[:X.shape[0]],X)    
    print('\\\\\\\\\\\\\\\\',len(np.unique(ind)))
    return C,u1[:X.shape[0]],X #for streaming flip X and C.
      
def Coreset_FBL1(P,w,B,is_sparse=0):
    w=np.ravel(w)
    #print('wwwwww',w.shape)
    """
    Input:
    P: Data matrix n*d
    w: n weights
    Bsize: size of beta approximation sampling
    m: size of coreset
    alg: algorithm to operate: 0- Benchmark 1-ours 2-ransac
    Output:
    S: coreset matrix m*d
    S_ind: m indeices of rows chosen
    u: sensitivity of every point  
    """
    Bsize=B.shape[0]
    partition=[]

    if is_sparse==1:
        P2=make_P(P)#.multiply(w1)
        dists,Tags,_=squaredis(P2,B)
    else:
        P2=make_P_dense(P)#.multiply(w1)

        dists,Tags,t=squaredis_dense(P2,B)
    dists[dists<0]=0
    sum_weights_cluster=np.zeros(Bsize)
    for t in range (0,Bsize):
        sum_weights_cluster[t]=np.sum(w[np.where(Tags==t)[0]])
        partition.append(np.where(Tags==t)[0])

    sumall=2*np.sum(np.multiply(w,dists))
    sumwei=2*Bsize*sum_weights_cluster[Tags]        
    A=np.multiply(w,dists)/sumall
    print('A',A.shape)
    AA=np.divide(w,sumwei)
    print('AA',AA.shape)

    Prob=AA+A
    return Prob,partition,sum_weights_cluster

def FBL_median1(Prob,P,w,Q,B,partition,sum_weights_cluster,coreset_size1,posi,is_sparse):
    coreset_size=coreset_size1-B.shape[0]
    S_ind=np.random.choice(len(Prob),coreset_size,p=Prob)
    uw=np.divide(w,Prob)/coreset_size
    if posi==1:
        uw=FBL_positive(P,uw,B,Q,0.1,is_sparse)
    u0=uw[S_ind]
    u2=np.zeros(len(partition))
    for i in range(len(partition)):
        u2[i]=np.sum(w[np.intersect1d(S_ind,partition[i])])
    u1=sum_weights_cluster-u2
    #print('len u',len(np.concatenate((u0,u1))))
    #print('len S',len(S_ind))
    print('u0=',u0.shape)
    print('u1=',u1.shape)

    return S_ind,np.concatenate((u0,u1),0)

def FBL_positive1(P,u,B,Q,epsit,is_sparse=1):
     if is_sparse==0:    
         BB=make_P_dense(B)
         d1,_,_=squaredis_dense(BB,P)
         d2,_,_=squaredis_dense(BB,Q)
     else:
         BB=make_P(B)
         d1,_,_=squaredis(BB,P)
         d2,_,_=squaredis(BB,Q)
     d=d1-d2/epsit
     u[np.where(d<0)[0]]=0
     return u
def FBL1(P0,P,Prob,partition,sum_weights_cluster,w,indsB,Q,coreset_size,is_not_sparse,full_sampling,posi):
    if is_not_sparse==0:
        P0=SM(P0)
    indsB=indsB.astype(int)
    if full_sampling==1:
        ind=np.random.choice(len(Prob),coreset_size,p=np.ravel(Prob)) 
        u=np.divide(np.ravel(w),Prob)/coreset_size    
        if posi==1:
            print('is_not_sparse',is_not_sparse)
            if is_not_sparse==0:
                u=FBL_positive(P0,u,P0[indsB,:],Q,0.1,1-is_not_sparse)
            else:
                u=FBL_positive(P,u,P[indsB,:],Q,0.1,1-is_not_sparse)
        u1=u[ind]
        print('u1',u1.shape)
        u1=np.reshape(u1,(u1.shape[0],1))  
        
    else:
        if is_not_sparse==0:

            ind,u1=FBL_median(Prob,P0,w,Q,P0[indsB,:],partition,sum_weights_cluster,coreset_size,posi,1-is_not_sparse)
        else:
            ind,u1=FBL_median(Prob,P,w,Q,P[indsB,:],partition,sum_weights_cluster,coreset_size,posi,1-is_not_sparse)

    ind=ind.astype(int)
    u1=np.reshape(u1,(len(u1),1))
        
    if full_sampling==0:
        if is_not_sparse==0:
            #if len(indsB)==2:
              #  indsB=indsB[1]
            print('indsB shape',indsB.shape)                        
            print('ind shape',len(ind))


            print(indsB)
            X=vstack((P0[ind,:],P0[indsB,:]))
            #print('B shape',B.shape)
        else:
            X=np.concatenate((P0[ind,:],P0[indsB,:]),0)

    else:
        if is_not_sparse==0:
            X=P0[ind,:]
            C=X.multiply(u1)
        else:
            X=P0[ind,:]
            C=np.multiply(u1,X)
    if is_not_sparse==0:
           print('u1s',u1.shape)
           print('Xs',X.shape)

           C=X.multiply(u1)
    else:
           C=np.multiply(u1,X)    
    return C,u1,X   
def clus_streaming(path,Data,j,is_pca,alg,h,spar,trial=None,datum=None,is_jl=1,gamma1=0.000000001):
    """
    alg=0 unif sampling
    alg=1 Sohler
    alg=2 CNW
    alg=3 Alaa
    """
    sizeB=j
    coreset_size=Data.shape[0]//(2**(h+1))
    gamma=j/(coreset_size-j+1)
    #Q=[]
    

    k=0
    T_h= [0] * (h+1)
    DeltaT_h= [0] * (h+1)
    u_h=[0]* (h+1) 
    leaf_ind=np.zeros(h+1)
    iter_num=1
    for jj in range(np.power(2,h)): #over all of the leaves
        w=np.ones(2*coreset_size)
        #w=w/np.sum(w)
        print('alg',alg)
        print('trial',trial)
        print('jj',jj)
        Q0=Data[k:k+2*coreset_size,:]       
        if alg>0:

            #B,inds=kmeans_plspls_FBL(Q0,w,sizeB)  
            B,inds= kmeans_plspls1(Q0,np.ravel(w),0,[],sizeB,np.ravel(w),0.01,1,0)

            print('inds of B',inds)
            Prob,partition,sum_weights_cluster=Coreset_FBL(Q0,w,B,1)  
    
        if alg>1:
                Q1,dists11=k_means_clustering(Q0,w,j,iter_num)

    
        k=k+2*coreset_size
        print('k',k)
        if alg==0:
            #T=Data[0:coreset_size,:]
            T=Q0[np.random.choice(Q0.shape[0],coreset_size),:]
            w=np.ones((T.shape[0],1))
            #w=w/np.sum(w)
        if alg==1:
            print('^^^^^^^^^^^^^^^',coreset_size)
            _,w,T=FBL(Q0,Q0,Prob,partition,sum_weights_cluster,w,inds,[],coreset_size,0,1,0)
            #w=w*T.shape[0]/np.sum(w)
            #w=np.ones((T.shape[0],1))

        if alg==2:
            _,w,T=FBL(Q0,Q0,Prob,partition,sum_weights_cluster,w,inds,Q1,coreset_size,0,0,0)
            print(w.shape)
        if alg==3:
            _,w,T=FBL(Q0,Q0,Prob,partition,sum_weights_cluster,w,inds,Q1,coreset_size,0,1,1)
        if alg==4:
            _,w,T=FBL(Q0,Q0,Prob,partition,sum_weights_cluster,w,inds,Q1,coreset_size,0,0,1)
        DeltaT=0
        #w=w/2
        #print('T[0]',T[0,:])
        print('leaf num',jj)
        print('j=',j)
        #print('eps=',eps)
        #print('h=',h)
        i=0
                        
        u_h[0]=w
        while (i<h)*(type(T_h[i])!=int): #every time the leaf has a neighbor leaf it should merged and reduced
            print('111111',w.shape)
            print('22222',np.asarray(u_h[i]).shape)
            wT=np.concatenate((w,np.asarray(u_h[i])),0)

            if spar==0:
               totT=np.concatenate((T,np.asarray(T_h[i])),0)
            else: 
               totT0=vstack((T,T_h[i]))
               print(type(T))
               print(T.shape)

               print(type(T_h[i]))
               print(T_h[i].shape)
            totT0=SM(totT0)
            if alg>0:
                #B,inds=kmeans_plspls_FBL(totT0,wT,sizeB) 
                B,inds= kmeans_plspls1(totT0,np.ravel(wT),0,[],sizeB,np.ravel(wT),0.01,1,0)

                print('************************',wT.shape)
                Prob,partition,sum_weights_cluster=Coreset_FBL(totT0,wT,B,1)  
                print('!!!!!!',Prob.shape)
            if alg>1:
                Q1,dists11=k_means_clustering(totT0,wT,j,iter_num)
            if alg==0:
                T=totT0[np.random.choice(totT0.shape[0],coreset_size),:]
                w=np.ones((T.shape[0],1))
                #w=w/np.sum(w)

                #w=w/np.sum(w)
            if alg==1:
                T1,w,T=FBL(totT0,totT0,Prob,partition,sum_weights_cluster,wT,inds,[],coreset_size,0,1,0)
                #w=w*T.shape[0]/np.sum(w)

                #w=w/np.sum(w)

            if alg==2:
                T1,w,T=FBL(totT0,totT0,Prob,partition,sum_weights_cluster,wT,inds,Q1,coreset_size,0,0,0)
            if alg==3:
                T1,w,T=FBL(totT0,totT0,Prob,partition,sum_weights_cluster,wT,inds,Q1,coreset_size,0,1,1)
            if alg==4:
                T1,w,T=FBL(totT0,totT0,Prob,partition,sum_weights_cluster,wT,inds,Q1,coreset_size,0,0,1)
            print('type T',T.shape)

            DeltaT=0
        

            print('i=',i)
            print('j=',j)
#           print('eps=',eps)
        
            u_h[i]=0
    
            DeltaT=DeltaT+0 #zeroing leaf which reduced
            T_h[i]=0
            DeltaT_h[i]=0
            leaf_ind[i]=leaf_ind[i]+1
            i=i+1
        T_h[i]=T
        u_h[i]=w
        print('T shape',T.shape)
        print('w shape',w.shape)          
        T1=T.multiply(w)
        
        if spar==0:            
            if datum==0:
                np.save(path+'leaves_gyro1/trial='+str(trial)+',j='+str(j)+',alg='+str(alg)+',floor='+str(i)+',leaf='+str(leaf_ind[i])+'.npy',T)
            if datum==1:
                np.save(path+'leaves_acc1/trial='+str(trial)+',j='+str(j)+',alg='+str(alg)+',floor='+str(i)+',leaf='+str(leaf_ind[i])+'.npy',T)
            if datum==2:
                np.save(path+'leaves_mnist/trial='+str(trial)+',j='+str(j)+',alg='+str(alg)+',floor='+str(i)+',leaf='+str(leaf_ind[i])+'.npy',T)
        
        else:

                ssp.save_npz(path+'trial='+str(trial)+',j='+str(j)+',alg='+str(alg)+',floor='+str(i)+',leaf='+str(leaf_ind[i])+'.npz',T)
                np.save(path+'trial='+str(trial)+',j='+str(j)+',alg='+str(alg)+',floor='+str(i)+',leaf='+str(leaf_ind[i])+'_weights.npy',w)
        DeltaT_h[i]=DeltaT
        Q=[]
        #print(T_h)
        
    if type(T_h[h])==int: #should be remained only the upper one. if not:
        all_levels=[]
        for g in range (h+1): #collecting all leaves which remained on tree.
            if type(T_h[g])!=int:
                if all_levels==[]:
                   all_levels=np.asarray(T_h[g])
                else:
                    all_levels=np.concatenate((all_levels,np.asarray(T_h[g])),0)
        DeltaT_hs=sum(DeltaT_h[h]) #summing its delta
    else:
        all_levels=T_h[h] 
        DeltaT_hs=DeltaT_h[h]
    return all_levels
