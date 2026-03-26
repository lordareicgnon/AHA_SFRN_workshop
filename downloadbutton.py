import numpy as np
import walk_likelihood as wl
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from time import time

class sparse_distance_matrix():
    def __init__(self,X):
        self.X=X
        self.shape=[X.shape[0],X.shape[0]]
        self.X2=np.sum(self.X*self.X,axis=1)

    def dot(self,b):
        prod = -2*self.X.dot((self.X).T.dot(b))
        prod+= self.X2.dot(b)
        if (len(b.shape)==1):
            pr=self.X2*sum(b)
        else:
            pr=np.outer(self.X2,sum(b))
        prod+=pr
        return prod

class easy_dot():
    def __init__(self,a):
        self.a=a
        self.shape=[a.shape[0],a.shape[0]]
        #print(np.max(np.sum(a,axis=1)))

    def dot(self,b):
        #return self.a.dot((self.a/sum(self.a)).T.dot(b))
        return self.a.dot((self.a).T.dot(b))

class VillageNet():
    def __init__(self, villages=60, normalize=1, neighbors=20):
        self.normalize=normalize
        self.villages=villages
        self.neighbors=neighbors

    def fit(self,X,comms=None,ref=None):
        self.X=X
        self.N=len(self.X)
        if self.normalize:
            self.X=(self.X-np.mean(self.X,axis=0))/(np.std(self.X,axis=0)+(np.std(self.X,axis=0)==0))
        t1=time()
        self.kmeans()
        t2=time()
        print('time='+str(t2-t1))
        self.grapher()
        t3=time()
        print('time='+str(t3-t2))
        self.get_communities(comms=comms)
        print('time='+str(time()-t3))
        if ref is not None:
            print(nmi(ref,self.comm_id))
        return self

    def kmeans(self):
        kmeans = KMeans(n_clusters=self.villages, random_state=13,n_init=1, max_iter=20,init='random').fit(self.X)
        #kmeans = KMeans(n_clusters=self.init_clusters, random_state=13, n_init=1).fit(self.X)
        self.labels=kmeans.labels_
        self.cluster_centers=kmeans.cluster_centers_

        self.ds=pairwise_distances(self.X,self.cluster_centers)**2
        self.distance_matrix = pairwise_distances(self.cluster_centers)
        np.fill_diagonal(self.distance_matrix,10000000000000000)
        self.U=np.zeros((self.N,self.villages))
        self.U[range(self.N),self.labels]=1
        #self.D=(self.ds-self.ds[range(self.N),self.labels][:,None])/self.distance_matrix[self.labels,:]
        Xv=self.X.dot(self.cluster_centers.T)
        v=self.cluster_centers.dot(self.cluster_centers.T)
        self.D = (-Xv+Xv[range(self.N),self.labels][:,None]+np.diagonal(v)-v[self.labels,:])/self.distance_matrix[self.labels,:]
        self.D = self.D - self.distance_matrix[self.labels,:]/2
        self.D[range(self.N),self.labels[range(self.N)]]=10000000000000000


    def grapher(self):
        self.M=self.U.copy()
        self.M[:,:]=0
        self.village_list=[]
        for i in range(self.U.shape[1]):
            arg =np.argpartition(self.D[:,i],self.neighbors)
            vl=[]
            for j in range(self.neighbors):
                self.M[arg[j],i]=1
                vl.append(arg[j])
            self.village_list.append(vl)

    def get_communities(self,thr_clusters=128,comms=None,**WLCF_args):
        self.A=(self.M.T).dot(self.U)
        self.A=self.A+self.A.T

        model=wl.walk_likelihood(self.A)

        if self.villages>thr_clusters:

            U=np.random.random((self.villages,thr_clusters))
            model.WLM(U=U,l_max=8,comms=comms,**WLCF_args)

        else:
            model.WLM(comms=comms,**WLCF_args)
        self.comm_id=model.comm_id[self.labels]

        return model
