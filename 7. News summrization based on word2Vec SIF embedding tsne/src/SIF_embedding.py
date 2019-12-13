import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



def get_weighted_average(We, x, w):
    """
    Compute the weighted average vectors
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in sentence i
    :param w: w[i, :] are the weights for the words in sentence i
    :return: emb[i, :] are the weighted average vector for sentence i
    """
    n_samples = x.shape[0]
    emb = np.zeros((n_samples, We.shape[1]))
   
    
    for i in range(n_samples):
    #for i in xrange(n_samples):
        emb[i,:] = w[i,:].dot(We[x[i,:],:]) / np.count_nonzero(w[i,:])
    return emb

def compute_pc(X,npc=1,method='PCA'):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    
    #print (np.max(X))
    
    if method=='PCA':
        model=PCA(n_components=npc, random_state=0)
        scaler = StandardScaler()
        X= scaler.fit_transform(X)
        model.fit(X)
        print ('PCA used for decompostion', model)
    else:
        model = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
        model.fit(X)
        print ('svd used for decompostion')
    return model.components_


    
    

def remove_pc(X, method,npc=1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    #print ('compute_pc start')
    pc = compute_pc(X, npc,method)
    #print ('compute_pc done')
    #print (X)
    #print ('pc', pc)
    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX


def SIF_embedding(We, x, w, params,method):
    """
    Compute the scores between pairs of sentences using weighted average + removing the projection on the first principal component
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in the i-th sentence
    :param w: w[i, :] are the weights for the words in the i-th sentence
    :param params.rmpc: if >0, remove the projections of the sentence embeddings to their first principal component
    :return: emb, emb[i, :] is the embedding for sentence i
    """
    #print ('started')
    #print ('We:' , We)
    emb = get_weighted_average(We, x, w)
    #print ('get_weighted_average done')
    print ('emb shape:' ,emb.shape)
    #print (emb)
    
    
    if  params.rmpc > 0:
 
        emb = remove_pc(emb, method,params.rmpc)
        
    #print ('We:', We.shape)
    #print ('x:', x.shape)
    #print ('w:', w.shape)
    return emb
