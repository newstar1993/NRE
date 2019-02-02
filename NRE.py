import tensorflow as tf
import xgboost as xgb
import numpy as np
import pandas as pd
import scipy.io
import re 
import random

from sklearn.preprocessing import  StandardScaler
from sklearn.metrics import accuracy_score 


_NODEPAT = re.compile(r'(\d+):\[f(.+)<(.+)\]')
_LEAFPAT = re.compile(r'(\d+):leaf=(.+)')
_EDGEPAT = re.compile(r'yes=(\d+),no=(\d+),missing=(\d+)')
_EDGEPAT2 = re.compile(r'yes=(\d+),no=(\d+)')

def logloss(yTrue,yPred):
    a = np.array([1 if i==1 else 0 for i in yTrue])
    b = np.array([1./(1.+np.exp(-i)) for i in yPred])
    return np.sum(-(a*np.log(b) + (1.-a)*np.log(1.-b)))/len(yTrue)

def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))    
    grad = preds - labels
    hess = preds * (1.0-preds)
    return grad, hess

def fit_tree(dtrain,grad,hess,depth):
    param      = {'max_depth':depth, 'eta':0.1, 'silent':1}
    tree = xgb.train(param, dtrain, 1, [], lambda a,b:(grad,hess))
    return tree

def ruleMapping(tree,leaf):
    decisions = []
    nodes     = []
    leaves    = []
    edges     = []
    for i, text in enumerate(tree):
        if text[0].isdigit():
            node = parse_node(nodes,leaves,text)

        else:
            if i == 0:
                raise ValueError('Unable to parse given string as tree') # 1st string must be node
            _parse_edge(edges, node, text)
            
    
    edges    = np.array(edges)
    nodes    = dict([(int(i),(int(j),float(k))) for i,j,k in nodes])
    leaves   = dict([(int(i),float(j)) for i,j in leaves])

    parents       = edges[:,0]
    leftchildren  = edges[:,1] 
    rightchildren = edges[:,2]
    

    child = leaf
    while child != 0:
        if len(np.where(leftchildren==child)[0]):
            father     = parents[np.where(leftchildren==child)][0]
            feature,b0 = nodes[father]
            decisions.append([b0,-1,feature])
        else:
            father     = parents[np.where(rightchildren==child)][0]
            feature,b0 = nodes[father]
            decisions.append([-b0,1,feature])
        child = father
    
    return decisions, leaves[leaf]


def parse_node(nodes,leaves, text):
    
    match = _NODEPAT.match(text)
    if match is not None:
        node = match.group(1)
        nodes.append(match.groups())
        return node
    match = _LEAFPAT.match(text)
    if match is not None:
        node = match.group(1)
        leaves.append(match.groups())
        return node
    raise ValueError('Unable to parse node: {0}'.format(text))


def _parse_edge(edges, node, text):
    
    try:
        match = _EDGEPAT.match(text)
        if match is not None:
            yes, no, missing = match.groups()
            edges.append((int(node),int(yes),int(no)))
            return
    except ValueError:
        pass
    match = _EDGEPAT2.match(text)
    if match is not None:
        yes, no = match.groups()
        edges.append((int(node),int(yes),int(no)))
        return
    raise ValueError('Unable to parse edge: {0}'.format(text))

def features(dtrain,dtest,depth,Ntrees):
    param = {'max_depth':depth, 'eta':0.1, 'silent':0,'objective':'binary:logistic'}
    model = xgb.train(param, dtrain, Ntrees)
    
    ruleNos  = []
    testLeaf = model.predict(dtest,pred_leaf=True)
    trainLeaf= model.predict(dtrain,pred_leaf=True)
    
    if Ntrees == 1:
        testLeaf = testLeaf[:,np.newaxis]
        trainLeaf = trainLeaf[:,np.newaxis]
        
    k = 0
    trainpred = np.zeros(shape = (trainLeaf.shape[0],sum([np.unique(trainLeaf[:,i]).shape[0] for i in range(Ntrees)])))
    testpred  = np.zeros(shape = (testLeaf.shape[0],trainpred.shape[1]))
    
    for col in range(Ntrees):
        leafIds = np.unique(trainLeaf[:,col])
        leafIds.sort()
       
        tree      =  model.get_dump(fmap='')[col].split()

   
        for aleafId in leafIds:
            _, leafweight = ruleMapping(tree,aleafId)
            normalizedleafweight = np.array([1.0 if leafweight > 0.0 else -1.0])
            
            trainpred[:,k] = np.where(trainLeaf[:,col]==aleafId,normalizedleafweight,0)
            testpred[:,k]  = np.where(testLeaf[:,col] ==aleafId,normalizedleafweight,0)
            ruleNos.append((col,aleafId))
            k += 1
    
    return trainpred,testpred,model,ruleNos


def rules(xTrain,xTest,yTrain,yTest,complexityList=[(10,1,1)],includeLinearFeatures=True):
    standardize = StandardScaler()
    xTrain      = standardize.fit_transform(xTrain)
    xTest       = standardize.transform(xTest)
    dtrain      = xgb.DMatrix(xTrain,label=yTrain)
    dtest       = xgb.DMatrix(xTest,label=yTest)
    
    boostModels  = []
    ruleNos     = []
    modelIds  = range(xTrain.shape[1])
    
    if includeLinearFeatures:
        trainfeatures = [xTrain]
        testfeatures  = [xTest]
    else:
        trainfeatures = []
        testfeatures  = []                

    np.random.seed(100)
    for adepth,Nruns,Numtrees in complexityList:
        for arun in range(Nruns):
            if arun != 0:
                margin = np.random.randn(len(yTrain))
                dtrain.set_base_margin(margin)
            else:
                dtrain.set_base_margin(np.zeros_like(yTrain))
                
            trainpred,testpred,amodel,tempRuleNos = features(dtrain,dtest,adepth,Numtrees)
            ruleNos += tempRuleNos
            boostModels.append(amodel)
            modelIds += [-1*(len(boostModels))]*trainpred.shape[1]
            trainfeatures.append(trainpred)
            testfeatures.append(testpred)
        
    trainfeatures = np.concatenate(trainfeatures,axis=1)
    testfeatures  = np.concatenate(testfeatures,axis=1)
    
    return trainfeatures,testfeatures,yTrain,yTest,boostModels,modelIds,ruleNos

def nn_layer(x,w,b,lw,lb):
    weights     = tf.Variable(tf.constant(w))
    biases      = tf.Variable(tf.constant(b))
    preactivate = tf.matmul(x, weights) + biases

    activations = tf.nn.relu(preactivate)
    pooling     = tf.reduce_min(activations,axis=1)
        
    leafweights = tf.Variable(tf.constant(lw))
    output      = leafweights*pooling 
    return output

def main(xTrain,xTest,yTrain,yTest,depth):
    
    tf.reset_default_graph()
    
    includeLinearFeatures=False
    complexityList = [(depth,1,1)] #[(1,1,10),(2,1,10),(3,1,10),(4,1,10),(5,1,10),(6,1,10)]#
    trainpred,testpred,yTrain,yTest,boostModels,modelIds,ruleNos = rules(xTrain,xTest,yTrain,yTest,complexityList=complexityList,includeLinearFeatures=includeLinearFeatures)

    sel = range(trainpred.shape[1])
    b   = [0 for i in sel]

    treeAndRules = []
    selLinearFeats = []
    for j,coef in zip(sel,b):
        if not includeLinearFeatures:
            j += xTrain.shape[1]
        if j >= xTrain.shape[1]:
            amodel = boostModels[-modelIds[j]-1]
            treeNo, ruleNo = ruleNos[j-xTrain.shape[1]]
            treeAndRules.append((amodel.get_dump(fmap='')[treeNo],ruleNo,coef))
        else:
             selLinearFeats.append(j)

    standardize = StandardScaler()
    xTrain      = standardize.fit_transform(xTrain)
    xTest       = standardize.transform(xTest)
    dtrain      = xgb.DMatrix(xTrain,label=yTrain)
    dtest       = xgb.DMatrix(xTest,label=yTest)


    dtrain.set_base_margin(np.zeros(xTrain.shape[0]))
    dtest.set_base_margin(np.zeros(xTest.shape[0]))

    nnModels          = []
    DataList          = []
    featureIdsList    = []
    overallBias       = 0.0
    linearUnit        = np.zeros((len(selLinearFeats),1)).astype(np.float32)
    preds             = np.zeros_like(yTrain).astype(np.float32).squeeze()

    featureIds = []
    for tree,rule,fsaweight  in  treeAndRules:
        tree      = tree.split()

        decisions, leafweight = ruleMapping(tree,rule)

        featureIds += list(np.unique([k for i,j,k in decisions]))
    featureIds = np.unique(featureIds)
    print (featureIds)   
    grad,hess  = logregobj(preds,dtrain)



    for tree,rule,fsaweight  in  treeAndRules:
        tree      = tree.split()

        decisions, leafweight = ruleMapping(tree,rule)

        featureIds = featureIds
        w = np.zeros((len(featureIds),len(decisions)),dtype=np.float32)
        b = np.zeros(len(decisions),dtype=np.float32)

        featureIdsList.append(featureIds)
        for i,(bj,wj,fj) in enumerate(decisions):
            w[np.where(featureIds==fj),i] = wj
            b[i] =   bj 

        lw = 0.0
        lb =  leafweight

        nnModels.append((w,b,lw,lb))
        DataList.append(tf.placeholder(tf.float32, [None, w.shape[0]]))


    print (len(nnModels))

    xLin  = tf.placeholder(tf.float32,[None,len(selLinearFeats)])
    yTrue = tf.placeholder(tf.float32, [None], name='y-input')
    ob    = tf.Variable(tf.constant(overallBias))
    lu    = tf.Variable(tf.constant(linearUnit))
    yPred = tf.squeeze(ob + tf.matmul(xLin, lu) )

    modelParamList = []

    for dataNUM, (w,b,lw,lb) in enumerate(nnModels):
        out = nn_layer(DataList[dataNUM],w,b,lw,lb)
        yPred += out

    diff = tf.nn.sigmoid_cross_entropy_with_logits(labels=yTrue, logits=yPred)
    cross_entropy = tf.reduce_mean(diff) 
    train_step    = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

    with tf.name_scope('correct_prediction'):
          correct_prediction = tf.equal(tf.greater(yTrue, 0), tf.greater(yPred, 0))
    with tf.name_scope('accuracy'):
          accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    train_dict={j: xTrain[:,d] for j, d in zip(DataList, featureIdsList)}
    train_dict[xLin] = xTrain[:,selLinearFeats]
    train_dict[yTrue] = yTrain.squeeze()

    test_dict={j: xTest[:,d] for j, d in zip(DataList, featureIdsList)}
    test_dict[xLin] = xTest[:,selLinearFeats]
    test_dict[yTrue] = yTest.squeeze()


    with tf.Session() as sess:
    #sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        overallAcc = 0.0
        min_epoch = 0
        for step in range(1000):
            _,acc,loss = sess.run([train_step,accuracy,cross_entropy],feed_dict=train_dict)
            testacc,testloss = sess.run([accuracy,cross_entropy],feed_dict=test_dict)
            print ("epoch", step, overallAcc)
                
            if testacc > overallAcc:
                overallAcc  = max(testacc, overallAcc)
                min_epoch = step
            elif step - min_epoch >= 100:
                return overallAcc
            else:
                continue

    return overallAcc






