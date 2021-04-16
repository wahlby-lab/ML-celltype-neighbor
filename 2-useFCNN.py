import os
import numpy as np
import pandas as pd
import utils
import seaborn as sns
import importlib
import warnings
import sklearn
import sklearn.metrics
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import fcnnpy

warnings.filterwarnings("ignore")
importlib.reload(utils)
importlib.reload(fcnnpy)
pd.options.display.max_columns = None
pd.options.display.max_rows = 100

#all d90s and all the values computed in qupath
featurecols=['12d90s', '13d90s', '14d90s', '15d90s', '16d90s', '17d90s', '18d90s', '23d90s', '24d90s', '25d90s', '26d90s', '27d90s', '28d90s', '34d90s', '35d90s', '36d90s', '37d90s', '38d90s', '45d90s', '46d90s', '47d90s', '48d90s', '56d90s', '57d90s', '58d90s', '67d90s', '68d90s', '78d90s','Cell: DAPI mean','Cell: DAPI std dev', 'Cell: DAPI max', 'Cell: DAPI min','Cell: IBA1 mean', 'Cell: IBA1 std dev', 'Cell: IBA1 max','Cell: IBA1 min', 'Cell: Ki67 mean', 'Cell: Ki67 std dev','Cell: Ki67 max', 'Cell: Ki67 min', 'Cell: TMEM119 mean','Cell: TMEM119 std dev', 'Cell: TMEM119 max', 'Cell: TMEM119 min','Cell: NeuroC mean', 'Cell: NeuroC std dev', 'Cell: NeuroC max','Cell: NeuroC min', 'Cell: MBP mean', 'Cell: MBP std dev','Cell: MBP max', 'Cell: MBP min', 'Cell: mutIDH1 mean','Cell: mutIDH1 std dev', 'Cell: mutIDH1 max', 'Cell: mutIDH1 min','Cell: CD34 mean', 'Cell: CD34 std dev', 'Cell: CD34 max','Cell: CD34 min', 'Cell: GFAP mean', 'Cell: GFAP std dev','Cell: GFAP max', 'Cell: GFAP min', 'Cell: Autofluorescence mean','Cell: Autofluorescence std dev', 'Cell: Autofluorescence max','Cell: Autofluorescence min']

classcols=['is_Astrocyte', 'is_Glioma', 'is_Neuron', 'is_Microglia', 'is_Macrophage', 'is_Endothelial','is_ambiguous']

cores_train=['8_1_C', '5_10_I',  '5_10_F', '8_12_B', '6_3_B', '7_1_E', '5_1_A',
              '5_9_D', '8_1_B', '5_12_H' , '5_3_C' ,'7_7_B']

cores_test =['7_3_A', '5_11_I', '8_2_A', '8_3_B',  '5_12_E', '5_4_D',  '5_9_D',
             '5_10_B', '7_5_D', '5_12_H', '8_5_B' ]
       
location="/mnt/hdd1/users/leslie/glioma/"
onlyfile="data/onlyRELabeledFullfeatured.csv"
onlylabeled=pd.read_csv(location+onlyfile)
testdf=onlylabeled[onlylabeled["Image"].isin(cores_test)].copy()

dataset_test = fcnnpy.d90sDataset(testdf,featurecols,classcols,onehotlabels=True)

test_loader  = torch.utils.data.DataLoader(dataset_test,  batch_size=64, shuffle=True, num_workers=12)

alllosses,allbestepochs,allids,allseeds,allsaves =[],[],[],[],[]

n_input, n_hidden1, n_hidden2, n_hidden3, n_output = len(featurecols), 100,200,300, len(classcols)

log_interval=100


saveat=location+"models/"

useold=False
df=None
if os.path.isfile(saveat+"summary.csv"):
    df=pd.read_csv(saveat+"summary.csv")
    useold=True

traindf=onlylabeled[onlylabeled["Image"].isin(cores_train)]

amountperclass=5000
batch_size=500
optimizer="AdaBelief"
epochs=35

i=0
while(i<1000):
    modelid="model"+str(i)
    seed=np.random.randint(1,9999999)
        
    bestl,bestepoch,savemodel,modelid=fcnnpy.trainFCNwithnewdataperepoch(n_input, n_hidden1, n_hidden2,n_hidden3, n_output, 
                    traindf,batch_size,featurecols,classcols,amountperclass, test_loader, 
                  "MultiMarginLoss", optimizer, epochs, seed, log_interval, modelid, saveat)

    alllosses.append(bestl)
    allbestepochs.append(bestepoch)
    allids.append(modelid)
    allseeds.append(seed)
    allsaves.append(savemodel)

    d={"modelid":allids,"losses":alllosses, "bestepoch":allbestepochs, "seed":allseeds, "saved":allsaves}
    dfi = pd.DataFrame(d)
    
    if useold:
        dfi=pd.concat([df, dfi])

    dfi.to_csv(saveat+"summary.csv",index=False)
 
    i+=1
