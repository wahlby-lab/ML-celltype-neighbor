import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from adabelief_pytorch import AdaBelief
import numpy as np
import sklearn
import sklearn.metrics
import utils
import pandas as pd

class Net(nn.Module):
    def __init__(self,n_input,n_hidden1,n_hidden2,n_hidden3,n_output):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden1)
        self.bn1 = nn.BatchNorm1d(n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.bn2 = nn.BatchNorm1d(n_hidden2)
        self.fc3 = nn.Linear(n_hidden2, n_hidden3)
        self.bn3 = nn.BatchNorm1d(n_hidden3)
        self.fco = nn.Linear(n_hidden3, n_output,bias=False)
    def forward(self, x):
        x = self.fc1(x)
        #x = self.bn1(x)
        x = F.relu(x)
        #x = F.relu(self.fc1(x))

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)

        #x = F.relu(self.fc2(x))
        x = F.relu(self.fco(x))
        return x #F.softmax(x)


class d90sDataset(Dataset):
    def __init__(self, df,fcols,ccols,inference=False,onehotlabels=False):
        self.data = df
        self.featurecols = fcols
        self.classcols = ccols
        self.onehotlabels=onehotlabels
        self.inference=inference

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        info = self.data.iloc[idx]
        x=info[self.featurecols].to_numpy().astype("float")
        x=torch.from_numpy(x)
        if(self.inference):
            return x
        y=info[self.classcols].to_numpy().astype("int")
        
        x=x.float()
        if not self.onehotlabels:
            y=torch.tensor(np.argmax(y))
            return x,y
        
        y=torch.from_numpy(y)
        
        return x, y


def d90sTrainingPerturbed(df,nsamples,fcols,ccols,scaleperfeature=False,scale=1e-2, scalescale=1.0):
    """Get a slightly randomly perturbed df with 
    equal number of samples per class"""
    #Get the names of the classes and their counts, 
    #for simplicity this expect that columns are one hot encoded

    stats=df[fcols].describe()
    
    dflist=[]
    for c in ccols:
        #for each class in ccols
        indf=df[df[c]==1]
        thedf=indf.sample(nsamples,replace=True)
        if scaleperfeature:
            allrands=np.zeros((len(thedf), len(fcols)),dtype="float")
            i=0
            for f in fcols:
                inscale=stats.loc["std"][f]
                allrands[:,i]=np.random.normal(loc=0.0, scale=inscale/scalescale, size=(len(thedf)) )
                i+=1
            thedf[fcols]+allrands
        else:
            thedf[fcols]+=np.random.normal(loc=0.0, scale=scale, size=(len(thedf), len(fcols)) )
        dflist.append(thedf)
        
    fulldf=pd.concat(dflist)
    return fulldf

def trainFCNwithnewdataperepoch(n_input, n_hidden1, n_hidden2,n_hidden3, n_output, train_df, bsize, fcols, ccols, nsamples, test_loader, 
                    criterion, optimizer, epochs,seed,log_interval, modelid, saveat, wholedata=None):

    torch.manual_seed(seed)
    net = Net(n_input, n_hidden1, n_hidden2,n_hidden3, n_output)
    net.cuda()

    bestl=0.1
    bestepoch=0

    crit=None
    if(criterion=="MultiMarginLoss"):
        crit= nn.MultiMarginLoss()

    opt=None
    if(optimizer=="AdaBelief"):
        opt = AdaBelief(net.parameters(), lr=1e-4,weight_decay=1e-4, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = False, print_change_log = False)
    if optimizer=="SGD":
        opt = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    savemodel=False
    strike=0
    strikesneeded=50
    for epoch in range(epochs):
        print("Creating new augmented traindf")
        newtraindf=d90sTrainingPerturbed(train_df,nsamples,fcols,ccols,scaleperfeature=True,scalescale=50)
        dataset_train = d90sDataset(newtraindf,fcols,ccols,onehotlabels=True)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=bsize, shuffle=True, num_workers=12)

        train_loss, valid_loss = [], []
        vyhs,vys=[],[]
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.cuda().to(non_blocking=True)
            target = target.cuda().to(non_blocking=True)
            target=torch.argmax(target, dim=1)

            opt.zero_grad()
            net_out = net(data.float())
            loss = crit(net_out, target)
            loss.backward()
            opt.step()
            
            train_loss.append(loss.item())
            
            if batch_idx % log_interval == 0:     
                print (modelid+", batch:", batch_idx, "Training Loss: ", np.mean(train_loss))
                
        with torch.no_grad():
            print("Eval")
            net.eval()
            savemodel=True
            for data, target in test_loader:
                data = data.cuda().to(non_blocking=True)
                target = target.cuda().to(non_blocking=True)
                target=torch.argmax(target, dim=1)
                output = net(data.float())
                sout= F.log_softmax(output)
                yhat=torch.argmax(sout, dim=1)
                
                vys.append(target.cpu().numpy())
                vyhs.append(yhat.cpu().numpy())
                eloss = crit(output, target)
                valid_loss.append(eloss.item()) 

            npvaloss=np.array(valid_loss)
            finalvalidloss=npvaloss.mean()
                
            meantrainloss=np.mean(train_loss)
            meanvalidloss=np.mean(valid_loss)

            strloss=str(meantrainloss)
            strloss=strloss[:7]

            if finalvalidloss < bestl:
                bestl=finalvalidloss
                bestepoch=epoch

            
            cm = sklearn.metrics.confusion_matrix(np.concatenate(vys), np.concatenate(vyhs))  
            #inlimits=[1200,17000,326,100,2750,800]
            
            maxes=[1532,17085,817,186,2711,789,4863]

            #maxes=[1014,14300,422,187,2720,767]
            printmsg=""
            for ci in range(len(ccols)):
                printmsg+=f'{cm[ci][ci]}/{maxes[ci]},'
            print(printmsg)

            avg=0
            for a in range(len(maxes)):
                avg+=cm[a][a]/maxes[a]
            avg/=len(maxes)
            print(avg)


            filename=""
            if (meantrainloss + 0.009 < bestl) and (epoch < epochs-1):
                bestl=meantrainloss
                filename=saveat+modelid+"inter-"+str(epoch)+"-"+strloss
                torch.save(net.state_dict(), filename+".pth" )
                cm = sklearn.metrics.confusion_matrix(np.concatenate(vys), np.concatenate(vyhs)) 
                np.save(filename+"CM.npy",cm)


            if epoch==epochs-1:
                filename=saveat+modelid+"last-"+str(epoch)+"-"+strloss
                torch.save(net.state_dict(), filename+".pth" )
                cm = sklearn.metrics.confusion_matrix(np.concatenate(vys), np.concatenate(vyhs)) 
                np.save(filename+"CM.npy",cm)
                #utils.plot_confusion_matrix(cm, ['is_Astrocyte', 'is_Glioma', 'is_Neuron', 'is_Microglia','is_Macrophage', 'is_Endothelial'],save=saveat+"model-"+modelid+"-"+str(epoch)+"-"+strloss+".png")

            
            print (modelid+": Epoch:", epoch, "Training Loss: ", np.mean(train_loss), "Valid Loss: ", np.mean(valid_loss))

    print ("Best for "+modelid+" "+str(bestepoch)+" loss:"+str(bestl)+"\n")
    
    return bestl, bestepoch, savemodel,modelid,filename



def trainanetwork(n_input, n_hidden1, n_hidden2, n_output, train_loader, test_loader, 
                    criterion, optimizer, epochs,seed,log_interval, modelid, saveat):

    torch.manual_seed(seed)
    net = Net(n_input, n_hidden1, n_hidden2, n_output)
    net.cuda()
    print(net)

    bestl=999.0
    bestepoch=0

    crit=None
    if(criterion=="MultiMarginLoss"):
        crit= nn.MultiMarginLoss()

    opt=None
    if(optimizer=="AdaBelief"):
        opt = AdaBelief(net.parameters(), lr=5e-4,weight_decay=1e-4, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = False)

    savemodel=False
    strike=0
    strikesneeded=3
    for epoch in range(epochs):
        train_loss, valid_loss = [], []
        vyhs,vys=[],[]
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.cuda().to(non_blocking=True)
            target = target.cuda().to(non_blocking=True)
            target=torch.argmax(target, dim=1)

            opt.zero_grad()
            net_out = net(data.float())
            loss = crit(net_out, target)
            loss.backward()
            opt.step()
            
            train_loss.append(loss.item())
            
            if batch_idx % log_interval == 0:     
                print ("batch:", batch_idx, "Training Loss: ", np.mean(train_loss))
                
        with torch.no_grad():
            net.eval()
            savemodel=False
            for data, target in test_loader:
                data = data.cuda().to(non_blocking=True)
                target = target.cuda().to(non_blocking=True)
                target=torch.argmax(target, dim=1)
                output = net(data.float())
                sout= F.log_softmax(output)
                yhat=torch.argmax(sout, dim=1)
                
                vys.append(target.cpu().numpy())
                vyhs.append(yhat.cpu().numpy())
                eloss = crit(output, target)
                valid_loss.append(eloss.item()) 

            npvaloss=np.array(valid_loss)
            finalvalidloss=npvaloss.mean()
                
            strloss=str(finalvalidloss)
            strloss=strloss[:5]

            if finalvalidloss < bestl:
                bestl=finalvalidloss
                bestepoch=epoch
                #savemodel=True

            if epoch>1:
                cm = sklearn.metrics.confusion_matrix(np.concatenate(vys), np.concatenate(vyhs))  
                inlimits=[730,9900,500,100,1500,320]
                print(cm[0][0], cm[1][1], cm[2][2], cm[3][3], cm[4][4], cm[5][5])
                print((cm[0][0] >= inlimits[0]) , (cm[1][1] >= inlimits[1]), (cm[2][2] >= inlimits[2]),
                        (cm[3][3] >= inlimits[3]) , (cm[4][4] >= inlimits[4]) , (cm[5][5] >= inlimits[5]) ) 
                if((cm[0][0] >= inlimits[0]) and (cm[1][1] >= inlimits[1]) and (cm[2][2] >= inlimits[2]) and
                             (cm[3][3] >= inlimits[3]) and (cm[4][4] >= inlimits[4]) and (cm[5][5] >= inlimits[5]) ):
                    savemodel=True
                    print("Saving "+modelid)
                else:
                    strike+=1
                    if strike >=strikesneeded:
                        print ("Strike "+str(strikesneeded)+" for "+modelid+": Epoch:", epoch, "Training Loss: ", np.mean(train_loss), "Valid Loss: ", np.mean(valid_loss))
                        if(finalvalidloss <= 0.08):
                            print("But final valid loss is "+str(finalvalidloss)+" <= 0.075, so save to observe")
                            savemodel=True
                            modelid+="W" #W for weird that has such a low loss but didn't ahve a good cm diagonal in test

            if savemodel and epoch>0:
                torch.save(net.state_dict(), saveat+"model-"+modelid+"-"+str(epoch)+"-"+strloss+".pth" )
                cm = sklearn.metrics.confusion_matrix(np.concatenate(vys), np.concatenate(vyhs))  
                utils.plot_confusion_matrix(cm, ['is_Astrocyte', 'is_Glioma', 'is_Neuron', 'is_Microglia','is_Macrophage', 'is_Endothelial'],save=saveat+"model-"+modelid+"-"+str(epoch)+"-"+strloss+".png")
            if strike >=strikesneeded:
                return bestl, bestepoch, savemodel,modelid
    
            
            print (modelid+": Epoch:", epoch, "Training Loss: ", np.mean(train_loss), "Valid Loss: ", np.mean(valid_loss))

    print ("Best for "+modelid+" "+str(bestepoch)+" loss:"+str(bestl)+"\n")
    
    return bestl, bestepoch, savemodel,modelid


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
