import json
import pandas as pd
import numpy as np
from skimage import io
from matplotlib.path import Path
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os
from scipy import stats
from datetime import datetime 
from sklearn.metrics.cluster import normalized_mutual_info_score,mutual_info_score
import warnings
warnings.filterwarnings("ignore")
pd.options.display.max_columns = None

def corrInMs(m1,m2):
    try:
        m1f=m1.flatten()
        m2f=m2.flatten()
        r,p=stats.pearsonr(m1f,m2f)
        return r,p
    except:
        print("failed at corrims",len(m1f),len(m2f))
        raise Exception

def d90s(m1,m2):
    try:
        m1f=np.percentile(m1.flatten(),90)
        m2f=np.percentile(m2.flatten(),90)
        val=m1f-m2f
        return val
    except:
        print("failed at d90s",len(m1f),len(m2f))
        raise Exception        
        
def MI(m1,m2,q=32):
    try:
        m1f=m1.flatten()*q
        m1f=m1f.astype(int)
        m2f=m2.flatten()*q
        m2f=m2f.astype(int)

        mi=mutual_info_score(m1f,m2f)
        return mi
    except:
        print("failed at corrims",len(m1f),len(m2f))
        raise Exception

def demom(m1,m2):
    try:
        #val=(m1+m2)/2 #old. this was a mistake, or at least just a differente feature
        val=m1-m2
        return val
    except:
        raise Exception
        
def demean(m1,m2):
    try:
        #val=(m1+m2)/2 #old. this was a mistake, or at least just a differente feature
        val=m1-m2
        return val
    except:
        raise Exception  
        
def geomean(m1,m2):
    try:
        val=np.sqrt(np.abs(m1)*np.abs(m2))
        return val
    except:
        raise Exception  

def demedian(m1,m2):
    try:
        v1=np.median(m1.flatten())
        v2=np.median(m2.flatten())
        val=v1-v2
        return val
    except:
        print (Exception)
        raise Exception
        
def absdemedian(m1,m2):
    try:
        v1=np.median(m1.flatten())
        v2=np.median(m2.flatten())
        val=np.abs((v1-v2))
        return val
    except:
        print (Exception)
        raise Exception

def MINorm(m1,m2,q=32):
    try:
        m1f=m1.flatten()*q
        m1f=m1f.astype(int)
        m2f=m2.flatten()*q
        m2f=m2f.astype(int)
        minorm=normalized_mutual_info_score(m1f,m2f)
        return minorm
    except:
        print("failed at corrims",len(m1f),len(m2f))
        raise Exception
 
def rasterizeMask(x,y,xmin,xmax,ymin,ymax,padding): 
    height =xmax-xmin
    width=ymax-ymin
    
    npps=np.vstack((x,y))
    npps[0,:]-=xmin
    npps[1,:]-=ymin
    
    if npps.shape[1] != 2:
        npps=npps.T
    
    poly_path=Path(npps)
    
    x, y = np.mgrid[:height+2*padding, :width+2*padding]
    x-=padding;y-=padding
    
    coors=np.hstack((x.reshape(-1, 1), y.reshape(-1,1)))    
    mask = poly_path.contains_points(coors)    
    mask = mask.reshape(height+2*padding,width+2*padding)
    #mask=mask.astype("uint8")*255
    
    return mask

def stripStr(mystr):
    mystr=mystr.replace("[","")
    mystr=mystr.replace("]","")
    mystr=mystr.replace(","," ")
    mystr=mystr.replace("{","")
    mystr=mystr.replace("}","")
    return mystr

def processObject(jo):
    tor={}
    if jo["id"] != "PathCellObject":
        return None    
    if jo["geometry"]["type"] != "Polygon":
        print("w: geometry is not polygon")     
    tor["geometry"]=stripStr(str(jo["geometry"]["coordinates"]))    
    tor["nucleusGeometry"]=stripStr(str(jo["nucleusGeometry"]["coordinates"]))    
    if "classification" in jo["properties"]:
        tor["Class"]=jo["properties"]["classification"]["name"]
    else:
        tor["Class"]=None 
    for m in jo["properties"]["measurements"]:
        tor[m["name"]]=m["value"]        
    return tor

def processName(name):
    info=name.split("/")[-1]
    info=info.replace("LGG","")
    info=info.replace("TMA","")
    info=info.replace(" ","")
    pieces=info.split("_")
    #LGG TMA 7_2_Core[1,7,B]_[8268,45852]_component_data.json"
    tma=pieces[0]
    core=pieces[2]
    core=core.replace("Core","")
    core=core.replace("[","")
    core=core.replace("]","")
    corep=core.split(",")
    core=(corep[1],corep[2])
    return tma, core
    
class CoreProcessor:

    def __init__(self):      
        self.location="/home/leslie/Documents/Uppsala/TissueMapsAll/5papers/data/ExportCellsToJson/"
        self.jsonFile="" #"LGG TMA 7_2_Core[1,7,B]_[8268,45852]_component_data.json"
        self.saveat="/home/leslie/Documents/Uppsala/TissueMapsAll/5papers/smallcells/"
        self.imagelocation="/media/leslie/Elements/LinaData/LGG TMA 8 plex image files/"
        self.imgFile="" # "LGG TMA 7_2_Core[1,7,B]_[8268,45852]_component_data.tif"
        self.markers=["DAPI" ,"IBA1" ,"Ki67" ,"TMEM119" ,"NeuroC" ,"MBP" ,"mutIDH1" ,"CD34" ,"GFAP" ,"Autofluorescence"]
        self.markersSkip=[True ,False ,False , False ,  False ,   False ,   False ,   False,  False ,  True]
        self.bigimage=None
        #self.resolution=0.4961
        self.multiprocessing=True
        self.debug=False
        self.padding=30
        self.measureNeedsImages=True
        self.loadBigImage=True
    def processCell(self,obj):
        #get geometry and get the piece from the original 
        #image and get the ask and the center and possibly
        #all the coorelations
        mask=None
        image=None
        shape=None
        maskfile=self.saveat+obj["Image"]+os.sep+"mask/"+str(obj["global_id"])+".png"
        cellfile=self.saveat+obj["Image"]+os.sep+"cell/"+str(obj["global_id"])+".npy"
        
        points=stripStr(obj["geometry"])
        points=np.fromstring(points,sep=" ")
        #apparently qupath exported the poygons in terms of pixels already....... que mamera
        #points/=self.resolution
        points=points.astype(int)
        
        x=points[::2]
        y=points[1::2]
        xmin = np.min(x); xmax = np.max(x)
        ymin = np.min(y); ymax = np.max(y)
        
        cx=np.mean(x);cy=np.mean(y)

        obj["cx"]=cx
        obj["cy"]=cy
        
        if self.measureNeedsImages:
            if os.path.isfile(cellfile):
                image=np.load(cellfile)
                shape=image.shape
            else:
                #if obj["global_id"]==45:
                #    print("stop and pay attention")
                image=self.bigimage[ymin-self.padding:ymax+self.padding,xmin-self.padding:xmax+self.padding,:]
                np.save(cellfile,image)
                shape=image.shape    
            
            if os.path.isfile(maskfile):
                mask=io.imread(maskfile)
                mask=mask.astype(bool)
            else:
                #create mask
                mask=rasterizeMask(y,x,ymin,ymax,xmin,xmax,self.padding)
                mask=mask.astype(bool)
                mask2=mask.astype("uint8")*255
                io.imsave(maskfile,mask2)
            
            if self.debug:
                inxmin=xmin-self.padding;inxmax=xmax+self.padding
                inymin=ymin-self.padding;inymax=ymax+self.padding
                #plt.imshow(self.bigimage[inymin:inymax,inxmin:inxmax,-1],cmap="viridis")
                fig,axs=plt.subplots(2,5)
                count=0
                for ax in axs.flat:
                    ax.set_title(self.markers[count])
                    ax.imshow(self.bigimage[inymin:inymax,inxmin:inxmax,count],cmap="viridis")
                    count+=1
                    ax.plot(x-inxmin,y-inymin)

                #plt.imshow(mask.astype(int)*255,alpha=0.1)
                #plt.plot(x-inxmin,y-inymin)
                plt.show()
            
        #now I have a mask and an image, start counting 
        num=len(self.markers)
        n=(num * ((num+1)//2) )
        
        for i in range(num):
            if self.markersSkip[i]:
                continue
            for j in range(i+1,num):
                if self.markersSkip[j]:
                    continue
                try:
                    if self.measureNeedsImages:
                        #for linear correlation
                        #obj[str(i)+str(j)],obj[str(i)+str(j)+"p"]=corrInMs(image[...,i],image[...,j])
                        #for mi and mi norm
                        #obj[str(i)+str(j)+"mi"]=MI(image[...,i],image[...,j])
                        #obj[str(i)+str(j)+"mino"]=MINorm(image[...,i],image[...,j])
                        #for demedian
                        #obj[str(i)+str(j)+"demedian"]=absdemedian(image[...,i],image[...,j])
                        #for difference of means                        
                        #obj[str(i)+str(j)+"demedian"]=demedian(image[...,i],image[...,j])                        
                        #difference of 90th perentile
                        obj[str(i)+str(j)+"d90s"]=d90s(image[...,i],image[...,j])
                        
                        
                    else:
                        #obj[str(i)+str(j)+"demom"]=obj["Cell: "+self.markers[i]+" mean"]-obj["Cell: "+self.markers[j]+" mean"]
                        obj[str(i)+str(j)+"geomean"]=geomean(obj["Cell: "+self.markers[i]+" mean"],obj["Cell: "+self.markers[j]+" mean"])

                except:
                    print("problem in obj "+str(obj["global_id"]))
                    #print(obj)
                    return None
        
        return obj
    
    def processCore(self,nameprocessor=None):
        data=None
        arrtodf=[]

        if nameprocessor is None:
            nameprocessor=processName
        
        tma,core=nameprocessor(self.jsonFile)
        
        with open(self.location+self.jsonFile) as json_file:
            data = json.load(json_file)

        for i in data:
            obj = processObject(i)
            if obj is not None:
                arrtodf.append(obj)    

        df=pd.DataFrame(arrtodf)
        df["cx"]=0;df["cy"]=0
        df["Image"]=str(tma)+"_"+str(core[0])+"_"+str(core[1])
        #This is asuming that both csv and json come from QuPath so the cells will be in the same order
        #and given the same indices, otherwise they have to be paired on imperfect floating coordinate matching
        df["global_id"]=df.index.values
        
        num=len(self.markers)
        #n=(num * ((num+1)//2) )
        # add n columns to df  

        maskpath=self.saveat+os.sep+str(tma)+"_"+str(core[0])+"_"+str(core[1])+os.sep+"mask/"
        cellpath=self.saveat+os.sep+str(tma)+"_"+str(core[0])+"_"+str(core[1])+os.sep+"cell/"
        if not os.path.exists(maskpath):
            os.makedirs(maskpath)

        if not os.path.exists(cellpath):
            os.makedirs(cellpath)

        for i in range(num):
            if self.markersSkip[i]:
                continue
            for j in range(i+1,num):
                if self.markersSkip[j]:
                    continue

                #for linear correlation
                #df[str(i)+str(j)]=0
                #df[str(i)+str(j)+"p"]=0 #to store the p-value just to have it

                #for mi and minorm
                #df[str(i)+str(j)+"mi"]=0
                #df[str(i)+str(j)+"mino"]=0 #to store the p-value just to have it
                
                #for differnece of 90th perc
                df[str(i)+str(j)+"d90s"]=0

                
        #now go per cell!
        #make an array of objects in the df
        allobjsincore=[]
        for i, row in df.iterrows():
            allobjsincore.append(row.to_dict())
        
        if self.loadBigImage:
            self.bigimage=io.imread(self.imagelocation+self.imgFile)
            #axis is C,Y,X should be Y,X,C
            self.bigimage=np.moveaxis(self.bigimage,0,-1) 

        if self.multiprocessing:
            with Pool(12) as p:
                self.processedcells=p.map(self.processCell,allobjsincore)
                newdf=pd.DataFrame(self.processedcells)
                newdf.to_csv(self.saveat+str(tma)+"_"+str(core[0])+"_"+str(core[1])+".csv")

        else:
            self.processedcells=[]
            for obj in allobjsincore:
                self.processedcells.append(self.processCell(obj))

            newdf=pd.DataFrame(self.processedcells)
            newdf.to_csv(self.saveat+str(tma)+"_"+str(core[0])+"_"+str(core[1])+".csv")
                
    def getAllCellsMaskInOne(self,coreid,df,imagefile,corecolumn="Image",cellcol="global_id"):
        
        #Since I needed to join the JSON polygons with the CSV then for this particular function 
        #I will actually input all the csv, both with threhsold classes and json polygons to create an image
        #full of labels, perhaps to use annotater to check this out. It will not go per cell but per core
        
        #But this function will not work if the json doesnt have a global cell id (global within the core)
        
        #here padding will always be 0, sae jsut in case
        oldpadding=self.padding
        self.padding=0
        
        #So, get an id and a df and loop
        indf=df[df[corecolumn]==coreid]
        
        inbigimage=io.imread(self.imagelocation+imagefile)
        #axis is C,Y,X should be Y,X,C
        inbigimage=np.moveaxis(inbigimage,0,-1)
        #create mask image
        maskbigimage=np.zeros((inbigimage.shape[0],inbigimage.shape[1]),dtype=np.uint16)
        
        #load the image to know the image size and shape and all
        
        #per cell in this core
        for i, row in indf.iterrows():
            
            points=stripStr(row["geometry"])
            points=np.fromstring(points,sep=" ")
            #apparently qupath exported the poygons in terms of pixels already....... que mamera
            #points/=self.resolution
            points=points.astype(int)

            x=points[::2]
            y=points[1::2]
            xmin = np.min(x); xmax = np.max(x)
            ymin = np.min(y); ymax = np.max(y)

            cx=np.mean(x);cy=np.mean(y)

            row["cx"]=cx
            row["cy"]=cy
            
            mask=rasterizeMask(y,x,ymin,ymax,xmin,xmax,self.padding)
            mask=mask.astype(bool)
            mask=mask.astype(np.uint16)*row[cellcol]
            
            maskbigimage[ymin:ymax,xmin:xmax]=np.fmax(maskbigimage[ymin:ymax,xmin:xmax],mask)
            
        
        io.imsave(self.saveat+coreid+"WHOLEMASK.png",maskbigimage)      
        print("whole mask saved for "+coreid)
            
        #reset padding
        self.padding=oldpadding
        

          
if __name__ == '__main__':

    #REMEMBER All settings:
    # location  ="/home/leslie/Documents/Uppsala/TissueMapsAll/5papers/data/ExportCellsToJson/"
    # jsonFile  ="" #"LGG TMA 7_2_Core[1,7,B]_[8268,45852]_component_data.json"
    # saveat  ="/home/leslie/Documents/Uppsala/TissueMapsAll/5papers/smallcells/"
    # imagelocation  ="/media/leslie/Elements/LinaData/LGG TMA 8 plex image files/"
    # imgFile  ="" # "LGG TMA 7_2_Core[1,7,B]_[8268,45852]_component_data.tif"
    # markers  =["DAPI" ,"IBA1" ,"Ki67" ,"TMEM119" ,"NeuroC" ,"MBP" ,"mutIDH1" ,"CD34" ,"GFAP" ,"Autofluorescence"]
    # markersSkip  =[True ,False ,False , False ,  False ,   False ,   False ,   False,  False ,  True]
    # bigimage  =None
    # resolution  =0.4961
    # multiprocessing  =True
    # debug  =False
    # padding  =30
    # measureNeedsImages  =True
    coreprocessor=CoreProcessor()
#    coreprocessor.measureNeedsImages=True
#    allimages=["LGG TMA 5_2_Core[1,10,B]_[6246,50326]_component_data.tif", "LGG TMA 5_2_Core[1,10,F]_[13866,50113]_component_data.tif", "LGG TMA 5_2_Core[1,10,I]_[19581,50232]_component_data.tif", "LGG TMA 5_2_Core[1,11,I]_[19462,51978]_component_data.tif", "LGG TMA 5_2_Core[1,12,E]_[11961,53764]_component_data.tif", "LGG TMA 5_2_Core[1,12,H]_[17557,53775]_component_data.tif", "LGG TMA 5_2_Core[1,1,A]_[3229,34793]_component_data.tif", "LGG TMA 5_2_Core[1,3,C]_[7595,37968]_component_data.tif", "LGG TMA 5_2_Core[1,4,D]_[9579,39596]_component_data.tif", "LGG TMA 5_2_Core[1,9,D]_[9976,48446]_component_data.tif", "LGG TMA 6_2_Core[1,3,B]_[7621,37931]_component_data.tif", "LGG TMA 7_2_Core[1,1,E]_[13706,34858]_component_data.tif", "LGG TMA 7_2_Core[1,3,A]_[6284,38907]_component_data.tif", "LGG TMA 7_2_Core[1,5,D]_[11999,42240]_component_data.tif", "LGG TMA 7_2_Core[1,7,B]_[8268,45852]_component_data.tif", "LGG TMA 8_2_Core[1,12,B]_[11642,52307]_component_data.tif", "LGG TMA 8_2_Core[1,1,B]_[12674,32502]_component_data.tif", "LGG TMA 8_2_Core[1,1,C]_[14579,32582]_component_data.tif", "LGG TMA 8_2_Core[1,2,A]_[10689,34249]_component_data.tif", "LGG TMA 8_2_Core[1,3,B]_[12515,36154]_component_data.tif", "LGG TMA 8_2_Core[1,5,B]_[12118,39646]_component_data.tif"]
#    alljsons=["LGG TMA 5_2_Core[1,10,B]_[6246,50326]_component_data.json", "LGG TMA 5_2_Core[1,10,F]_[13866,50113]_component_data.json", "LGG TMA 5_2_Core[1,10,I]_[19581,50232]_component_data.json", "LGG TMA 5_2_Core[1,11,I]_[19462,51978]_component_data.json", "LGG TMA 5_2_Core[1,12,E]_[11961,53764]_component_data.json", "LGG TMA 5_2_Core[1,12,H]_[17557,53775]_component_data.json", "LGG TMA 5_2_Core[1,1,A]_[3229,34793]_component_data.json", "LGG TMA 5_2_Core[1,3,C]_[7595,37968]_component_data.json", "LGG TMA 5_2_Core[1,4,D]_[9579,39596]_component_data.json", "LGG TMA 5_2_Core[1,9,D]_[9976,48446]_component_data.json", "LGG TMA 6_2_Core[1,3,B]_[7621,37931]_component_data.json", "LGG TMA 7_2_Core[1,1,E]_[13706,34858]_component_data.json", "LGG TMA 7_2_Core[1,3,A]_[6284,38907]_component_data.json", "LGG TMA 7_2_Core[1,5,D]_[11999,42240]_component_data.json", "LGG TMA 7_2_Core[1,7,B]_[8268,45852]_component_data.json", "LGG TMA 8_2_Core[1,12,B]_[11642,52307]_component_data.json", "LGG TMA 8_2_Core[1,1,B]_[12674,32502]_component_data.json", "LGG TMA 8_2_Core[1,1,C]_[14579,32582]_component_data.json", "LGG TMA 8_2_Core[1,2,A]_[10689,34249]_component_data.json", "LGG TMA 8_2_Core[1,3,B]_[12515,36154]_component_data.json", "LGG TMA 8_2_Core[1,5,B]_[12118,39646]_component_data.json"]
#   
#    coreprocessor.multiprocessing=True
#    
#    for imgf,jsf in zip(allimages,alljsons):
#        now = datetime.now() 
#        print("now = ", now) 
#        print(imgf)
#        coreprocessor.jsonFile=jsf
#        coreprocessor.imgFile=imgf
#        coreprocessor.processCore()
#
##     coreprocessor=CoreProcessor()
##     coreprocessor.multiprocessing=False
##     #coreprocessor.resolution=1.0
##     coreprocessor.debug=True
##     coreprocessor.jsonFile="LGG TMA 7_2_Core[1,7,B]_[8268,45852]_component_data.json"
##     coreprocessor.imgFile="LGG TMA 7_2_Core[1,7,B]_[8268,45852]_component_data.tif"
##     coreprocessor.processCore()

    coreprocessor.multiprocessing=False
    #if images already exist, dont waste time loading the big core image
    coreprocessor.loadBigImage=True
    coreprocessor.location="/home/leslie/Documents/Uppsala/TissueMapsAll/5papers/data/new20cores/JSON/"
    coreprocessor.imagelocation="/media/leslie/Elements/LinaData/new20cores/"
    coreprocessor.saveat="/home/leslie/Documents/Uppsala/TissueMapsAll/5papers/smallcellsNew20/"
    allimages=["LGG TMA 6_2_Core[1,3,C]_[9486,38010]_component_data.tif","LGG TMA 6_2_Core[1,3,D]_[11391,38050]_component_data.tif","LGG TMA 6_2_Core[1,3,F]_[15042,38050]_component_data.tif","LGG TMA 6_2_Core[1,7,D]_[11153,44876]_component_data.tif","LGG TMA 6_2_Core[1,9,B]_[7224,48091]_component_data.tif","LGG TMA 6_2_Core[1,9,D]_[11074,48329]_component_data.tif","LGG TMA 6_2_Core[1,9,F]_[14884,48567]_component_data.tif","LGG TMA 6_2_Core[1,9,H]_[18694,48567]_component_data.tif","LGG TMA 6_2_Core[1,9,I]_[20599,48607]_component_data.tif","LGG TMA 7_2_Core[1,10,B]_[8467,50972]_component_data.tif","LGG TMA 7_2_Core[1,12,C]_[10530,54623]_component_data.tif","LGG TMA 7_2_Core[1,3,F]_[15762,38475]_component_data.tif","LGG TMA 7_2_Core[1,3,H]_[19540,38629]_component_data.tif","LGG TMA 7_2_Core[1,4,B]_[8268,40415]_component_data.tif","LGG TMA 7_2_Core[1,5,G]_[17833,42161]_component_data.tif","LGG TMA 7_2_Core[1,6,H]_[19698,43828]_component_data.tif","LGG TMA 7_2_Core[1,7,C]_[10213,45693]_component_data.tif","LGG TMA 7_2_Core[1,7,F]_[15888,45614]_component_data.tif","LGG TMA 7_2_Core[1,8,E]_[14070,47461]_component_data.tif","LGG TMA 7_2_Core[1,8,I]_[21643,47202]_component_data.tif"]
    alljsons=["LGG TMA 6_2_Core[1,3,C]_[9486,38010]_component_data.json", "LGG TMA 6_2_Core[1,3,D]_[11391,38050]_component_data.json", "LGG TMA 6_2_Core[1,3,F]_[15042,38050]_component_data.json", "LGG TMA 6_2_Core[1,7,D]_[11153,44876]_component_data.json", "LGG TMA 6_2_Core[1,9,B]_[7224,48091]_component_data.json", "LGG TMA 6_2_Core[1,9,D]_[11074,48329]_component_data.json", "LGG TMA 6_2_Core[1,9,F]_[14884,48567]_component_data.json", "LGG TMA 6_2_Core[1,9,H]_[18694,48567]_component_data.json", "LGG TMA 6_2_Core[1,9,I]_[20599,48607]_component_data.json", "LGG TMA 7_2_Core[1,10,B]_[8467,50972]_component_data.json", "LGG TMA 7_2_Core[1,12,C]_[10530,54623]_component_data.json", "LGG TMA 7_2_Core[1,3,F]_[15762,38475]_component_data.json", "LGG TMA 7_2_Core[1,3,H]_[19540,38629]_component_data.json", "LGG TMA 7_2_Core[1,4,B]_[8268,40415]_component_data.json", "LGG TMA 7_2_Core[1,5,G]_[17833,42161]_component_data.json", "LGG TMA 7_2_Core[1,6,H]_[19698,43828]_component_data.json", "LGG TMA 7_2_Core[1,7,C]_[10213,45693]_component_data.json", "LGG TMA 7_2_Core[1,7,F]_[15888,45614]_component_data.json", "LGG TMA 7_2_Core[1,8,E]_[14070,47461]_component_data.json", "LGG TMA 7_2_Core[1,8,I]_[21643,47202]_component_data.json"]  

    now = datetime.now() 
    print("now = ", now) 
    print(allimages[0])
    coreprocessor.jsonFile=alljsons[0]
    coreprocessor.imgFile=allimages[0]
    coreprocessor.processCore()
