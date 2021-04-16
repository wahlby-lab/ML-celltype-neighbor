import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
import seaborn as sns

def plotUMAP(indf,colx,coly,colz,color_col=None,s=5,marker=".",frac=1,linewidths=0,xlim=[-4.0, 15.0],ylim=[-4.0, 15.0],edgecolors=None):
    fig=plt.figure(figsize=(9,3),dpi=500)
    cycled = [colx,coly,colz,colx]
    adf=indf.sample(frac=frac)
    indices= adf["global_id"].values
    colors=[]
    if color_col is None:
        colors=Y_umap[indices]
    else:
        colors=adf[color_col]
    for i in range(3):
        ax=plt.subplot(1,3,i+1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        plt.scatter(adf[cycled[i]], adf[cycled[i+1]], c=colors,  
                    s=s, marker=marker, linewidths=linewidths, edgecolors=edgecolors)
        plt.xlabel(str(cycled[i]))
        plt.ylabel(str(cycled[i+1]))

    plt.tight_layout()
    return fig

def printColsAsString(df,wrap=None,sep=" "):
    string=""
    for c in df.columns:
        if wrap is not None:
            string+=sep+wrap+c+wrap
        else:
            string+=sep+c
    print(string)

def plotTwoSets(df1,df2,col1,col2,mainX,mainY,cmap1=None,cmap2=None,s1=None,s2=None,scol1=None,scol2=None,frac=1.0,xlim=None,ylim=None,grid=(5,5),figsize=(14,14),tight=False,dpi=120,save=None):
    if figsize is None:
        figsize=(6,12)
    if grid is None:
        grid=(5,4)
    fig, axs = plt.subplots(grid[0],grid[1],figsize=figsize,dpi=dpi)
    
    if tight:
        plt.tight_layout()
    
    for ax in axs.flat:
        ax.axis("off")
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.tick_params(top=False, bottom=False, left=False, right=False,labelleft=False, labelbottom=False)
            
    
    
    for ax,exp in zip(axs.flat, sorted(df1["Image"].unique()) ):        
        ax.set_title(exp)
        sample=df1[df1["Image"]==exp].sample(frac=frac)
        colors=[]
        if col1 is None:
            #colors=sample[indices].values
            colors="#0000ff"
        else:
            colors=sample[col1]
        if scol1 is None:
            if s1 is None:
                s1=2
        else:
            s1=sample[scol1]
            
        if cmap1 is None:
            ax.scatter(sample[mainX],-sample[mainY],marker=".",c=colors,s=s1)
        else:
            ax.scatter(sample[mainX],-sample[mainY], marker=".", c=colors,cmap=cmap1, s=s1)
            
    for ax,exp in zip(axs.flat, sorted(df2["Image"].unique()) ):        
        ax.set_title(exp)
        sample=df2[df2["Image"]==exp].sample(frac=frac)
        colors=[]
        if col2 is None:
            #colors=sample[indices].values
            colors="#0000ff"
        else:
            colors=sample[col2]
        if scol2 is None:
            if s2 is None:
                s2=2
        else:
            s2=sample[scol2]
            
        if cmap2 is None:
            ax.scatter(sample[mainX],-sample[mainY],marker=".",c=colors,s=s2)
        else:
            ax.scatter(sample[mainX],-sample[mainY], marker=".", c=colors,cmap=cmap2, s=s2)
            
    if save is not None:
        plt.savefig(save,dpi=dpi)
    plt.show()
            
    
    
    

def plotAll(odf,mainX,mainY,groupby="Image",rux=None,ruy=None,ruz=None,figsize=None,frac=1.0,grid=None,cmap=None, cols=["3DUMX","3DUMY","3DUMZ"],xlim=None,ylim=None,color_col=None,s=None,scol=None,dpi=120,drawUMAP=False,save=None,linewidth=None,ignoretitle=False,completeforempty=False,order=None,darkbg=False,show=False):
    
    indf=odf
    if rux != None:
        indf=indf[(indf[cols[0]]>rux[0]) & (indf[cols[0]]<rux[1])]
    if ruy != None:
        indf=indf[(indf[cols[1]]>ruy[0]) & (indf[cols[1]]<ruy[1])]
    if ruz != None:
        indf=indf[(indf[cols[2]]>ruz[0]) & (indf[cols[2]]<ruz[1])]
    
    if figsize is None:
        figsize=(6,12)
    if grid is None:
        grid=(5,4)
    fig, axs = plt.subplots(grid[0],grid[1],figsize=figsize,dpi=dpi)
    
    if len(indf)==0:
        print("Filtered all out, check boundaries")
        return None
           
    for ax in axs.flat:
        ax.axis("off")
        if xlim is not None:
            ax.set_xlim(xlim)
        #else:
        #    ax.set_xlim([0.0, 1])
            
        if ylim is not None:
            ax.set_ylim(ylim)
        #else:
        #    ax.set_ylim([0.0, 1])
            
        ax.tick_params(top=False, bottom=False, left=False, right=False,labelleft=False, labelbottom=False)
    
    listcores=None
    if order==None:
        listcores=sorted(indf[groupby].unique())
    else:
        listcores=order
        
    for ax,exp in zip(axs.flat, listcores ):   
        if not ignoretitle:
            ax.set_title(exp)
        sample=indf[indf[groupby]==exp].sample(frac=frac)
        indices= sample["global_id"].values
        colors=[]
        if color_col is None:
            #colors=sample[indices].values
            colors="#0000ff"
        else:
            colors=sample[color_col]
        if scol is None:
            if s is None:
                s=2
        else:
            s=sample[scol]
            
        if cmap is None:
            ax.scatter(sample[mainX],-sample[mainY],marker=".",c=colors,s=s,linewidth=linewidth)
        else:
            ax.scatter(sample[mainX],-sample[mainY], marker=".", c=colors,cmap=cmap, s=s,linewidth=linewidth)
        
    
    fig2=None
    
    if drawUMAP:
        fig2=plotUMAP(indf,cols[0],cols[1],cols[2],color_col=color_col,xlim=[-1,3],ylim=[-1,3])
        
    
    if save is not None:
        plt.savefig(save,dpi=dpi,transparent=True)
    plt.show()
    if fig2 is not None:
        return fig,fig2
    #else:
    #    return figimagema
    

def hexTofloatRGB(hexSTR):
    hexSTR=hexSTR.replace("#","")
    r=0;g=0;b=0
    if(len(hexSTR)==3):
        r=hexSTR[0]+hexSTR[0]
        g=hexSTR[1]+hexSTR[1]
        b=hexSTR[2]+hexSTR[2]
    
    r=float(int(hexSTR[0:2],16))/255.0
    g=float(int(hexSTR[2:4],16))/255.0
    b=float(int(hexSTR[4:6],16))/255.0
    
    return [r,g,b]

def hexToIntRGB(hexSTR):
    hexSTR=hexSTR.replace("#","")
    r=0;g=0;b=0
    if(len(hexSTR)==3):
        r=hexSTR[0]+hexSTR[0]
        g=hexSTR[1]+hexSTR[1]
        b=hexSTR[2]+hexSTR[2]
    
    r=int(hexSTR[0:2],16)
    g=int(hexSTR[2:4],16)
    b=int(hexSTR[4:6],16)
    
    return [r,g,b]
       
def floatToHex(col):
    col=np.array(col)
    col*=255.0
    col=col.astype("uint8")
    col=tuple(col)
    hexcol='#%02x%02x%02x' % col
    return hexcol
    
def floatRGBtoIntRGB(RGB):
    r=int(RGB[0]*255)
    g=int(RGB[1]*255)
    b=int(RGB[2]*255)
    return [r,g,b]

def randColor(typ="hex"):
    rand=np.random.random()
        
    cx=rand
    cy=np.clip(np.abs(rand-np.random.random()),0.0,1.0)
    cz=np.clip(np.abs(rand-np.random.random()),0.0,1.0)
    
    if typ=="float":
        col=np.array([cx,cy,cz])
        np.random.shuffle(col) 
        return col
    if typ=="hex":
        col=np.array([cx,cy,cz])
        np.random.shuffle(col) 
        col*=255.0
        col=col.astype("uint8")
        col=tuple(col)
        thecol='#%02x%02x%02x' % col        
        return thecol
       
        
def writeply(name,adf,colorcol=None
             ,umapcols=["3DUX","3DUY","3DUZ"],umapcolors=True):
    ishex=False;isfloat=False;isint=False
    if colorcol is not None:
        h=adf.head(1)
        print()
        if isinstance(h[colorcol].values[0],tuple):
            ishex=isinstance(h[colorcol].values[0][0],str)
            isfloat=isinstance(h[colorcol].values[0][0],float)
            isint=isinstance(h[colorcol].values[0][0],int)
        elif str(type(h[colorcol].values[0]))=="<class 'numpy.ndarray'>":
            isfloat=True
        else:    
            ishex=isinstance(h[colorcol].values[0],str)
            isfloat=isinstance(h[colorcol].values[0],float)
            isint=isinstance(h[colorcol].values[0],int)
        
    print(ishex,isfloat,isint)
    with open(name,"w") as writer:
        writer.write("ply\nformat ascii 1.0\ncomment author: Leslie Solorzano\ncomment object: UMAP vis\nelement vertex "+str(len(adf))+"\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for i, row in adf.iterrows():
            
            line=str(row[umapcols[0]])+" "+str(row[umapcols[1]])+" "+str(row[umapcols[2]])+" "
            if umapcolors:
                irgb=floatRGBtoIntRGB(row[umapcols].values)
                line+=str(irgb[0])+" "+str(irgb[1])+" "+str(irgb[2])
            else:
                if colorcol is None:
                    print("colorcol needs to be defined if umap color is false")
                    return 1
                
                if ishex==False and isfloat==False and isint==False:
                    print("cI dont know how to convert these colors")
                    print(row[colorcol])
                    return 1
                if ishex:
                    frgb=hexTofloatRGB(row[colorcol])
                    irgb=floatRGBtoIntRGB(frgb)
                    line+=str(irgb[0])+" "+str(irgb[1])+" "+str(irgb[2])
                elif isfloat:
                    irgb=floatRGBtoIntRGB(row[colorcol])
                    line+=str(irgb[0])+" "+str(irgb[1])+" "+str(irgb[2])
                elif isint:
                    irgb=row[colorcol].values[0]
                    line+=str(irgb[0])+" "+str(irgb[1])+" "+str(irgb[2])
            writer.write(line+"\n")   
    

def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    i = int(h*6.0) # XXX assume int() truncates!
    f = (h*6.0) - i
    p = v*(1.0 - s)
    q = v*(1.0 - s*f)
    t = v*(1.0 - s*(1.0-f))
    i = i%6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q

    
def norm_cm(cm,eps=0.000001) :
    n_cm = (cm.astype('float')) / ((cm.sum(axis=1)[:, np.newaxis])+0.000001)
    return n_cm 
    
    
def plot_confusion_matrix(cm, classes=None, norm_cm=None, cmap='viridis',save=None,pad_inches=None,vmin=0.0,vmax=1.0):
    
    plt.figure(figsize=[7, 6])       
        
    if norm_cm is None:
        norm_cm = (cm.astype('float')) / ((cm.sum(axis=1)[:, np.newaxis])+0.000001)
        
    if classes is not None:
        sns.heatmap(norm_cm, annot=cm, fmt='g', xticklabels=classes, yticklabels=classes,
                    cmap=cmap,mask=np.isnan(norm_cm),vmin=vmin, vmax=vmax)
    else:
        sns.heatmap(norm_cm, annot=cm, fmt='g',
                    cmap=cmap,mask=np.isnan(norm_cm),vmin=vmin, vmax=vmax)
    plt.tight_layout()
    
    if save is not None:
        plt.savefig(save, dpi=150,pad_inches=pad_inches)
    else:
        plt.show()

        

def myOwnCM(y,yhat,numclasses=6,labels=None,dtype=None):
    #This is to circument the problm when there is 0 of a class and somehow sklearn cm is not managing it properly! 
    cm=[]
    for i in range(numclasses):
        yhats=yhat[y==i]
        row=[0]*numclasses
        if len(yhats)==0:
            cm.append(row)
            continue
        else:
            for j in yhats:
                j=int(j)
                row[j]+=1
            cm.append(row)
    if dtype is not None:
        cm=np.array(cm,dtype=dtype)
    else:
        cm=np.array(cm)
    return cm

def paletteOppositeColors(numc,countsdf,countscolin,countscolout,ds=0.45, dl=0.75, hueoffset=310, opposite=180, shuffle=False, show=False):
    ds=ds; dl=dl; hueoffset=hueoffset; opposite=opposite
    coldict={}

    n_clusters_=numc

    temp_n_clusters=n_clusters_+1
    if n_clusters_ %2==1:
        temp_n_clusters=n_clusters_+1
    huestep=np.floor(360.0/temp_n_clusters)
    hues=np.arange(0,360,huestep)
    hues[::2]=(hues[::2]+opposite)%360
    hues=(hues+hueoffset)%360; hues/=360.0
    sats=np.ones(temp_n_clusters,dtype=float)
    sats[temp_n_clusters//2:]=ds
    vals=np.ones(temp_n_clusters,dtype=float)
    vals[2::4]=dl
    vals[3::4]=dl
    colorsrr=[]

    for i in range(temp_n_clusters):
        c=hsv_to_rgb(hues[i], sats[i], vals[i])
        c=np.array(c,dtype=float)
        colorsrr.append(c)

    colorsrr=np.array(colorsrr,dtype=float)
    
    if shuffle:
        np.random.shuffle(colorsrr)
        
    if show:                
        plt.figure (figsize=(10,1))
        plt.imshow(colorsrr.reshape((1,temp_n_clusters,3)))
        plt.title(str(hueoffset))
        plt.show()

    for i, row in countsdf.sort_values(by="count",ascending=False).iterrows():
        col=colorsrr[i]
        coldict[row[countscolin]]={"rgb":col,"hex":floatToHex(col)}

    nrgbs=[]
    nhexs=[]
    for i, row in countsdf.iterrows():
        nhexs.append(coldict[row[countscolin]]["hex"])

    countsdf[countscolout]=nhexs
    #counts["MSCOLORRGBND"]=nrgbs
    
    return coldict

def greenCMAP():
    #green cmap
    N = 70
    vals = np.ones((N, 4))
    vals[:N//2, 0] = np.linspace(1.0, 0.0, N//2); vals[N//2:, 0] = np.linspace(0.0, 84/256, N//2)
    vals[:N//2, 1] = np.linspace(1.0, 0.8, N//2); vals[N//2:, 1] = np.linspace(0.8, 121/256, N//2)
    vals[:N//2, 2] = np.linspace(1.0, 0.0, N//2); vals[N//2:, 2] = np.linspace(0.0, 46/256, N//2)

    whites=np.ones((30,4),dtype="float")
    vals=np.vstack((whites,vals))

    mygreencmp = ListedColormap(vals)
    
    return mygreencmp

def purpleCMAP():
    N = 70
    vals = np.ones((N, 4))
    vals[:N//2, 0] = np.linspace(1.0, 1.0, N//2); vals[N//2:, 0] = np.linspace(1.0, 125/255, N//2)
    vals[:N//2, 1] = np.linspace(1.0, 0.0, N//2); vals[N//2:, 1] = np.linspace(0.0, 47/256 , N//2)
    vals[:N//2, 2] = np.linspace(1.0, 1.0, N//2); vals[N//2:, 2] = np.linspace(1.0, 107/255, N//2)

    whites=np.ones((30,4),dtype="float")
    vals=np.vstack((whites,vals))

    myvioletcmp = ListedColormap(vals)

    return myvioletcmp
    