import qupath.lib.objects.PathObjects
import qupath.lib.roi.ROIs
import qupath.lib.regions.ImagePlane
import qupath.lib.io.GsonTools
import qupath.lib.geom.Point2
import qupath.lib.objects.classes.PathClass
import qupath.lib.objects.classes.PathClassFactory

def plane = ImagePlane.getPlane(0, 0)

def loc="/home/leslie/Documents/Uppsala/TissueMapsAll/5papers/data/trimmed/d90s/toQupath/"

def gson=GsonTools.getInstance(true)
BufferedReader bufferedReader = new BufferedReader(new FileReader(loc+"5_10_B_xg.json"));
HashMap<String, String> myjson = gson.fromJson(bufferedReader, HashMap.class); 

def classmap=["Astrocyte":[41, 83, 185] ,"Glioma":[255, 127, 14] ,"Neuron":[234, 134, 213] ,
"Microglia":[234, 35, 37] ,"Macrophage":[148, 103, 189] ,"Endothelial":[38, 153, 177]]

def tableclassmap=["1":[41,83,185], "129":[85,116,190], "2":[255,136,31], "3":[210,169,59], "6":[171,141,64], "7":[146,88,36], 
"130":[195,145,45], "131":[171,103,42], "134":[182,123,45], "135":[111,57,9], "4":[234,134,213], "8":[234,35,37], "24":[187,75,75], 
"136":[187,75,75], "152":[173,102,102], "16":[148,103,189], "144":[159,137,178], "32":[38,153,177], "160":[120,193,208], 
"128":[53,193,53], "999":[120,120,120]] 

usexgclass=false

pathClasses=[]
tablePathClasses=[]

for(c in classmap){
    v=c.value;k=c.key;
    pathClasses.add(PathClassFactory.getPathClass(k, getColorRGB(v[0],v[1],v[2])));
}

for(c in tableclassmap){
    v=c.value;k=c.key;
    tablePathClasses.add(PathClassFactory.getPathClass(k, getColorRGB(v[0],v[1],v[2])));
}

xCoords = myjson["allx"]
yCoords = myjson["ally"]


classcol="allc"
if(! usexgclass){
    
    classcol="alllt"
}
classes_index = myjson[classcol]


//this should be empty , I just call it to get the right data type
annotations = getAnnotationObjects()

for (c=0; c < xCoords.size(); c++) {

    List<Point2> points = []
    
    def xarr= xCoords[c] as double[]
    def yarr= yCoords[c] as double[]
    
    for( i=0; i< xarr.size();i++){
        points.add(new Point2(xarr[i], yarr[i]));        
    }

    def cell_roi = ROIs.createPolygonROI(points, plane)
    def cell = PathObjects.createAnnotationObject(cell_roi)
    annotations.add(cell)
    if(usexgclass){
        cell.setPathClass(pathClasses[ classes_index[c] ])
    }else{
        cell.setPathClass(tablePathClasses[ classes_index[c] ])
    }
    //
    
}

def viewer = getCurrentViewer()
def imageData = viewer.getImageData()
def hierarchy = imageData.getHierarchy()
hierarchy.getRootObject().addPathObjects(annotations);
hierarchy.fireHierarchyChangedEvent(this)

print "Done"
