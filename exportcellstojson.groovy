import static qupath.lib.gui.scripting.QPEx.*

def cells = getCellObjects()

boolean prettyPrint = true
def gson = GsonTools.getInstance(prettyPrint)

def imageData = getCurrentImageData()

def server = imageData.getServer()

def name = server.getMetadata().getName();

def end = name.indexOf(".tif")

def jsonname=name[0..end-1]+".json"


File file = new File('C:/Users/linwik/Desktop/'+jsonname)
file.withWriter('UTF-8') {
     gson.toJson(cells,it)
}

 
 