import cv2 as cv
import numpy as np

FONT=cv.FONT_HERSHEY_SIMPLEX
COLORS ={
    "cyan":(0,255, 255),
    "white":(255,255,255),
    "red":(0,0,255),
}
class DepthVisualizer:
    def __init__(self, scale=0.5,num_layers=10,min_depth_mm=100,max_depth_mm=2500):
        self.scale=scale
        self.num_layers=num_layers
        self.min_depth=min_depth_mm
        self.max_depth=max_depth_mm
        self._layer_step=256//num_layers

    
    def render(self, color_img,depth_img,objects=None):
        
        if objects is None:
            objects =[]

        h =int(depth_img.shape[0]* self.scale)
        w = int(depth_img.shape[1]*self.scale)

# scaling down the imgs...

        color_small= cv.resize(color_img,(w,h))
        depth_small =cv.resize(depth_img,(w,h))
    # normalise the depth to 255
        clipped = np.clip(depth_small, self.min_depth, self.max_depth)
        normalized= cv.normalize(clipped, None,0,255,cv.NORM_MINMAX, dtype=cv.CV_8U)

        normalized = 255 - normalized
        normalized[depth_small ==0] =0  # Invalid depth stays black

        #convert to layers
        layered = (normalized//self._layer_step) *self._layer_step

        #applying turbo colormap for the heatmap look
        colormap =cv.applyColorMap(layered,cv.COLORMAP_TURBO)
        colormap[depth_small ==0] = [0,0,0]  # black out invalid pixels... 

        edges = cv.Canny(layered,50,150)
        colormap[edges>0] = [0,255,0]

        for obj in objects:
            sx1,sy1, sx2,sy2 =[int(v *self.scale) for v in obj["box"]]
            scx=int(obj["cx"]*self.scale)
            scy=int(obj["cy"]*self.scale)

            depth_mm = depth_img[obj["cy"], obj["cx"]]

            cv.rectangle(colormap, (sx1, sy1), (sx2, sy2), (255, 255, 255), 1)
            cv.circle(colormap, (scx, scy), 3, (0, 0, 255), -1)
            cv.putText(
                colormap, f"{obj['color']} ({depth_mm}mm)",
                (sx1, sy1 - 5), FONT, 0.4, (255, 255, 255), 1
            )

        # to setup title 
        cv.putText(
            colormap, "TOPOGRAPHIC DEPTH LAYERS",
            (10, 25), FONT, 0.6, COLORS["cyan"], 2
        )

        # stacking side by side...
        return np.hstack((color_small, colormap))


