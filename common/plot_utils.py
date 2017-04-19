from pylab import *

def multidraw(shape, images, dfs = None, color_code = None, blackwhite = False):
    figure()
    for i in range(len(images)):
        if i+1 > shape[0]*shape[1]:
            break
        subplot(shape[0], shape[1], 1+i)
        if dfs is not None:
            draw(images[i], dfs[i], color_code, blackwhite, multiplot = True)
        else:
            draw(images[i], None, color_code, blackwhite, multiplot = True)

# Plots image and dots in the given coordinates
## color_code: dictionary including keys class and values a color string
## df: a pandas dataframe with columns x, y and class
def draw(image, df = None, color_code = None, blackwhite = False, multiplot = False):
    if not multiplot:
        figure()
    if blackwhite:
        imshow(image, cmap = cm.Greys)
    else:
        imshow(image)
    if df is not None:
        for classid in df['class'].unique():
            x = df[df['class'] == classid]['x']
            y = df[df['class'] == classid]['y']
            if color_code:
                plot(y, x, 'o' + color_code[classid])
            else:
                plot(y, x, 'o')



