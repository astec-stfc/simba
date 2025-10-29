import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui


## Create a subclass of GraphicsObject.
## The only required methods are paint() and boundingRect()
## (see QGraphicsItem documentation)
class rectangleElement(pg.GraphicsObject):

    def __init__(self, start, end, color="g", width=1, scale=1):
        pg.GraphicsObject.__init__(self)
        self.color = color
        self.start = start
        self.end = end
        self.width = width
        self.scale = scale
        self.yoffset = 0
        self.generatePicture()

    def generatePicture(self):
        ## pre-computing a QPicture object allows paint() to run much more quickly,
        ## rather than re-drawing the shapes every time.
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        color = pg.mkColor(self.color)
        color.setAlphaF(0.5)
        # p.setPen(pg.mkPen(color, width=0.001))
        # p.setBrush(pg.mkBrush(color))
        p.fillRect(
            QtCore.QRectF(
                self.start,
                -0.5 * abs(self.width * self.scale),
                self.end - self.start,
                abs(self.width * self.scale),
            ),
            color,
        )
        p.end()

    def paint(self, p, *args):
        p.drawPicture(QtCore.QPointF(0, self.yoffset), self.picture)

    def boundingRect(self):
        ## boundingRect _must_ indicate the entire area that will be drawn on
        ## or else we will get artifacts and possibly crashing.
        ## (in this case, QPicture does all the work of computing the bouning rect for us)
        return QtCore.QRectF(self.picture.boundingRect())

    def setScale(self, scale):
        self.scale = scale
        self.generatePicture()

    def setOffset(self, offset):
        self.yoffset = offset
        self.generatePicture()


class lineElement(pg.GraphicsObject):

    def __init__(self, start, end, color="g", width=1, scale=1):
        pg.GraphicsObject.__init__(self)
        self.color = color
        self.start = start
        self.end = end
        self.width = width
        self.scale = scale
        self.yoffset = 0
        self.generatePicture()

    def generatePicture(self):
        ## pre-computing a QPicture object allows paint() to run much more quickly,
        ## rather than re-drawing the shapes every time.
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        color = pg.mkColor(self.color)
        # color.setAlphaF(1)
        p.setPen(pg.mkPen(color))
        p.setBrush(pg.mkBrush(color))
        p.drawRect(
            QtCore.QRectF(
                self.start,
                -0.5 * abs(self.width * self.scale),
                self.end - self.start,
                abs(self.width * self.scale),
            )
        )
        p.end()

    def paint(self, p, *args):
        p.drawPicture(QtCore.QPointF(0, self.yoffset), self.picture)

    def boundingRect(self):
        ## boundingRect _must_ indicate the entire area that will be drawn on
        ## or else we will get artifacts and possibly crashing.
        ## (in this case, QPicture does all the work of computing the bouning rect for us)
        return QtCore.QRectF(self.picture.boundingRect())

    def setScale(self, scale):
        self.scale = scale
        self.generatePicture()

    def setOffset(self, offset):
        self.yoffset = offset
        self.generatePicture()


class latticeDraw:

    colorwidths = {
        "dipole": ("b", 0.5),
        "quadrupole": ("r", 0.6),
        "cavity": ("g", 0.35),
        "solenoid": ("c", 0.4),
        "aperture": ("k", 0.3),
        "kicker": ("m", 0.25),
        "wall_current_monitor": ((100, 100, 200), 0.2),
        "beam_position_monitor": ((100, 100, 100), 0.2),
        "screen": ((150, 150, 100), 0.3),
        "collimator": ("k", 0.2),
        "line": ("k", 0.05),
    }

    def __init__(self, element_positions):
        self.element_positions = element_positions
        self.elements = []
        self._scale = 1
        self.defineElements()

    def defineElements(self):
        self.elements = []
        end0 = 0
        for name, elem in self.element_positions.items():
            type = elem["type"]
            (start, end) = elem["position"]
            if type in self.colorwidths:
                self.elements.append(
                    rectangleElement(
                        min([start, end]), max([start, end]), *self.colorwidths[type]
                    )
                )
                if end0 < start:
                    self.elements.append(
                        lineElement(end0, start, *self.colorwidths["line"])
                    )
                if end > end0:
                    end0 = end

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, newscale):
        self._scale = newscale
        for elem in self.elements:
            elem.setScale(self._scale)

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, newoffset):
        self._offset = newoffset
        for elem in self.elements:
            elem.setOffset(self._offset)


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == "__main__":

    import sys, os

    sys.path.append(
        os.path.abspath(
            __file__ + "/../../../../ASTRA_COMPARISONRunner-HMCC/OnlineModel/"
        )
    )
    from data import data

    dataObject = data.Data(
        settings_directory=os.path.abspath(
            __file__ + "/../../../../ASTRA_COMPARISONRunner-HMCC/OnlineModel/"
        )
    )
    elemdata = dataObject.latticeDict
    elements = latticeDraw(elemdata).elements
    pg.setConfigOptions(antialias=True)
    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")
    plt = pg.plot()
    for e in elements:
        plt.addItem(e)
    plt.setWindowTitle("pyqtgraph example: customGraphicsItem")

    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        QtGui.QApplication.instance().exec_()
