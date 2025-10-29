import sys, os, time, math, datetime, copy, re, h5py
from collections import OrderedDict
import glob

try:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
except:
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    from PyQt5.QtWidgets import *
import pyqtgraph as pg
from .latticeDraw import latticeDraw


class mainWindow(QMainWindow):
    def __init__(self):
        super(mainWindow, self).__init__()
        self.resize(1800, 900)
        self.centralWidget = QWidget()
        self.layout = QVBoxLayout()
        self.centralWidget.setLayout(self.layout)

        self.tab = QTabWidget()
        self.multiPlot = multiPlotWidget()

        self.layout.addWidget(self.multiPlot)

        self.setCentralWidget(self.centralWidget)

        self.setWindowTitle("ASTRA Data Plotter")
        menubar = self.menuBar()
        fileMenu = menubar.addMenu("&File")

        exitAction = QAction("&Exit", self)
        exitAction.setShortcut("Ctrl+Q")
        exitAction.setStatusTip("Exit application")
        exitAction.triggered.connect(self.close)
        fileMenu.addAction(exitAction)


class multiPlotWidget(QWidget):
    """QWidget containing various Twiss plots"""

    # Styles for the plot lines
    colors = [
        QColor("#F5973A"),
        QColor("#A95AA1"),
        QColor("#85C059"),
        QColor("#0F2080"),
        QColor("#BDB8AD"),
        "r",
        "k",
        "m",
        "g",
    ]
    styles = [Qt.SolidLine, Qt.DashLine, Qt.DotLine, Qt.DashDotLine, Qt.DashDotDotLine]

    # Layout oder for the individual Tiwss plot items
    plotParams = []

    highlightCurveSignal = pyqtSignal(str)
    unHighlightCurveSignal = pyqtSignal(str)

    def __init__(
        self,
        xmin=None,
        ymin=None,
        xlabel=None,
        xlabelunits=None,
        setTitles=True,
        **kwargs,
    ):
        super(multiPlotWidget, self).__init__()
        """ multiPlotWidget - main pyQtGraph display widgets """
        self.multiPlotView = pg.GraphicsView(useOpenGL=True)
        self.multiPlotWidget = pg.GraphicsLayout()
        self.multiPlotView.setCentralItem(self.multiPlotWidget)
        """ multiPlotWidgets - holds the base plotWidgets for each Twiss plot """
        self.multiPlotWidgets = {}
        """ multiPlotLattice - holds the lattice items for each Twiss plot """
        self.multiPlotLattice = {}
        """ curves - a dictionary containing {directory: [individual plotItems for each twiss plot,]} """
        self.curves = {}
        for n, param in enumerate(self.plotParams):
            if param == "next_row":
                self.multiPlotWidget.nextRow()
            else:
                """p - the relevant plotWidget for each param in plotParams"""
                if setTitles:
                    p = self.multiPlotWidget.addPlot(title=param["label"])
                else:
                    p = self.multiPlotWidget.addPlot()
                """ this just links the horizontal axis for all plots.
                    The first plot is selected to be the master axis """
                if n == 0:
                    self.linkAxis = p.vb
                else:
                    p.setXLink(self.linkAxis)
                """ Connect the scaleChanged signal to redraw lattice """
                p.showGrid(x=True, y=True)
                p.setLabel(
                    "left",
                    text=param["name"],
                    units=param["units"],
                    **{"font-size": "12pt", "font-family": "Arial"},
                )
                p.setLabel("bottom", text=xlabel, units=xlabelunits)
                if "xlabel" in param and param["xlabel"] is not None:
                    xlu = param["xlabelunits"] if "xlabelunits" in param else None
                    p.setLabel("bottom", text=param["xlabel"], units=xlu)
                """ set lower plot limit for all plots """
                if xmin is not None:
                    p.vb.setLimits(xMin=xmin)
                if ymin is not None:
                    p.vb.setLimits(yMin=ymin)
                # paramater xmin/ymin overrides global setting
                if "xmin" in param and param["xmin"] is not None:
                    p.vb.setLimits(xMin=param["xmin"])
                if "ymin" in param and param["ymin"] is not None:
                    p.vb.setLimits(yMin=param["ymin"])
                """ set the vertical viewRange according to the plotParams """
                if "range" in param:
                    p.vb.setYRange(*param["range"])
                """ record the plotWidget in the dictionary indexed by Twiss item """
                self.multiPlotWidgets[param["label"]] = p

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.layout.addWidget(self.multiPlotView)

        """ used for style cycling """
        self.plotColor = 0
        self.shadowCurves = []

    def add_lattice_plot(self, elementlist):
        self.elementlist = elementlist
        for name, p in self.multiPlotWidgets.items():
            lattice = latticeDraw(self.elementlist)
            self.multiPlotLattice[p.getViewBox()] = lattice
            for elem in lattice.elements:
                elem.setZValue(0)
                p.addItem(elem)
            self.reScaleLattice(p.getViewBox())
            p.getViewBox().sigYRangeChanged.connect(self.reScaleLattice)

    def reScaleLattice(self, p):
        lattice = self.multiPlotLattice[p]
        yScale = p.viewRange()[1]
        latticeScale = float((yScale[1] - yScale[0]) / 5.0)
        lattice.scale = latticeScale
        lattice.offset = float(yScale[1]) - latticeScale

    def set_horizontal_axis_label(self, label=None, units=None):
        for n, param in enumerate(self.plotParams):
            if not param == "next_row":
                """p - the relevant plotWidget for each param in plotParams"""
                p = self.multiPlotWidgets[param["label"]]
                if label is not None and units is not None:
                    p.setLabel("bottom", text=label, units=units)
                elif label is not None and units is not None:
                    p.setLabel("bottom", text=label)
                elif units is not None:
                    p.setLabel("bottom", units=label)

    def addCurve(self, x, y, name, label, pen):
        """adds a curve to the main plot"""
        self.curves[name][label] = self.multiPlotWidgets[label].plot(x=x, y=y, pen=pen)
        # self.curves[name][label].curve.setClickable(True)
        # self.curves[name][label].sigClicked.connect(lambda: self.curveClicked(name))
        # self.updateCurveHighlights()

    def updateCurve(self, x, y, name, label):
        """updates a curve to the main plot"""
        self.curves[name][label].setData(
            x=x, y=y, pen=self.curves[name][label].opts["pen"]
        )

    def removeCurve(self, name):
        """finds all curves based on a key name, and removes them"""
        if not isinstance(name, (list, tuple)):
            name = [name]
        for n in name:
            if n in self.shadowCurves:
                self.shadowCurves.remove(n)
            if n in self.curves:
                for param in self.plotParams:
                    if not param == "next_row":
                        """Remove the plotItem from the relevant plotWidget"""
                        # print('REMOVING curve: ', name)
                        try:
                            self.multiPlotWidgets[param["label"]].removeItem(
                                self.curves[n][param["label"]]
                            )
                        except:
                            pass
                del self.curves[n]
                self.removeData(n)
        # self.updateCurveHighlights()

    def removeData(self, name):
        pass

    def clearCurves(self):
        self.removeCurve(self.curves.keys())

    def curveClicked(self, name):
        if name in self.shadowCurves:
            self.unHighlightPlot(name)
            self.unHighlightCurveSignal.emit(name)
        else:
            self.highlightPlot(name)
            self.highlightCurveSignal.emit(name)

    def highlightPlot(self, name):
        """highlights a particular plot"""
        # print('highligher clicked! = ', name)
        if not isinstance(name, (list, tuple)):
            name = [name]
        for n in name:
            self.addShadowPen(n)
        self.updateCurveHighlights()

    def unHighlightPlot(self, name):
        """highlights a particular plot"""
        # print('highligher clicked! = ', name)
        if not isinstance(name, (list, tuple)):
            name = [name]
        for n in name:
            self.removeShadowPen(n)
        self.updateCurveHighlights()

    def updateCurveHighlights(self):
        for n in self.curves.keys():
            if n in self.shadowCurves or not len(self.shadowCurves) > 0:
                self.setPenAlpha(n, 255, 3)
            else:
                self.setPenAlpha(n, 75, 3)

    def addShadowPen(self, name):
        """add/remove a shadow pen to a given plot curve"""
        if not name in self.shadowCurves:
            self.shadowCurves.append(name)
            # for param in self.plotParams:
            #     if not param == 'next_row':
            #         label = param['label']
            #         # if name in self.curves and label in self.curves[name]:
            #         curve = self.curves[name][label]
            #         pen = curve.opts['pen']
            #         shadowpencolor = pen.color()
            #         shadowpencolor.setAlpha(100)
            #         shadowpen = pg.mkPen(color=shadowpencolor, width=(pen.width()+3))
            #         curve.setShadowPen(shadowpen)

    def removeShadowPen(self, name):
        """add/remove a shadow pen to a given plot curve"""
        if name in self.shadowCurves:
            self.shadowCurves.remove(name)
            # for param in self.plotParams:
            #     if not param == 'next_row':
            #         label = param['label']
            #         # if name in self.curves and label in self.curves[name]:
            #         curve = self.curves[name][label]
            #         curve.setShadowPen(None)
            #         curve.opts['shadowPen'] = None

    def setPenAlpha(self, name, alpha=255, width=3):
        """change the alpha channel and width of a curves pen"""
        for param in self.plotParams:
            if not param == "next_row":
                label = param["label"]
                if name in self.curves and label in self.curves[name]:
                    curve = self.curves[name][label]
                    pen = curve.opts["pen"]
                    pencolor = pen.color()
                    pencolor.setAlpha(alpha)
                    pen = pg.mkPen(color=pencolor, width=width, style=pen.style())
                    curve.setPen(pen)

    def clear(self):
        for n, param in enumerate(self.plotParams):
            if param == "next_row":
                pass
            else:
                self.multiPlotWidgets[param["label"]].clear()
        self.plotColor = 0
        self.curves = {}
        self.shadowCurves = []


pg.setConfigOptions(antialias=True)
pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")


def main():
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")
    ex = mainWindow()
    ex.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
