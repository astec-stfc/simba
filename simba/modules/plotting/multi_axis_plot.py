from simba._compat import DeprecatedMethodAliases
import sys, os, time, math, datetime, copy, re, h5py
from copy import copy
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

sys.path.append(os.path.abspath(os.path.realpath(__file__) + "/../../../"))

sys.path.append(os.path.realpath(__file__) + "/../../../../")


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.resize(1800, 900)
        self.centralWidget = QWidget()
        self.layout = QVBoxLayout()
        self.centralWidget.setLayout(self.layout)

        self.tab = QTabWidget()
        self.multiaxisPlot = MultiAxisPlotWidget()

        self.layout.addWidget(self.multiaxisPlot)

        self.setCentralWidget(self.centralWidget)

        self.setWindowTitle("ASTRA Data Plotter")
        menubar = self.menuBar()
        fileMenu = menubar.addMenu("&File")

        exitAction = QAction("&Exit", self)
        exitAction.setShortcut("Ctrl+Q")
        exitAction.setStatusTip("Exit application")
        exitAction.triggered.connect(self.close)
        fileMenu.addAction(exitAction)


class MultiAxisPlotWidget(DeprecatedMethodAliases, QWidget):
    """QWidget containing pyqtgraph plot showing beam parameters"""

    # Legacy names, served with a FutureWarning by the mixin, which
    # also warns if a subclass overrides one under its old name.
    _DEPRECATED_METHOD_ALIASES = {
        "addCurve": "add_curve",
        "addShadowPen": "add_shadow_pen",
        "highlightPlot": "highlight_plot",
        "plotParams": "plot_params",
        "removeCurve": "remove_curve",
        "setPenAlpha": "set_pen_alpha",
        "updateMultiAxisPlot": "update_multi_axis_plot",
        "updateViews": "update_views",
    }

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

    plot_params = []

    def __init__(self, **kwargs):
        super(MultiAxisPlotWidget, self).__init__(**kwargs)
        """ These are for reading data files from ASTRA and Elegant """
        self.multiaxisPlotWidget = QWidget()
        self.multiaxisPlotLayout = QVBoxLayout()
        self.multiaxisPlotWidget.setLayout(self.multiaxisPlotLayout)
        self.multiaxisPlotWidgetGraphicsLayout = pg.GraphicsLayoutWidget()
        self.multiaxisPlotCheckbox = {}
        self.viewboxes = {}
        self.curves = {}
        self.multiaxisaxis = {}
        self.multiaxisPlotCheckboxWidget = QWidget()
        self.multiaxisPlotCheckboxLayout = QGridLayout()
        self.multiaxisPlotCheckboxWidget.setLayout(self.multiaxisPlotCheckboxLayout)
        self.multiaxisPlot = self.multiaxisPlotWidgetGraphicsLayout.addPlot(
            title="Slice", row=2, col=len(self.plot_params) + 1, rowspan=1, colspan=1
        )
        self.multiaxisPlot.showAxis("left", False)
        self.multiaxisPlot.showGrid(x=True, y=True)
        self.multiaxisPlot.vb.sigResized.connect(self.update_views)
        self.shadowCurves = []
        i = 0
        for param in self.plot_params:
            axis = pg.AxisItem("left")
            labelStyle = {"color": "#" + pg.colorStr(pg.mkColor(self.colors[i]))[0:-2]}
            axis.setLabel(text=param["name"], units=param["units"], **labelStyle)
            axis.setZValue(-10000)
            viewbox = pg.ViewBox()
            self.viewboxes[param["label"]] = viewbox
            axis.linkToView(viewbox)
            viewbox.setXLink(self.multiaxisPlot.vb)
            self.multiaxisaxis[param["label"]] = [axis, viewbox]
            self.multiaxisPlotWidgetGraphicsLayout.addItem(
                axis, row=2, col=(len(self.plot_params) - i), rowspan=1, colspan=1
            )
            self.multiaxisPlot.scene().addItem(viewbox)
            viewbox.setLimits(yMin=0)
            self.multiaxisPlotCheckbox[param["label"]] = QCheckBox(param["label"])
            self.multiaxisPlotCheckbox[param["label"]].setChecked(False)

            self.multiaxisPlotCheckboxLayout.addWidget(
                self.multiaxisPlotCheckbox[param["label"]], i % 3, int(i / 3)
            )
            self.multiaxisPlotCheckbox[param["label"]].stateChanged.connect(
                self.update_multi_axis_plot
            )
            i += 1

        self.multiaxisPlotAxisWidget = QWidget()
        self.multiaxisPlotAxisLayout = QHBoxLayout()
        self.multiaxisPlotAxisWidget.setLayout(self.multiaxisPlotAxisLayout)
        self.multiaxisPlotAxisLayout.addWidget(self.multiaxisPlotCheckboxWidget)
        self.multiaxisPlotLayout.addWidget(self.multiaxisPlotAxisWidget)
        self.multiaxisPlotLayout.addWidget(self.multiaxisPlotWidgetGraphicsLayout)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.layout.addWidget(self.multiaxisPlotWidget)

        """ used for style cycling """
        self.plotColor = 0

    def update_views(self):
        for param in self.plot_params:
            ax, vb = self.multiaxisaxis[param["label"]]
            vb.setGeometry(self.multiaxisPlot.sceneBoundingRect())

    def update_multi_axis_plot(self):
        """update main plot"""
        for n, param in enumerate(self.plot_params):
            label = param["label"]
            if self.multiaxisPlotCheckbox[label].isChecked():
                for c in self.curves:
                    self.curves[c][label].setVisible(True)
                self.multiaxisaxis[label][0].setVisible(True)
            else:
                for c in self.curves:
                    self.curves[c][label].setVisible(False)
                self.multiaxisaxis[label][0].setVisible(False)
            self.multiaxisaxis[label][1].autoRange()
            currentrange = self.multiaxisaxis[label][1].viewRange()
            self.multiaxisaxis[label][1].setYRange(0, currentrange[1][1])

    def add_curve(self, x, y, name, label, pen):
        self.curves[name][label] = pg.PlotDataItem()
        self.curves[name][label].curve.setClickable(True)
        self.curves[name][label].sigClicked.connect(lambda: self.highlight_plot(name))
        self.viewboxes[label].addItem(self.curves[name][label])
        self.curves[name][label].setData(x=x, y=y, pen=pen)

    def remove_curve(self, directory, filename=None):
        """finds all multiaxis curves based on a set of names, and removes them"""
        if not isinstance(directory, (list, tuple)):
            directory = [directory]
        for d in directory:
            d = d + "/" + filename if filename is not None else d
            for n, param in enumerate(self.plot_params):
                if c in self.curves[d]:
                    self.multiaxisPlotWidgets[param["label"]].removeItem(
                        c[param["label"]]
                    )

    def highlight_plot(self, name):
        """highlights a particular plot"""
        # print('highligher clicked! = ', name)
        if not isinstance(name, (list, tuple)):
            name = [name]
        for n in name:
            self.add_shadow_pen(n)
            for n in self.curves.keys():
                if n in self.shadowCurves or not len(self.shadowCurves) > 0:
                    self.set_pen_alpha(n, 255, 3)
                else:
                    self.set_pen_alpha(n, 25, 3)

    def add_shadow_pen(self, name):
        for param in self.plot_params:
            if not param == "next_row":
                label = param["label"]
                curve = self.curves[name][label]
                if curve.opts["shadowPen"] is None:
                    self.shadowCurves.append(name)
                    pen = curve.opts["pen"]
                    shadowpencolor = pen.color()
                    shadowpencolor.setAlpha(100)
                    shadowpen = pg.mkPen(color=shadowpencolor, width=(pen.width() + 3))
                    curve.setShadowPen(shadowpen)
                else:
                    self.shadowCurves.remove(name)
                    curve.setShadowPen(None)
                    curve.opts["shadowPen"] = None

    def set_pen_alpha(self, name, alpha=255, width=3):
        for param in self.plot_params:
            if not param == "next_row":
                label = param["label"]
                curve = self.curves[name][label]
                pen = curve.opts["pen"]
                pencolor = pen.color()
                pencolor.setAlpha(alpha)
                pen = pg.mkPen(color=pencolor, width=width, style=pen.style())
                curve.setPen(pen)


pg.setConfigOptions(antialias=True)
pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")


def main():
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


from simba._compat import deprecated_aliases  # noqa: E402

__getattr__ = deprecated_aliases(
    __name__,
    globals(),
    {
        "mainWindow": "MainWindow",
        "multiAxisPlotWidget": "MultiAxisPlotWidget",
    },
)
