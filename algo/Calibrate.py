from mantid.kernel import *
from mantid.api import *
from mantid.simpleapi import Load, Rebin, SaveNexusProcessed, RenameWorkspace

import numpy as np
import os.path

class Calibrate(PythonAlgorithm):
    _strip_edges = {
        1040: [-0.562365234,-0.524046455],
        920: [-0.44052572, -0.402347555],
        755: [-0.27475211, -0.236573945],
        590: [-0.1089785, -0.070800335],
        425: [0.056795111, 0.094973275],
        260: [0.22350643, 0.261684595],
        95:  [0.388342331, 0.426520496],
        5:   [0.4787643, 0.516942465]}

    @staticmethod
    def _parse_strip(description):
        """Parse a pair of strip edge position and file name"""
        parts = description.split("=")
        if len(parts) != 2:
            raise RuntimeError("Cannot part strip run '{}'.  Expecting a string in the format of '920=SANS2D00064390.nxs', where 920 is the strip position and SANS2D00064390.nxs is the file name")
        return (int(parts[0]), parts[1])



    def get_integrated_workspace(self, data_file):
        """Load a rebin a tube calibration run."""
        # check to see if have this file already loaded
        ws_name = os.path.splitext(data_file)[0]
        self.log().debug("look for:  {}".format(ws_name))
        try:
            ws = mtd[ws_name]
            self.log().information("Using existing {} workspace".format(ws_name))
            return ws
        except:
            pass
        try:
            ws = Load(Filename="saved_" + data_file, OutputWorkspace=ws_name)
            self.log().information("Loaded saved file from {}.".format("saved_" + data_file))
            return ws
        except:
            pass

        ws = Load(Filename=data_file, OutputWorkspace=ws_name)
        self.log().information("Loaded and integrating data from {}.".format(data_file))
        # turn event mode into histogram with a single bin
        ws = Rebin(ws, self.timebin, PreserveEvents=False)
        # else for histogram data use integration or sumpsectra
        # ws = Integration(ws, OutputWorkspace=ws_name)
        SaveNexusProcessed(ws, "saved_" + data_file)
        RenameWorkspace(ws, ws_name)

        return ws


    def category(self):
        return 'SANS\\TubeCalibration'

    def PyInit(self):
        # Declare properties
        self.declareProperty('Strip Positions',
                             ["920=SANS2D00064390.nxs",
                              "755=SANS2D00064391.nxs",
                              "590=SANS2D00064392.nxs",
                              "425=SANS2D00064393.nxs",
                              "260=SANS2D00064388.nxs"],
                             direction=Direction.Input,
                             doc="Which strip positions were used for which runs")
        self.declareProperty('Threshold', 600, direction=Direction.Input,
                             doc="Threshold is the number of counts past which we class something as an edge.  This is quite sensitive to change, since we sometimes end up picking.")
        self.declareProperty('Starting Pixel', 20, direction=Direction.Input,
                             doc="Lower bound of detector's active region")
        self.declareProperty('Ending Pixel', 495, direction=Direction.Input,
                             doc="Upper bound of detector's active region")
        self.declareProperty('Fit Edges', False, direction=Direction.Input,
                             doc="FIXME: Fit the full edge of a shadow, instead of just the top and bottom.")
        self.declareProperty('Use Saved Files', False, direction=Direction.Input,
                             doc="Use a preprocessed saved file, instead of manually rebinning.")

        self.declareProperty('Time bins', '5000,93000,98000', direction=Direction.Input,
                             doc="Time of flight bins to use")
        self.declareProperty('Background', 10, direction=Direction.Input,
                             doc="Baseline detector background")
        self.declareProperty('Vertical Offset', -0.005, direction=Direction.Input,
                             doc="Estimate of how many metres off-vertical the Cd strip is at bottom of the detector. Negative if strips are more to left at bottom than top of cylindrical Y plot.")

    def PyExec(self):
        # Run the algorithm
        self.BACKGROUND = self.getProperty("Background").value
        self.timebin = self.getProperty("Time Bins").value
        self.OFF_VERTICAL = self.getProperty("Vertical Offset").value
        self.THRESHOLD = self.getProperty("Threshold").value
        self.STARTPIXEL = self.getProperty("Starting Pixel").value
        self.ENDPIXEL = self.getProperty("Ending Pixel").value
        self.FITEDGES = self.getProperty("Fit Edges").value
        self.USESAVEDFILES = self.getProperty("Use Saved Files").value

        data_files = [self._parse_strip(x) for x in self.getProperty("Strip positions").value]

        known_edge_pairs = np.array([self._strip_edges[x[0]] for x in data_files])
        data_files = [x[1] for x in data_files]

        ws_list = [self.get_integrated_workspace(data_file) for data_file in data_files]

# Register algorithm with Mantid
AlgorithmFactory.subscribe(Calibrate)
