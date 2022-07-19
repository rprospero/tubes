from mantid.kernel import *
from mantid.api import *
from mantid.simpleapi import Load, Rebin, SaveNexusProcessed, RenameWorkspace, CropWorkspace, Scale

import itertools
import numpy as np
import os.path
import copy
import sys

INF = sys.float_info.max # Convenient approximation for infinity

def pairwise(iterable):
    """Helper function from: http://docs.python.org/2/library/itertools.html:
    s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return list(zip(a, b))

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


    @staticmethod
    def set_counts_to_one_between_x_range(ws, x_1, x_2):
        """"""
        if x_1 > x_2: x_1, x_2 = x_2, x_1
        for wsIndex in range(ws.getNumberHistograms()):
            try:
                if x_1 < ws.getDetector(wsIndex).getPos().getX() < x_2:
                    ws.dataY(wsIndex)[0] = 1
            except RuntimeError:
                break
                # pass # Ignore "Detector with ID _____ not found" errors.

    def set_counts_to_one_outside_x_range(self, ws, x_1, x_2):
        """"""
        if x_1 > x_2: x_1, x_2 = x_2, x_1
        self.set_counts_to_one_between_x_range(ws, -INF, x_1)
        self.set_counts_to_one_between_x_range(ws, x_2, INF)

    def get_integrated_workspace(self, data_file):
        """Load a rebin a tube calibration run.  Searched multiple levels of cache to ensure faster loading."""
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


    @staticmethod
    def get_merged_edge_pairs_and_boundaries(edge_pairs):
        """Merge overlapping edge pairs, then return the merged edges and the midpoint of each edge pair."""
        #FIXME: There's probably a cleaner way to do this. ALW 2022
        boundaries = [-INF]
        edge_pairs_merged = []

        temp = edge_pairs[0]

        for start, end in sorted([sorted(edge_pair) for edge_pair in edge_pairs]):
            if start <= temp[1]:
                boundary = start + (temp[1] - start) / 2
                temp[1] = max(temp[1], end)
                if start != temp[0]:
                    boundaries.append(boundary)
            else:
                boundaries.append(temp[1] + (start - temp[1]) / 2)
                edge_pairs_merged.append(tuple(temp))
                temp[0] = start
                temp[1] = end
        edge_pairs_merged.append(tuple(temp))
        boundaries.append(INF)

        return edge_pairs_merged, boundaries

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
        self.declareProperty('Rear Detector', True, direction=Direction.Input,
                             doc="Whether to use the front or rear detector.")
        self.declareProperty('Threshold', 600, direction=Direction.Input,
                             doc="Threshold is the number of counts past which we class something as an edge.  This is quite sensitive to change, since we sometimes end up picking.")
        self.declareProperty('Starting Pixel', 20, direction=Direction.Input,
                             doc="Lower bound of detector's active region")
        self.declareProperty('Ending Pixel', 495, direction=Direction.Input,
                             doc="Upper bound of detector's active region")
        self.declareProperty('Fit Edges', False, direction=Direction.Input,
                             doc="FIXME: Fit the full edge of a shadow, instead of just the top and bottom.")
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
        self.rear = self.getProperty("Rear Detector").value


        if self.rear:
            index1 = 0
            index2 = 120 * 512 - 1
        else:
            index1 = 120*512
            index2 = 2*120*512 -1

        data_files = [self._parse_strip(x) for x in self.getProperty("Strip positions").value]

        known_edge_pairs = np.array([self._strip_edges[x[0]] for x in data_files])
        data_files = [x[1] for x in data_files]

        ws_list = [self.get_integrated_workspace(data_file) for data_file in data_files]

        # Scale workspaces
        i = 0
        def charge(ws):
            return mtd[ws].getRun()["proton_charge_by_period"].value

        uamphr_to_rescale = charge(data_files[0].split('.')[0])
        for ws in data_files:
            ws2 = ws.split('.')[0]
            CropWorkspace(InputWorkspace=ws2, OutputWorkspace=ws2 + '_scaled', StartWorkspaceIndex=index1,
                        EndWorkspaceIndex=index2)
            Scale(uamphr_to_rescale / charge(ws2), "Multiply", InputWorkspace=ws2 + '_scaled', OutputWorkspace=ws2 + '_scaled')
            i += 1


        known_left_edge_pairs = copy.copy(known_edge_pairs)

        _, boundaries = self.get_merged_edge_pairs_and_boundaries(known_edge_pairs)
        known_left_edges, _ = self.get_merged_edge_pairs_and_boundaries(known_left_edge_pairs)


        for ws, (boundary_start, boundary_end) in zip(ws_list, pairwise(boundaries)):
            print(("Isolating shadow in %s between boundaries %g and %g." % (str(ws), boundary_start, boundary_end)))
            # set to 1 so that we can multiply all the shadows together, instead of running merged workspace 5 times.
            ws2 = str(ws) + '_scaled'
            self.set_counts_to_one_outside_x_range(mtd[ws2], boundary_start, boundary_end)

# Register algorithm with Mantid
AlgorithmFactory.subscribe(Calibrate)
