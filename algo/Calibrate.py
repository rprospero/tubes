from mantid.kernel import *
from mantid.api import *
from mantid.simpleapi import Load, Rebin, SaveNexusProcessed, RenameWorkspace, CropWorkspace, Scale, CloneWorkspace, Multiply, ApplyCalibration, CreateEmptyTableWorkspace

import itertools
import numpy as np
import os.path
import copy
import sys

from tube_spec import TubeSpec
from tube_calib_fit_params import TubeCalibFitParams
import tube_RKH

class TubeSide:
    LEFT = "left"
    RIGHT = "right"

    @classmethod
    def getTubeSide(cls, tube_id):
        if tube_id % 2 == 0:
            return TubeSide.LEFT
        else:
            return TubeSide.RIGHT

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
    def multiply_ws_list(ws_list, output_ws_name):
        print("Multiplying workspaces together...")
        it = iter(ws_list)
        total = str(next(it)) + '_scaled'
        for element in it:
            ws = str(element) + '_scaled'
            total = Multiply(RHSWorkspace=total, LHSWorkspace=ws, OutputWorkspace=output_ws_name)
        return total

    @staticmethod
    def get_tube_name(tube_id, detector_name):
        # Construct the name of the tube based on the id (0-119) given.
        side = TubeSide.getTubeSide(tube_id)
        tube_side_num = tube_id // 2  # Need int name, not float appended
        return detector_name + "-detector/" + side + str(tube_side_num)


    def get_tube_data(self, tube_id, ws, detector_name):
        tube_name = self.get_tube_name(tube_id, detector_name)

        # Piggy-back the TubeSpec class from Karl's Calibration code so that dealing with tubes is easier than interrogating the IDF ourselves.
        tube_spec = TubeSpec(ws)
        tube_spec.setTubeSpecByString(tube_name)
        assert tube_spec.getNumTubes() == 1
        tube_ws_index_list = tube_spec.getTube(0)[0]
        assert len(tube_ws_index_list) == 512

        # Return an array of all counts for the tube.
        return np.array([ws.dataY(ws_index)[0] for ws_index in tube_ws_index_list])

    def get_tube_edge_pixels(self, detector_name, tube_id, ws, cutoff, first_pixel=0, last_pixel=sys.maxsize):
        count_data = self.get_tube_data(tube_id, ws, detector_name)

        if count_data[first_pixel] < cutoff:
            up_edge = True
        else:
            up_edge = False

        for i, count in enumerate(count_data[first_pixel:last_pixel + 1]):
            pixel = first_pixel + i
            if pixel > last_pixel:
                break
            if up_edge:
                if count >= cutoff:
                    up_edge = False
                    yield pixel
            else:
                if count < cutoff:
                    up_edge = True
                    yield pixel

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

    def get_integrated_workspace(self, data_file, prog):
        """Load a rebin a tube calibration run.  Searched multiple levels of cache to ensure faster loading."""
        # check to see if have this file already loaded
        ws_name = os.path.splitext(data_file)[0]
        self.log().debug("look for:  {}".format(ws_name))
        try:
            ws = mtd[ws_name]
            self.log().information("Using existing {} workspace".format(ws_name))
            prog.report("Loading {}".format(ws_name))
            return ws
        except:
            pass
        try:
            ws = Load(Filename="saved_" + data_file, OutputWorkspace=ws_name)
            self.log().information("Loaded saved file from {}.".format("saved_" + data_file))
            prog.report("Loading {}".format(ws_name))
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

        prog.report("Loading {}".format(ws_name))
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
        self.declareProperty('Margin', 25, direction=Direction.Input,
                             doc="FIXME: Detector margin")
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
        margin = self.getProperty("Margin").value
        OFF_VERTICAL = self.getProperty("Vertical Offset").value
        THRESHOLD = self.getProperty("Threshold").value
        STARTPIXEL = self.getProperty("Starting Pixel").value
        ENDPIXEL = self.getProperty("Ending Pixel").value
        FITEDGES = self.getProperty("Fit Edges").value
        self.rear = self.getProperty("Rear Detector").value


        if self.rear:
            index1 = 0
            index2 = 120 * 512 - 1
            detector_name = "rear"
        else:
            index1 = 120*512
            index2 = 2*120*512 -1
            detector_name = "front"

        data_files = [self._parse_strip(x) for x in self.getProperty("Strip positions").value]

        known_edge_pairs = np.array([self._strip_edges[x[0]] for x in data_files])
        data_files = [x[1] for x in data_files]

        load_report = Progress(self, start=0, end=0.9, nreports=len(data_files))
        ws_list = [self.get_integrated_workspace(data_file, load_report) for data_file in data_files]

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

        result_ws_name = "result"

        self.multiply_ws_list(ws_list, result_ws_name)

        result = mtd[result_ws_name]

        original = CloneWorkspace(InputWorkspace=result_ws_name, OutputWorkspace="original")

        known_edges_left = list(itertools.chain.from_iterable(known_left_edges))
        failed_pixel_guesses = []
        pixel_guesses = []
        meanCvalue = []
        tubeList = []

        dx = (522.2 + 519.2) / 511
        # default size of a pixel in real space in mm
        # setting caltable to None  will start all over again, comment this line out to add further tubes to existing table
        caltable = None
        # caltable = True
        diag_output = dict()

        tube_report = Progress(self, start=0.9, end=1.0, nreports=120)
        for tube_id in range(120):
            # for tube_id in range(116,120):
            diag_output[tube_id] = []

            tube_name = self.get_tube_name(tube_id, detector_name)
            tube_report.report("Calculating tube {}".format(tube_name))
            print("\n==================================================")
            print(("ID = %i, Name = \"%s\"" % (tube_id, tube_name)))
            known_edges1 = copy.copy(known_edges_left)
            if TubeSide.getTubeSide(tube_id) == TubeSide.LEFT:
                # first pixel in mm, as per idf file for rear detector
                x0 = -519.2
            else:
                x0 = -522.2

            np.array(known_edges1)

            known_edges = []
            for index in range(len(known_edges1)):
                known_edges.append(known_edges1[index] + (tube_id - 119.0) * OFF_VERTICAL / 119.0)

            guessed_pixels = list(self.get_tube_edge_pixels(detector_name, tube_id, result, THRESHOLD, STARTPIXEL, ENDPIXEL))

            # Store the guesses for printing out later, along with the tube id and name.
            # pixel_guesses.append([tube_name, guessed_pixels])

            print(), print((len(guessed_pixels), guessed_pixels))
            print(), print((len(known_edges), known_edges))

            # note funcForm==2 fits an edge using error function, (see SANS2DEndErfc above, and code in tube_calib_RKH.py,) while any other value fits a Gaussian
            if FITEDGES:
                funcForm = [2] * len(guessed_pixels)
                fitPar = TubeCalibFitParams(guessed_pixels, outEdge=10.0, inEdge=10.0)
            else:
                # average pairs of edges for single peak fit, could in principle do only some tubes or parts of tubes this way!
                guess = []
                known = []
                for i in range(0, len(guessed_pixels), 2):
                    guess.append((guessed_pixels[i] + guessed_pixels[i + 1]) / 2)
                    known.append((known_edges[i] + known_edges[i + 1]) / 2)
                funcForm = [3] * len(guess)
                guessed_pixels = guess
                known_edges = known
                fitPar = TubeCalibFitParams(guessed_pixels, height=2000, width=2 * margin, margin=margin, outEdge=10.0,
                                            inEdge=10.0)
                fitPar.setAutomatic(False)
                print(("halved guess ", len(guessed_pixels), guessed_pixels))
                print(("halved known ", len(known_edges), known_edges))

            module = int(tube_id / 24) + 1
            tube_num = tube_id - (module - 1) * 24
            print(("module ", module, "   tube ", tube_num))

            if caltable:
                # does this after the first time, it appends to a table
                caltable, peakTable, meanCTable = tube_RKH.calibrate(
                    result,
                    tube_name,
                    np.array(known_edges),
                    funcForm,
                    # outputPeak=peakTable falls over at addrow if number of peaks changes, so can only save one row at a time
                    outputPeak=True,
                    outputC=True,
                    rangeList=[0],
                    plotTube=[0],
                    margin=margin,
                    fitPar=fitPar,
                    calibTable=caltable)
            else:
                # do this the FIRST time, starts a new table
                print("first time, generate calib table")
                caltable, peakTable, meanCTable = tube_RKH.calibrate(
                    result,
                    tube_name,
                    np.array(known_edges),
                    funcForm,
                    # outputPeak=True,
                    rangeList=[0],
                    plotTube=[0],
                    outputPeak=True,
                    outputC=True,
                    margin=margin,
                    fitPar=fitPar)
            diag_output[tube_id].append(CloneWorkspace(InputWorkspace="FittedTube0",
                                        OutputWorkspace="Fit" + str(tube_id) + "_" + str(module) + "_" + str(tube_num)))
            diag_output[tube_id].append(CloneWorkspace(InputWorkspace="TubePlot0",
                                        OutputWorkspace="Tube" + str(tube_id) + "_" + str(module) + "_" + str(tube_num)))
            # 8/7/14 save the fitted positions to see how well the fit does, all in mm
            x_values = []
            x0_values = []
            bb = list(mtd["PeakTable"].row(0).values())
            del bb[0]  # Remove string that can't be sorted
            bb.sort()
            # bb still contains a name string at the end
            # it's just for disgnostics
            for i in range(len(guessed_pixels)):
                x0_values.append(bb[i] * dx + x0)
                x_values.append(known_edges[i] * 1000. - bb[i] * dx - x0)
            cc = CreateWorkspace(DataX=x0_values, DataY=x_values)
            diag_output[tube_id].append(RenameWorkspace(InputWorkspace="cc",
                                        OutputWorkspace="Data" + str(tube_id) + "_" + str(module) + "_" + str(tube_num)))

            bb = list(mtd["meanCTable"].row(0).values())
            meanCvalue.append(bb[1])
            tubeList.append(tube_id)

        ApplyCalibration(result, caltable)
        print(tubeList)
        print(meanCvalue)
        cvalues = CreateWorkspace(DataX=tubeList, DataY=meanCvalue)

        print(outputfilename)
        SaveNexusProcessed(result, outputfilename)
        # you will next need to run merge_calib_files.py to merge the tables for front and rear detectors

        # expts
        aa = mtd["PeakTable"].row(0)
        print(aa)
        print((aa.get('Peak10')))
        bb = list(aa.values())

        bb = list(mtd["PeakTable"].row(0).values())
        del bb[0]  # Remove string that can't be sorted
        print(bb)
        bb.sort()
        print(bb)

        # now interrogate CalibTable to see how much we have shifted pixels by for each tube
        # 18/3/15 this new version will work when we have skipped tubes as it reads the Detector ID's  from the table itself
        # All this creates ws to check results, doesn't affect the main code
        nentries = int(len(mtd["CalibTable"]) / 512)
        print(("nentries in CalibTable = ", nentries))
        i1 = 0
        i2 = 512
        dx = (522.2 + 519.2) / 511
        for i in range(0, nentries):
            tube_num = mtd["CalibTable"].column("Detector ID")[i1]
            tube_num /= 1000
            det = int(tube_num / 1000)
            tube_num -= det * 1000
            module = int(tube_num / 100)
            tube_num = tube_num - module * 100
            tube_id = (module - 1) * 24 + tube_num
            print((tube_id, module, tube_num))
            x_values = []
            x0_values = []
            # use left tube value here for now, right tube starts at -522.2    WHY THIS HERE ????
            if TubeSide.getTubeSide(tube_id) == TubeSide.LEFT:
                x0 = -519.2
            else:
                x0 = -522.2
            for pos in mtd["CalibTable"].column("Detector Position")[i1:i2]:
                x_values.append(pos.getX() * 1000.0 - x0)
                x0_values.append(x0)
                x0 += dx
            plotN = CreateWorkspace(DataX=x0_values, DataY=x_values)
            diag_output[i].append(RenameWorkspace(InputWorkspace="plotN",
                                OutputWorkspace="Shift" + str(tube_id) + "_" + str(module) + "_" + str(tube_num)))
            i1 = i1 + 512
            i2 = i2 + 512

        for tube_id, workspaces in diag_output.items():
            GroupWorkspaces(InputWorkspaces=workspaces, OutputWorkspace=f"Tube_{tube_id:03}")

        for x in (i for j in (list(range(0, 2)), list(range(10, 12)), list(range(23, 29))) for i in j):
            print(x)
            # Notice to self, the result will look wiggly in 3D but looks good in cylindrical Y

# Register algorithm with Mantid
AlgorithmFactory.subscribe(Calibrate)
