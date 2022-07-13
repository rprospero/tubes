import sys
import itertools
import operator
import copy
import os

import numpy
import scipy.special
from mantid.simpleapi import *

# ========  own version does not use tube centre when moving the tube ! ==============================
from tube_calib_RKH import getCalibration_RKH
import tube_RKH
from tube_spec import TubeSpec
from tube_calib_fit_params import TubeCalibFitParams


def pairwise(iterable):
    """Helper function from: http://docs.python.org/2/library/itertools.html:
    s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return list(zip(a, b))


def edge_pairs_overlap(edges_pair_a, edges_pair_b):
    """"""
    if edges_pair_a[0] < edges_pair_b[0] < edges_pair_a[1]:
        return True
    if edges_pair_b[0] < edges_pair_a[0] < edges_pair_b[1]:
        return True
    return False


def sort_and_merge_overlapping_edge_pairs(edge_pairs):
    """Algorithm from: http://stackoverflow.com/a/5679899/778572"""
    temp = edge_pairs[0]
    for start, end in sorted([sorted(edge_pair) for edge_pair in edge_pairs]):
        if start <= temp[1]:
            temp[1] = max(temp[1], end)
        else:
            yield tuple(temp)
            temp[0] = start
            temp[1] = end
    yield tuple(temp)


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


def set_counts_to_one_outside_x_range(ws, x_1, x_2):
    """"""
    if x_1 > x_2: x_1, x_2 = x_2, x_1
    set_counts_to_one_between_x_range(ws, -INF, x_1)
    set_counts_to_one_between_x_range(ws, x_2, INF)


def get_merged_edge_pairs_and_boundaries(edge_pairs):
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


def get_integrated_workspace(data_file, timebin, USESAVEDFILES):
    # check to see if have this file already loaded 
    ws_name = os.path.splitext(data_file)[0]
    print(("look for:  ", ws_name))
    try:
        ws = mtd[ws_name]
    except:
        if USESAVEDFILES:
            print(("Loading saved file from %s." % "saved_" + data_file))
            ws = Load(Filename="saved_" + data_file, OutputWorkspace=ws_name)
            # RenameWorkspace(ws,ws_name)
        else:
            print(("Loading and integrating data from %s." % data_file))
            ws = Load(Filename=data_file, OutputWorkspace=ws_name)
            # turn event mode into histogram with a single bin
            ws = Rebin(ws, timebin, PreserveEvents=False)
            # else for histogram data use integration or sumpsectra
            # ws = Integration(ws, OutputWorkspace=ws_name)
            SaveNexusProcessed(ws, "saved_" + data_file)
            RenameWorkspace(ws, ws_name)

    return ws


def multiply_ws_list(ws_list, output_ws_name):
    print("Multiplying workspaces together...")
    it = iter(ws_list)
    total = str(next(it)) + '_scaled'
    for element in it:
        ws = str(element) + '_scaled'
        total = Multiply(RHSWorkspace=total, LHSWorkspace=ws, OutputWorkspace=output_ws_name)
    return total


# this is not used
# def get_pixels_from_edges(edges):
#    pixels = []
#    for edge in edges:
#        pixels.append(int((edge + 0.5207) * 512))
#    return numpy.array(pixels)

def get_tube_name(tube_id):
    # Construct the name of the tube based on the id (0-119) given.
    side = TubeSide.getTubeSide(tube_id)
    tube_side_num = tube_id // 2  # Need int name, not float appended
    return "rear-detector/" + side + str(tube_side_num)


def get_tube_data(tube_id, ws):
    tube_name = get_tube_name(tube_id)

    # Piggy-back the TubeSpec class from Karl's Calibration code so that dealing with tubes is easier than interrogating the IDF ourselves.
    tube_spec = TubeSpec(ws)
    tube_spec.setTubeSpecByString(tube_name)
    assert tube_spec.getNumTubes() == 1
    tube_ws_index_list = tube_spec.getTube(0)[0]
    assert len(tube_ws_index_list) == 512

    # Return an array of all counts for the tube.
    return numpy.array([ws.dataY(ws_index)[0] for ws_index in tube_ws_index_list])


def get_tube_edge_pixels(tube_id, ws, cutoff, first_pixel=0, last_pixel=sys.maxsize):
    count_data = get_tube_data(tube_id, ws)

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


def compile_param_table(tube_id, peaks):
    # Dirty hack we need to replace with something better.
    full_table = CreateEmptyTableWorkspace(OutputWorkspace="ParamTable_Tube" + str(tube_id))
    full_table.addColumn("int", "Peak #")
    full_table.addColumn("float", "A")
    full_table.addColumn("float", "B")
    full_table.addColumn("float", "C")
    full_table.addColumn("float", "D")
    full_table.addColumn("float", "E")
    base_name = "CalibPoint_Parameters_tube%i_peak_%i"
    table_names = [base_name % (tube_id, peak) for peak in range(peaks)]
    for peak, table_name in enumerate(table_names):
        table = mtd[table_name]
        row = [
            peak,
            table.cell("Value", 0),
            table.cell("Value", 1),
            table.cell("Value", 2),
            table.cell("Value", 3),
            table.cell("Value", 4)
        ]
        full_table.addRow(row)


def cleanup_param_tables(tube_id, peaks):
    # Dirty hack we need to replace with something better.
    base_name = "CalibPoint_Parameters_tube%i_peak_%i"
    table_names = [base_name % (tube_id, peak) for peak in range(peaks)]
    for table_name in table_names:
        table = mtd[table_name]
        table.delete()


BACKGROUND = 10


# 11/11/19 I think we have been using Mantid DEFAULT EndErfc NOT this function here!
class SANS2DEndErfc(IFunction1D):
    def init(self):
        self.declareParameter("A", 350.0)
        self.declareParameter("B", 50.0)
        self.declareParameter("C", 6.0)
        self.declareParameter("D", BACKGROUND)

    def function1D(self, xvals):
        a = self.getParameterValue("A")
        b = self.getParameterValue("B")
        c = self.getParameterValue("C")
        d = self.getParameterValue("D")
        # erfc goes from -1 to +1
        out = a * scipy.special.erfc((b - xvals) / c) + d
        if a < 0:
            # WHY IS THIS HERE ???????????????
            out = -2 * a * out

        return out

    def setActiveParameter(self, index, value):
        # Heavy-handed way to constrain background.
        if self.parameterName(index) == "D" and value < 0.0:
            self.setParameter(index, 0.0, False)
        elif self.parameterName(index) == "D" and value > BACKGROUND:
            self.setParameter(index, BACKGROUND, False)
        else:
            self.setParameter(index, value, False)


# Required to have Mantid recognise the new function
FunctionFactory.subscribe(SANS2DEndErfc)


# RKH 11/11/19 new experimental peak shape A,B,C,D,E are height, centre, resolution width, background, peak width
# this should allow the error function edges to overlap in the middle of the Cd strip
class FlatTopPeak(IFunction1D):
    def init(self):
        self.declareParameter("A", -2400.)
        self.declareParameter("B", 50.0)
        self.declareParameter("C", 3.0)
        self.declareParameter("D", 2500.)
        self.declareParameter("E", 20.0)

    def function1D(self, xvals):
        a = self.getParameterValue("A")
        b = self.getParameterValue("B")
        c = self.getParameterValue("C")
        d = self.getParameterValue("D")
        e = self.getParameterValue("E")

        out = a * (scipy.special.erfc((b - 0.5 * e - xvals) / c) - scipy.special.erfc((b + 0.5 * e - xvals) / c)) + d
        return out


#    def setActiveParameter(self, index, value):
# Heavy-handed way to constrain background.
#        if self.parameterName(index) == "D" and value < 0.0:
#            self.setParameter(index, 0.0, False)
#        elif self.parameterName(index) == "D" and value > BACKGROUND:
#            self.setParameter(index, BACKGROUND, False)
#        else:
#            self.setParameter(index, value, False)

# Required to have Mantid recognise the new function
FunctionFactory.subscribe(FlatTopPeak)


class TubeSide:
    LEFT = "left"
    RIGHT = "right"

    @classmethod
    def getTubeSide(cls, tube_id):
        if tube_id % 2 == 0:
            return TubeSide.LEFT
        else:
            return TubeSide.RIGHT


# ====================================  Main code starts here ============================================================

INF = sys.float_info.max
# 8/7/14 set this to ZERO not 0.003, as calib goes direct from pixel number to position
TUBE_OFFSET = 0.00

# estimate of how many metres off-vertical the Cd strip is at bottom of the detector, -ve if strips are more to left 
# at bottom than top of cylindrical Y plot
OFF_VERTICAL = -0.005

# THRESHOLD is the number of counts past which we class something as an edge.  This is quite sensitive to change, since we sometimes end up picking
# up edges more than once, (e.g. we might see an edge at pixels 207 and 209, obviously due to the counts dipping back down below the
# threshold at pixel 208) which we then have to remove using ignore_guesses.  So, if THRESHOLD is changed you're probably going
# to have to change the contents of ignore_guesses for any affected tubes.
# STARTPIXEL and ENDPIXEL are where the code looks for edges, these can be used to avoid picking up the actual ends of the active regions
# of the tubes as "edges".
# check the TubeNN and FitNN workspaces overplot well, the sequential fit results are appened end to end, so there may be straight lines between them.
# It is very improtant to check plots of DataNN and ShiftNN workspaces to see whether the quadractic interpolation is actually working OK,
# Do not assume that it is just because the "result" workspace has nice straight edges.
# 
THRESHOLD = 600
STARTPIXEL = 20
ENDPIXEL = 495
FITEDGES = False
USESAVEDFILES = False
margin = 25
ttt = TubeCalibFitParams([0])
ttt.setAutomatic(True)
print((ttt.getAutomatic()))

# 11/11/19 FITEDGES default is True, set to False to use FlatTopPeak on PAIRS OF EDGES,  but increase "margin" from say 10 to 25
#
# event mode files need to be turned to histogram with a single bin (else use sumspectra if histogram files)
timebin = '5000,93000,98000'
#
# Also set THRESHOLD above
#                                     and CHANGE THE NAME OF THE OUTPUT FILE for SaveNexusProcessed towards end of this file
outputfilename = 'TubeCalibrationTable_512pixel_64393rear_TEST.nxs'
# string for start of "original" workspace file names
output_name = '64393'
#  16/9/19                    THIS IS FOR REAR  (NOTE "Load" for event files does not load the monitors !)
index1 = 0
index2 = 120 * 512 - 1
# this is equal number of tubes times number of pixels
# for FRONT use
# index1 = 120*512
# index2 = 2*120*512 -1
# 16/9/19
data_files = [
			  "SANS2D00064390.nxs",
			  "SANS2D00064391.nxs",
			  "SANS2D00064392.nxs",
			  "SANS2D00064393.nxs",
              "SANS2D00064388.nxs"]
			  
			  
uamphr = [140, 143.3, 145.2, 82.5, 141.3]
# can't be done automatically as this often needs hand tweaking
uamphr_to_rescale = 140.0

#The known_edge_pairs are listed below in the following strip position sequence: [1040, 920, 755, 590, 425, 260, 95, 5]
#DON'T ERASE THE LINES BELOW! Just comment the lines that you don't need. Remember to erase the comma of the final pair, if needed.  (LPC)
known_edge_pairs = [
					#[-0.562365234,-0.524046455],
					[-0.44052572, -0.402347555],
					[-0.27475211, -0.236573945],
					[-0.1089785, -0.070800335],
                    [0.056795111, 0.094973275],
					[0.22350643, 0.261684595]
					##[0.388342331, 0.426520496]
					#[0.4787643, 0.516942465],
					]

assert len(known_edge_pairs) == len(data_files)

# === OLD VERSIONS OF IGNORE_GUESSES are overwritten by newer ones below ....
ignore_guesses = {
    25: [0],  # double edge
    29: [0],  # double edge
    45: [7, 8],  # botom b/s, the first edge is ZERO, tube number(list of edges)
    46: [7, 8],  #
    47: [7, 8],  #
    48: [7, 8],  #
    49: [7, 8],  #
    50: [7, 8],  #
    51: [7, 8],  #
    52: [7, 8],  #
    53: [0, 8, 9],  # note the extra edge [0] renumbers 7 & 8 to 8 & 9
    54: [7, 8],  # top b/s
    85: [0],  # double edge
    93: [0],  # double edge
    106: [7, 8],  # finds same edge three times, likely THRESHOLD dependent
    107: [0],  # double edge
    113: [0],  # double edge
    117: [0],  # double edge
    119: [0]  # double edge
}

ignore_guesses = {
    36: [0, 1, 2, 3, 4, 5, 6, 8, 10],
    71: [13]
}
ignore_guesses = {110: [13], 112: [13], 114: [13], 116: [13]
                  }
ignore_guesses = {2: [2, 4], 95: [10, 12]}
ignore_guesses = {}

# === OLD VERSIONS OF edges_not_to_fit are overwritten by newer ones below ....
#    Note that STARTPXIEL and ENDPIXEL help to avoid fining "edges" at the ends of the tubes
edges_not_to_fit = {
    0: [0, 15],
    1: [0, 15],
    2: [0, 15],
    3: [0, 15],
    4: [0, 15],
    5: [0, 15],
    6: [0, 15],
    7: [0, 15],
    8: [0, 15],
    9: [0, 15],
    10: [0, 15],
    11: [0, 15],
    12: [0, 15],
    13: [0, 15],
    14: [0, 15],
    15: [0, 15],
    16: [0, 15],
    17: [0, 15],
    18: [0, 15],
    19: [0, 15],
    20: [0, 15],
    21: [0, 15],
    22: [0, 15],
    23: [0, 15],
    24: [0, 15],
    25: [0, 15],
    26: [0, 15],
    27: [0, 15],
    28: [0, 15],
    29: [0, 15],
    30: [0, 15],
    31: [0, 15],
    32: [0, 15],
    33: [0, 15],
    34: [0, 15],
    35: [0, 15],
    36: [0, 15],
    37: [0, 15],
    38: [0, 15],
    39: [0, 15],
    40: [0, 15],
    41: [0, 15],
    42: [0, 15],
    43: [0, 15],
    44: [0, 15],
    45: [0, 15],
    46: [0, 15],
    47: [0, 15],
    48: [0, 15],
    49: [0, 15],
    50: [0, 15],
    51: [0, 15],
    52: [0, 15],
    53: [0, 15],
    54: [0, 15],
    55: [0, 15],
    56: [0, 15],
    57: [0, 15],
    58: [0, 15],
    59: [0, 15],
    60: [0, 15],
    61: [0, 15],
    62: [0, 15],
    63: [0, 15],
    64: [0, 15],
    65: [0, 15],
    66: [0, 15],
    67: [0, 15],
    68: [0, 15],
    69: [0, 15],
    70: [0, 15],
    71: [0, 15],
    72: [0, 15],
    73: [0, 15],
    74: [0, 15],
    75: [0, 15],
    76: [0, 15],
    77: [0, 15],
    78: [0, 15],
    79: [0, 15],
    80: [0, 15],
    81: [0, 15],
    82: [0, 15],
    83: [0, 15],
    84: [0, 15],
    85: [0, 15],
    86: [0, 15],
    87: [0, 15],
    88: [0, 15],
    89: [0, 15],
    90: [0, 15],
    91: [0, 15],
    92: [0, 15],
    93: [0, 15],
    94: [0, 15],
    95: [0, 15],
    96: [0, 15],
    97: [0, 15],
    98: [0, 15],
    99: [0, 15],
    100: [0, 15],
    101: [0, 15],
    102: [0, 15],
    103: [0, 15],
    104: [0, 15],
    105: [0, 15],
    106: [0, 15],
    107: [0, 15],
    108: [0, 15],
    109: [0, 15],
    110: [0, 15],
    111: [0, 15],
    112: [0, 15],
    113: [0, 15],
    114: [0, 15],
    115: [0, 15],
    116: [0, 15],
    117: [0, 15],
    118: [0, 15],
    119: [0, 15]
}

edges_not_to_fit = {
}

# Array indices of shadows (edge pairs) to remove.
shadows_to_remove = []

for shadow in reversed(sorted(shadows_to_remove)):
    del known_edge_pairs[shadow]
    del data_files[shadow]

known_edge_pairs = numpy.array(known_edge_pairs)

# this gets whole file, is cropped for front or rear bank below, so only have to do this ONCE
ws_list = [get_integrated_workspace(data_file, timebin, USESAVEDFILES) for data_file in data_files]

# scale to same uampHr  RKH 16/9/19 OOPS this could repeatedly rescale same data if re-run the code
#  so copy into _scaled workspace, then multiply the _scaled ones together, this preserves the originals for later attempts.
i = 0
for ws in data_files:
    ws2 = ws.split('.')[0]
    CropWorkspace(InputWorkspace=ws2, OutputWorkspace=ws2 + '_scaled', StartWorkspaceIndex=index1,
                  EndWorkspaceIndex=index2)
    print((ws2, uamphr[i]))
    Scale(uamphr_to_rescale / uamphr[i], "Multiply", InputWorkspace=ws2 + '_scaled', OutputWorkspace=ws2 + '_scaled')
    i += 1

known_left_edge_pairs = copy.copy(known_edge_pairs)
known_right_edge_pairs = copy.copy(known_edge_pairs + TUBE_OFFSET)

_, boundaries = get_merged_edge_pairs_and_boundaries(known_edge_pairs)
known_left_edges, _ = get_merged_edge_pairs_and_boundaries(known_left_edge_pairs)
known_right_edges, _ = get_merged_edge_pairs_and_boundaries(known_right_edge_pairs)

for ws, (boundary_start, boundary_end) in zip(ws_list, pairwise(boundaries)):
    print(("Isolating shadow in %s between boundaries %g and %g." % (str(ws), boundary_start, boundary_end)))
    # set to 1 so that we can multiply all the shadows together, instead of running merged workspace 5 times.
    ws2 = str(ws) + '_scaled'
    set_counts_to_one_outside_x_range(mtd[ws2], boundary_start, boundary_end)

result_ws_name = "result"

multiply_ws_list(ws_list, result_ws_name)

result = mtd[result_ws_name]

original = CloneWorkspace(InputWorkspace=result_ws_name, OutputWorkspace="original")
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX              SaveNexusProcessed(original,output_name+'_strips_original.nxs')

# eek      ================================  START HERE IF LOADING A PREVIOUS STITCHED "original" to process ============================
# LoadNexusProcessed(OutputWorkspace="result",Filename='40673_strips_original_rear.nxs')
# original = CloneWorkspace(InputWorkspace=result_ws_name, OutputWorkspace="original")

known_edges_left = list(itertools.chain.from_iterable(known_left_edges))
known_edges_right = list(itertools.chain.from_iterable(known_right_edges))

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
for tube_id in range(120):
    # for tube_id in range(116,120):
    diag_output[tube_id] = []

    tube_name = get_tube_name(tube_id)
    print("\n==================================================")
    print(("ID = %i, Name = \"%s\"" % (tube_id, tube_name)))
    if TubeSide.getTubeSide(tube_id) == TubeSide.LEFT:
        known_edges1 = copy.copy(known_edges_left)
        # first pixel in mm, as per idf file for rear detector
        x0 = -519.2
    else:
        known_edges1 = copy.copy(known_edges_right)
        x0 = -522.2

    numpy.array(known_edges1)

    known_edges = []
    for index in range(len(known_edges1)):
        known_edges.append(known_edges1[index] + (tube_id - 119.0) * OFF_VERTICAL / 119.0)

    guessed_pixels = list(get_tube_edge_pixels(tube_id, result, THRESHOLD, STARTPIXEL, ENDPIXEL))

    # Store the guesses for printing out later, along with the tube id and name.
    # pixel_guesses.append([tube_name, guessed_pixels])
	
    print(), print((len(guessed_pixels), guessed_pixels))
    print(), print((len(known_edges), known_edges))

    # Remove the pixel guesses that don't correspond to Cd shadows:
    if tube_id in ignore_guesses:
        print(("Removing guesses", list(reversed(sorted(ignore_guesses[tube_id])))))
        for index in reversed(sorted(ignore_guesses[tube_id])):
            del guessed_pixels[index]
        print((len(guessed_pixels), guessed_pixels))
        print((len(known_edges), known_edges))
    assert len(guessed_pixels) == len(known_edges)

    # Remove the edges that have been altered by the presence of the beam stop and arm.
    if tube_id in edges_not_to_fit:
        for index in reversed(sorted(edges_not_to_fit[tube_id])):
            del guessed_pixels[index]
            del known_edges[index]

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
            numpy.array(known_edges),
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
            numpy.array(known_edges),
            funcForm,
            # outputPeak=True,
            rangeList=[0],
            plotTube=[0],
            outputPeak=True,
            outputC=True,
            margin=margin,
            fitPar=fitPar)
        # BUG inMantid 4.0 gives this message - try more recent Mantid, but some changes have been made to tube.py and tube_calib.py which need replicating in tube_RKH.py and tube_calib_RKH.py
        #                           TypeError: slice indices must be integers or None or have an __index__ method  
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
