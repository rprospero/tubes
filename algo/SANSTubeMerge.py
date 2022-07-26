from mantid.kernel import *
from mantid.api import *
from mantid.simpleapi import *


class SANSTubeMerge(PythonAlgorithm):
    def category(self):
        return 'SANS\\Calibration'

    def summary(self):
        return 'Merge the calibration of the SANS Tubes'

    def PyInit(self):
        # Declare properties
        self.declareProperty(FileProperty(name="Front", defaultValue="",
                                          action=FileAction.Load,
                                          extensions=["nxs"]))
        self.declareProperty(FileProperty(name="Rear", defaultValue="",
                                          action=FileAction.Load,
                                          extensions=["nxs"]))

    def PyExec(self):
        # Run the algorithm
        load_report = Progress(self,
                               start=0.0,
                               end=0.5,
                               nreports=2)
        load_report.report("Loading Rear")
        rear_calib = Load(self.getProperty("Rear").value)
        load_report.report("Loading Front")
        front_calib = Load(self.getProperty("Front").value)

        det_id_list = []
        load_report = Progress(self,
                               start=0.5,
                               end=0.75,
                               nreports=rear_calib.getNumberHistograms())
        for ws_index in range(rear_calib.getNumberHistograms()):
                load_report.report(f"Loading {ws_index}")
                spectrum = rear_calib.getSpectrum(ws_index)
                if spectrum.getSpectrumNo() < 0:
                        continue
                for det_id in spectrum.getDetectorIDs():
                        if det_id > 2523511:
                                continue
                        det_id_list.append(det_id)

        rear_inst = rear_calib.getInstrument()

        # Creating an unmanaged version of the algorithm is important here.  Otherwise, things get
        # really slow, and the workspace history becomes unmanageable.
        move = AlgorithmManager.createUnmanaged('MoveInstrumentComponent')
        move.initialize()
        move.setChild(True)
        move.setProperty("Workspace", "front_calib")
        move.setProperty("RelativePosition", False)

        load_report = Progress(self,
                               start=0.75,
                               end=1.0,
                               nreports=len(det_id_list))
        for det_id in det_id_list:
                load_report.report(f"Gettings Detector {det_id}")
                det = rear_inst.getDetector(det_id)
                if "rear-detector" in det.getFullName():
                        move.setProperty("DetectorID", det_id)
                        move.setProperty("X", det.getPos().getX())
                        move.setProperty("Y", det.getPos().getY())
                        move.setProperty("Z", det.getPos().getZ())
                        move.setProperty("DetectorID", det_id)
                        move.execute()

        RenameWorkspace(InputWorkspace="front_calib", OutputWorkspace="merged")
        DeleteWorkspace(Workspace="rear_calib")
        #outputfilename='C:/Users/pzt29813/Documents/Sarah/Mantid_stuff/Tubes/TubeCalibrationTable_512pixel_40673BOTH_15Nov16.nxs'
        #SaveNexusProcessed('merged',outputfilename)
        #RemoveWorkspaceHistory('TubeCalibrationTable_512pixel_40673_BOTH_15Nov16')

        empty_instr = LoadEmptyInstrument('c:/MANTIDINSTALL/instrument/SANS2D_Definition_Tubes.xml')
        CopyInstrumentParameters('merged', empty_instr)
        RemoveWorkspaceHistory(empty_instr)

AlgorithmFactory.subscribe(SANSTubeMerge)
