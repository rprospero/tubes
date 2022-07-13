# merge calibration files.
# Note that the viewing the merged file in Instrument View will show that the data for the rear detector is missing
#“ this should not be a problem as only the instrument data is used by the TUBECALIBFILE command.
rear_calib_file = "TubeCalibrationTable_512pixel_64393rear_TEST.nxs"
front_calib_file = "TubeCalibrationTable_512pixel_64393front_TEST.nxs"

rear_calib = Load(rear_calib_file)
front_calib = Load(front_calib_file)

det_id_list = []
for ws_index in range(rear_calib.getNumberHistograms()):
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

for det_id in det_id_list:
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
SaveNexus(empty_instr,'C:/Users/vde76979/OneDrive - Science and Technology Facilities Council/Documents/SANS2D/Tube calibration/March_2020/TUBE_SANS2D_BOTH_64393_15Mar20.nxs')
