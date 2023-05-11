#####################################################################
#                                                                   #
# Copyright 2019, Monash University and contributors                #
#                                                                   #
# This file is part of labscript_devices, in the labscript suite    #
# (see http://labscriptsuite.org), and is licensed under the        #
# Simplified BSD License. See the license.txt file in the root of   #
# the project for the full license.                                 #
#                                                                   #
#####################################################################

import labscript_devices

labscript_devices.register_classes(
  "SpectrumAwg",
  BLACS_tab = "user_devices.spectrum_awg.SpectrumAwgTab",
  runviewer_parser = "user_devices.spectrum_awg.SpectrumAwgViewer"
  # runviewer_parser = "user_devices.awg.SpectrumAwgParser",
)
labscript_devices.register_classes(
  "SpectrumAwgOut"
)
labscript_devices.register_classes(
  "SpectrumAwgOutCopy"
)