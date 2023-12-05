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

#####################################################################
#                                                                   #
# Module written by Alex Tritt under Lincoln Turner, 2023           #
#                                                                   #
#####################################################################

import numpy as np
import math
import spectrum_card as sc
import time as tm
import datetime as dtm
import multiprocessing as mp

import labscript
import labscript_devices
import blacs.tab_base_classes
import blacs.device_base_class
import labscript_utils.h5_lock, h5py

TIME_OUT = 1e-6

#  =============================================================================================================
#    ____        _               _         =====================================================================
#   / __ \      | |             | |        =====================================================================
#  | |  | |_   _| |_ _ __  _   _| |_ ___   =====================================================================
#  | |  | | | | | __| '_ \| | | | __/ __|  =====================================================================
#  | |__| | |_| | |_| |_) | |_| | |_\__ \  =====================================================================
#   \____/ \__,_|\__| .__/ \__,_|\__|___/  =====================================================================
#                   | |                    =====================================================================
#                   |_|                    =====================================================================
#  =============================================================================================================

class SpectrumAwgOut(labscript.Output):
  """
  A :obj:`labscript.Device` representing an individual output port of an AWG.

  PARAMETERS
  ----------
  name : :obj:`str`
    Name of the channel output in labscript and BLACs.
  parent_device : :obj:`SpectrumAwg`
    The AWG card that this output is a channel of.
  connection : :obj:`str`
    A string of the port index of the channel on the card.
    For example, if this represents the channel sent to output port 2 on a 4-port card, write :obj:`"2"`.
  """
  def __init__(self, name, parent_device, connection):
    super().__init__(name, parent_device, connection, default_value = 0)

  def set_wave_and_enable(self, time, wave:np.ndarray):
    """
    Starts a waveform :obj:`wave` playing on the channel at time :obj:`time`.
    This waveform will keep looping until either another :meth:`set_wave_and_enable` or :meth:`disable` command is called at a later time.

    .. note::

      The length of this waveform must be a multiple of :obj:`32`.
      Also, when using multiple channels, you must make sure that each waveform being looped has the same length.

    .. note::

      If you are triggering the card, you will need to trigger the card the first time :meth:`set_wave_and_enable` is called, as well as every time :meth:`set_wave_and_enable` is called after :meth:`disable` is called.
    
    PARAMETERS
    ----------
    time : :obj:`float`
      The time at which to start playing the waveform.
    wave : :obj:`numpy.ndarray`
      The waveform to be looped.
      Samples should be between :obj:`-1.0` and :obj:`1.0`.
      Note that the length of this waveform must be a multiple of :obj:`32`.

    RAISES
    ------
    :obj:`ValueError` :
      If the length of :obj:`wave` is not a multiple of :obj:`32`.
    """
    if wave.size % 32:
      raise ValueError("Waveform length must be a multiple of 32.")
    
    self.parent_device._set_wave_and_enable(int(self.connection), time, wave)
    return time
  
  def loop(self, time, wave:np.ndarray):
    """
    Legacy method.
    """
    self.disable(self, time, wave)

  def disable(self, time):
    """
    Stops the channel (and AWG card) from outputting anything.
    That is, it stops a waveform from looping (that was set started by a :meth:`set_wave_and_enable`)
    It will stay dormant until another :meth:`set_wave_and_enable` command is called at a later time (or call it at the end of a sequence).

    .. warning::

      In the current implementation, calling :meth:`disable` on one channel will disable all channels.

    PARAMETERS
    ----------
    time : :obj:`float`
      The time at which to start playing the waveform.

    """
    self.parent_device._disable(int(self.connection), time)
    return time
  
  def wait(self, time):
    """
    Legacy method.
    """
    self.disable(time)
  
  def init_amplitude(self, amplitude):
    """
    Sets the amplitude of the output.
    The output will vary from + to - this value.
    The waveform samples are proportions of this value that the output should be set to.
    Note that it will stay at this value throughout the whole experiment.
    
    PARAMETERS
    ----------
    amplitude : :obj:`float`
      Amplitude in volts.
      Will be precise to the nearest millivolt.
    """
    self.parent_device._init_amplitude(int(self.connection), amplitude)
  
class SpectrumAwgOutCopy(labscript.Output):
  """
  A :obj:`labscript.Device` representing an individual output port of an AWG.
  This particular output port will copy the behaviour of another port (see double and differential output modes in AWG manual).

  .. note::

    A port can only be copied if it has an even port number, and the port that copies it must have the next port number after that.
    For example, port 3 can only copy port 2.

  PARAMETERS
  ----------
  name : :obj:`str`
    Name of the channel output in labscript and BLACs.
  parent_device : :obj:`SpectrumAwg`
    The AWG card that this output is a channel of.
  connection : :obj:`str`
    A string of the port index of the channel on the card.
    For example, if this represents the channel sent to output port 2 on a 4-port card, write :obj:`"2"`.
  copied_device : :obj:`SpectrumAwgOut`
    The output port that this port is going to copy.
  differential : :obj:`bool`
    If set to :obj:`False` (default), the port will be set to double output mode (copying the output), otherwise the port will be set to differential output mode (mirroring the output).
    See card manual for more details.
  """
  def __init__(self, name, parent_device, connection, copied_device:SpectrumAwgOut, differential = False):
    super().__init__(name, parent_device, connection, default_value = 0)
    self.differential = differential
    self.copied_device = copied_device

    self.parent_device._make_copy(copied_device.connection, connection, differential)

  def init_amplitude(self, amplitude):
    """
    Sets the amplitude of the output.
    The output will vary from + to - this value.
    The waveform samples are proportions of this value that the output should be set to.
    Note that it will stay at this value throughout the whole experiment.
    
    PARAMETERS
    ----------
    amplitude : :obj:`float`
      Amplitude in volts.
      Will be precise to the nearest millivolt.
    """
    self.parent_device._init_amplitude(int(self.connection), amplitude)

#  =============================================================================================================
#   _____                                                           ============================================
#  |  __ \                                                          ============================================
#  | |__) | __ ___   __ _ _ __ __ _ _ __ ___  _ __ ___   ___ _ __   ============================================
#  |  ___/ '__/ _ \ / _` | '__/ _` | '_ ` _ \| '_ ` _ \ / _ \ '__|  ============================================
#  | |   | | | (_) | (_| | | | (_| | | | | | | | | | | |  __/ |     ============================================
#  |_|   |_|  \___/ \__, |_|  \__,_|_| |_| |_|_| |_| |_|\___|_|     ============================================
#                    __/ |                                          ============================================
#                   |___/                                           ============================================
#  =============================================================================================================

class SpectrumAwg(labscript.Device):
  """
  A :obj:`labscript.Device` representing the AWG card itself.

  PARAMETERS
  ----------
  name : :obj:`str`
    Name of the card in labscript and BLACs.
  BLACS_connection : :obj:`str`
    The handle name of the card (as seen in Spectrum Control Center).
    Defaults to :obj:`"/dev/spcm0"`, which should be correct if you are using a single, local card.
  """
  description = "Spectrum AWG"

  trigger_delay =  4.781e-6
  """
  The delay in seconds between the application of a trigger and a response from the card.
  Measured as 4.781(7) us, from 2023-04-13 by Alex Tritt.
  """

  allowed_children = [SpectrumAwgOut, SpectrumAwgOutCopy]
  """
  This device accepts output ports :obj:`SpectrumAwgOut` and copied output ports (that is ports running in double or differential modes) :obj:`SpectrumAwgOutCopy` as children.
  """

  def __init__(self, name, BLACS_connection = "/dev/spcm0", **kwargs):
    self.BLACS_connection = BLACS_connection
    labscript.Device.__init__(self, name, None, connection = BLACS_connection, **kwargs)

    self.sample_rate      = None
    self.instructions     = []
    self.wave_table       = []
    self.differentials    = []
    self.copies           = []
    self.amplitudes       = []
    self.software_trigger = True
    self.trigger_port     = 0
    self.trigger_level    = 0
    self.re_arm_level     = 0

  def generate_code(self, hdf5_file):
    """
    Prepares the code based on commands in experiment file.
    The :obj:`hdf5_file` is structured to contain information for instructions that the card natively understands; namely :obj:`"SEGMENT"` and :obj:`"SEQUENCE"` information.

    The :obj:`"SEGMENT"` key points to a 3-dimensional array containing the waveforms themselves.
    The array is structured :obj:`segments[segment_index, time_index, channel_index]`.
    Each segment contains one waveform per channel.
    That is, only one segment (one combination of waveforms) can be played at once.
    
    The :obj:`"SEGMENT"` array also contains attributes associated with the card waveforms:
    
    * :obj:`"LENGTHS"` is a :obj:`numpy.ndarray` of :obj:`int` of the length of each segment (this allows for zero padding)
    * :obj:`"CONNECTIONS"` is a :obj:`numpy.ndarray` of :obj:`int` of the output ports being used in the sequence (excluding copies).
    * :obj:`"SAMPLE_RATE"` is an :obj:`int` of the sample rate of the card in Hz.
    * :obj:`"DIFFERENTIALS"` is a :obj:`numpy.ndarray` of any ports being copied in differential mode.
    * :obj:`"COPIES"` is a :obj:`numpy.ndarray` of :obj:`int` of any ports being copied in double mode.
    * :obj:`"AMPLITUDES"` is a :obj:`numpy.ndarray` of :obj:`float` of the amplitudes of ports, in volts.
    * :obj:`"AMPLITUDE_INDICES"` is a :obj:`numpy.ndarray` of :obj:`int` of the output ports being used in the sequence (including copies). Amplitudes in :obj:`AMPLITUDES"` correspond to these ports.

    This :obj:`generate_code` method unrolls the playing of simultaneous waveforms into a sequence of segments (waveforms played simultaneously across all channels).
    While the segments are loaded into the :obj:`"SEGMENT"` array, the sequence instructions are saved into a :obj:`"SEQUENCE"` :obj:`h5py.Group`.
    The :obj:`"SEQUENCE"` :obj:`h5py.Group` contains a :obj:`numpy.ndarray` for all the parameters used when programming a sequence on the card.
    Namely,

    * :obj:`"STEP"` :obj:`numpy.ndarray` of :obj:`int`: the step in the sequence.
    * :obj:`"NEXT_STEP"` :obj:`numpy.ndarray` of :obj:`int`: the step that this step will follow on to.
    * :obj:`"SEGMENT"` :obj:`numpy.ndarray` of :obj:`int`: the segment that plays during this step.
    * :obj:`"LOOPS"` :obj:`numpy.ndarray` of :obj:`int`: The number of times the segment should be looped during this step.
    * :obj:`"END_OF_SEQUENCE"` :obj:`numpy.ndarray` of :obj:`bool`: if :obj:`True`, the card exits the sequence and (if appropriate) will prepare for the next one.
    * :obj:`"LOOP_UNTIL_TRIGGER"` :obj:`numpy.ndarray` of :obj:`bool`: currently unused in this implementation.

    The :obj:`"SEQUENCE"` :obj:`h5py.Group` also contains some attributes relating to sequencing and triggers:

    * :obj:`"START_STEPS"`: a :obj:`numpy.ndarray` of :obj:`int that determines the start point of the different sequences. A new sequence is started every time :meth:`SpectrumAwgOut.set_wave_and_enable` is called after a call of :meth:`SpectrumAwgOut.disable`.
    * :obj:`"SOFTWARE_TRIGGER"`: a :obj:`bool` where if :obj:`True`, the card will be triggered automatically (with inconsistent timing) as soon as the shot is loaded in BLACs.
    * :obj:`"TRIGGER_LEVEL"`: a :obj:`float` that determines the voltage on the trigger port that will trigger the card when first reached (that is, triggering happens on a positive edge).
    * :obj:`"RE_ARM_LEVEL"`: the voltage level that the trigger port must reach after a trigger before it is able to accept a second trigger.
    * :obj:`"TRIGGER_PORT"`: the trigger port number for the active trigger.

    PARAMETERS
    ----------
    hdf5_file : :obj:`h5py.File`
      The HDF5 file to be written to.
    """
    labscript.Device.generate_code(self, hdf5_file)
    
    if len(self.instructions) == 0:
      return
    
    if self.sample_rate is None:
      raise Exception(f"Please set the sample rate for {self.name}.")

    self.instructions.sort(key = lambda instruction : instruction["time"])
    # print("Instructions:", self.instructions)

    connections = []
    for instruction in self.instructions:
      if instruction["instruction"] != "wait":
        if instruction["connection"] not in connections:
          connections.append(instruction["connection"])
    if len(connections) == 0:
      return
    connections = np.array(sorted(connections), dtype = int)
    connection_to_channel = {connection:channel for channel, connection in enumerate(connections)}

    interpolated_instructions = []
    current_waves = [None for channel in connections]
    current_time = -1
    for instruction in self.instructions:
      if instruction["instruction"] == "loop":
        current_wave = instruction["wave_index"]
      else:
        current_wave = 0
      if current_time < instruction["time"]:
        interpolated_instructions.append({
          "instruction" : instruction["instruction"],
          "time": instruction["time"],
          "wave_indices" : [current_wave_index for current_wave_index in current_waves],
        })
        current_time = instruction["time"]
      current_waves = interpolated_instructions[-1]["wave_indices"]
      current_waves[connection_to_channel[instruction["connection"]]] = current_wave
    interpolated_wave_table_indices = []
    for instruction in interpolated_instructions:
      waves_index = None
      waves = instruction["wave_indices"]
      for interpolated_wave_index, interpolated_waves in enumerate(interpolated_wave_table_indices):
        if waves == interpolated_waves:
          waves_index = interpolated_wave_index
          break
      if waves_index is None:
        interpolated_wave_table_indices.append(waves)
        waves_index = len(interpolated_wave_table_indices) - 1
      instruction["waves_index"] = waves_index

    signal_segments = [[self.wave_table[wave_index] for wave_index in interpolated_wave_indices] for interpolated_wave_indices in interpolated_wave_table_indices]
    signal_length = max([waves[0].size for waves in signal_segments])
    for waves in signal_segments:
      first_length = waves[0].size
      for wave in waves:
        if wave.size != first_length:
          raise Exception("Waveforms in each channel need to be the same length at any point in time.")

    segments = np.zeros((len(signal_segments), signal_length, len(signal_segments[0])), dtype = np.float32)
    segment_lengths = np.zeros(len(signal_segments), dtype = np.int64)
    for segment_index, signals in enumerate(signal_segments):
      for channel_index, signal in enumerate(signals):
        segments[segment_index, :signal.size, channel_index] = signal
        segment_lengths[segment_index] = signal.size  

    # print("Interpolated instructions:", interpolated_instructions)

    sequence = {
      "step"              : [],
      "next_step"         : [],
      "segment"           : [],
      "loops"             : [],
      "end_of_sequence"   : [],
      "loop_until_trigger": []
    }

    start_steps = []
    started = False
    step_index = 0
    # next_time = interpolated_instructions[0]["time"]
    for instruction_index, instruction in enumerate(interpolated_instructions):
      if instruction["instruction"] == "loop":
        if not started:
          started = True
          start_steps.append(step_index)
          next_time = instruction["time"]
        
        current_time = next_time
        if instruction_index >= len(interpolated_instructions):
          raise Exception("No end condition for channel")
        instruction_next = interpolated_instructions[instruction_index + 1]
        next_time = instruction_next["time"]
        duration = next_time - current_time
        length = self.wave_table[instruction["wave_indices"][0]].size
        number_of_loops = int(duration*self.sample_rate/length)

        next_time = current_time + number_of_loops*length/self.sample_rate

        sequence["step"].append               (step_index)
        sequence["next_step"].append          (step_index + 1)
        sequence["segment"].append            (instruction["waves_index"])
        sequence["loops"].append              (number_of_loops)
        sequence["end_of_sequence"].append    (False)
        sequence["loop_until_trigger"].append (False)

      elif instruction["instruction"] == "wait":
        started = False
        
        sequence["step"].append               (step_index)
        sequence["next_step"].append          (step_index)
        sequence["segment"].append            (0)
        sequence["loops"].append              (1)
        sequence["end_of_sequence"].append    (True)
        sequence["loop_until_trigger"].append (False)

      step_index += 1

    group = self.init_device_group(hdf5_file)

    amplitude_indices = [amplitude[0] for amplitude in self.amplitudes]
    for connection in connections:
      if connection not in amplitude_indices:
        raise Exception(f"No defined amplitude for channel {connection}.")
    group.create_dataset("SEGMENTS", compression=labscript.config.compression, data = segments)
    group["SEGMENTS"].attrs["LENGTHS"]            = segment_lengths
    group["SEGMENTS"].attrs["CONNECTIONS"]        = connections
    group["SEGMENTS"].attrs["SAMPLE_RATE"]        = int(self.sample_rate)
    group["SEGMENTS"].attrs["DIFFERENTIALS"]      = np.array(self.differentials,                              dtype = int)
    group["SEGMENTS"].attrs["COPIES"]             = np.array(self.copies,                                     dtype = int)
    group["SEGMENTS"].attrs["AMPLITUDES"]         = np.array([amplitude[1] for amplitude in self.amplitudes], dtype = float)
    group["SEGMENTS"].attrs["AMPLITUDE_INDICES"]  = np.array([amplitude[0] for amplitude in self.amplitudes], dtype = int)

    sequence_group                        = group.create_group("SEQUENCE")
    sequence_group["STEP"]                = np.array(sequence["step"],                dtype = int)
    sequence_group["NEXT_STEP"]           = np.array(sequence["next_step"],           dtype = int)
    sequence_group["SEGMENT"]             = np.array(sequence["segment"],             dtype = int)
    sequence_group["LOOPS"]               = np.array(sequence["loops"],               dtype = int)
    sequence_group["END_OF_SEQUENCE"]     = np.array(sequence["end_of_sequence"],     dtype = bool)
    sequence_group["LOOP_UNTIL_TRIGGER"]  = np.array(sequence["loop_until_trigger"],  dtype = bool)
    sequence_group.attrs["START_STEPS"]   = np.array(start_steps,                     dtype = int)

    sequence_group.attrs["SOFTWARE_TRIGGER"]  = self.software_trigger
    sequence_group.attrs["TRIGGER_LEVEL"]     = self.trigger_level
    sequence_group.attrs["RE_ARM_LEVEL"]      = self.re_arm_level
    sequence_group.attrs["TRIGGER_PORT"]      = self.trigger_port

  def init_sample_rate(self, sample_rate):
    """
    Sets the sample rate of the AWG.
    Will stay constant for the whole sequence.

    PARAMETERS
    ----------
    sample_rate : :obj:`float`
      The sample rate in Hz.
    """
    self.sample_rate = sample_rate

  def init_trigger(self, port, level, re_arm_level = None):
    """
    Sets triggering setup for the AWG.
    If not called, the card will automatically be software triggered at the start of a shot.

    PARAMETERS
    ----------
    port : :obj:`int`
      The trigger port to be used.
    level : :obj:`float`
      The voltage on the trigger port that will trigger the card when first reached (that is, triggering happens on a positive edge).
    re_arm_level : :obj:`float`
      The voltage level that the trigger port must reach after a trigger before it is able to accept a second trigger.
      If :obj:`None` (default), will be set to :obj:`level/2`.
    """
    if re_arm_level is None:
      re_arm_level = 0.5*level
    self.software_trigger = False
    self.trigger_port     = port
    self.trigger_level    = float(level)
    self.re_arm_level     = float(re_arm_level)

  def _set_wave_and_enable(self, connection, time, wave):
    wave_index = None
    for wave_table_index, wave_table_instance in enumerate(self.wave_table):
      if wave_table_instance is wave:
        wave_index = wave_table_index
        break
    if wave_index is None:
      self.wave_table.append(wave)
      wave_index = len(self.wave_table) - 1
    self.instructions.append(
      {
        "instruction" : "loop",
        "connection"  : connection,
        "time"        : time,
        "wave_index"  : wave_index
      }
    )
  def _disable(self, connection, time):
    self.instructions.append(
      {
        "instruction" : "wait",
        "connection"  : connection,
        "time"        : time
      }
    )

  def _make_copy(self, server, client, differential):
    server = int(server)
    client = int(client)
    if client - server != 1:
      raise ValueError(f"A copy of channel {server} must be placed on channel {server + 1}.")
    
    if differential:
      self.differentials.append(server)
    else:
      self.copies.append(server)

  def _init_amplitude(self, connection, amplitude):
    self.amplitudes.append([connection, amplitude])

#  =============================================================================================================
#   __  __ _     _         __ _ _       _     _                        _               =========================
#  |  \/  (_)   | |       / _| (_)     | |   | |                      | |              =========================
#  | \  / |_  __| |______| |_| |_  __ _| |__ | |_  __      _____  _ __| | _____ _ __   =========================
#  | |\/| | |/ _` |______|  _| | |/ _` | '_ \| __| \ \ /\ / / _ \| '__| |/ / _ \ '__|  =========================
#  | |  | | | (_| |      | | | | | (_| | | | | |_   \ V  V / (_) | |  |   <  __/ |     =========================
#  |_|  |_|_|\__,_|      |_| |_|_|\__, |_| |_|\__|   \_/\_/ \___/|_|  |_|\_\___|_|     =========================
#                                  __/ |                                               =========================
#                                 |___/                                                =========================
#  =============================================================================================================

class SpectrumAwgWorkerMidFlight(mp.Process):
  """
  A :obj:`multiprocessing.Process` that handles programming the AWG when going into buffered mode, as well as rearming the card before any retriggering occurs.

  PARAMETERS
  ----------
  done_queue : :obj:`multiprocessing.Queue`
    A :obj:`multiprocessing.Queue` that posts status messages to the main BLACs worker, :obj:`SpectrumAwgWorker`.
    :obj:`SpectrumAwgWorker` will allow the shot to run once the final status message has been received by it.
  manual_queue : :obj:`multiprocessing.Queue`
    A :obj:`multiprocessing.Queue` that receives pseudo "interrupt request" messages from the main BLACs worker, :obj:`SpectrumAwgWorker`.
    If a message is received, this thread will shut down the :obj:`spectrum_card.Card` instance safely as to not block it from being used by the main worker thread.
  h5file : :obj:`str`
    The path to the :obj:`h5py.File` which contains the instructions written by :obj:`SpectrumAwg` to be programmed to the card.
  device_name : :obj:`str`
    The name of the card, as written in the :obj:`h5py.File`.
  address : :obj:`str`
    The handle name of the card (as seen in Spectrum Control Center).
  """

  wait_time = 1e-6
  """
  Polling time in seconds for checking card status during the shot.
  """

  def __init__(self, done_queue:mp.Queue, manual_queue:mp.Queue, h5file:str, device_name:str, address:str):
    self.done_queue   = done_queue
    self.manual_queue = manual_queue
    self.h5file       = h5file
    self.address      = address
    self.device_name  = device_name
    super().__init__()

  def _print(self, *arguments, **keyword_arguments):
    """
    Sends messages to be printed out by the :obj:`SpectrumAwgWorker` thread.
    Uses the same arguments as the usual :obj:`print` function.
    """
    self.done_queue.put(([*arguments], {**keyword_arguments}))
  
  def run(self):
    """
    Programs the AWG when going into buffered mode, as well as rearms the card before any retriggering occurs.
    """

    # Wait to get control of the AWG from any other thread using it.
    self.card = None
    while self.card is None:
      try:
        self.card = sc.Card(bytes(self.address, 'utf-8'))
        self.card.get_max_sample_rate()
      except Exception:
        self.card.close()
        self.card = None
        tm.sleep(TIME_OUT)
    
    # The following code is designed to safely close the card at any issues or requests to do so.
    try:
      self._print("  Transferred card from main worker to mid-flight worker.")

      # Program the card, while printing status.
      self._print(f"  Reading from HDF5...      ", end = " ")
      data              = h5py.File(self.h5file, "r")
      group             = data[f"devices/{self.device_name}"]
      if "SEGMENTS" not in group:
        data.close()
        self._print("Done!")
        self._print("  No instructions found.")
        self._print("Release")
        raise Exception
      segments          = np.asarray(group["SEGMENTS"])
      segment_lengths   = np.asarray(group["SEGMENTS"].attrs["LENGTHS"])
      connections       = np.asarray(group["SEGMENTS"].attrs["CONNECTIONS"])
      sample_rate       = np.asarray(group["SEGMENTS"].attrs["SAMPLE_RATE"])
      copies            = np.asarray(group["SEGMENTS"].attrs["COPIES"])
      differentials     = np.asarray(group["SEGMENTS"].attrs["DIFFERENTIALS"])
      amplitudes        = np.asarray(group["SEGMENTS"].attrs["AMPLITUDES"])
      amplitude_indices = np.asarray(group["SEGMENTS"].attrs["AMPLITUDE_INDICES"])
      number_of_channels = connections.size

      sequence_group    = group["SEQUENCE"]
      start_steps       = np.asarray(sequence_group.attrs["START_STEPS"])
      software_trigger  = sequence_group.attrs["SOFTWARE_TRIGGER"]
      trigger_port      = int(sequence_group.attrs["TRIGGER_PORT"])
      trigger_level     = int(sequence_group.attrs["TRIGGER_LEVEL"])
      re_arm_level      = int(sequence_group.attrs["RE_ARM_LEVEL"])
      sequence          = {
        "step":               np.asarray(sequence_group["STEP"]),
        "next_step":          np.asarray(sequence_group["NEXT_STEP"]),
        "segment":            np.asarray(sequence_group["SEGMENT"]),
        "loops":              np.asarray(sequence_group["LOOPS"]),
        "end_of_sequence":    np.asarray(sequence_group["END_OF_SEQUENCE"]),
        "loop_until_trigger": np.asarray(sequence_group["LOOP_UNTIL_TRIGGER"])
      }
      data.close()
      self._print("Done!")

      self._print(f"  Setting card parameters...", end = " ")
      self.card.stop()
      self.card.reset()
      self.card.use_mode_sequence()

      self.card.set_sample_rate(sample_rate)
      for copy in copies:
        self.card.double_enable(int(copy))
      for differential in differentials:
        self.card.differential_enable(int(differential))

      number_of_channels = segments.shape[2]
      number_of_segments = segment_lengths.size
      number_of_segments_round = 2**(math.ceil(math.log2(number_of_segments + 1)))
      self._print(f"\n    Number of segments (rounded): {number_of_segments_round}")

      self.card.set_number_of_segments(number_of_segments_round)
      self.card.set_memory_size(number_of_segments_round*segment_lengths.max())
      self._print(f"    Memory size: {number_of_segments_round*segment_lengths.max()}")

      self.card.set_channels_enable(**{f"channel_{channel_index}":True for channel_index in connections})
      self._print(f"    Using channels: {self.card.get_channels_enable()}")
      for channel_index in connections:
        for amplitude_index, amplitude in zip(amplitude_indices, amplitudes):
          if amplitude_index == channel_index:
            self.card.output_enable(int(channel_index))
            self.card.set_amplitude(int(channel_index), amplitude)
          elif amplitude_index == channel_index + 1:
            if (channel_index in copies) or (channel_index in differentials):
              self.card.output_enable(int(channel_index) + 1)
              self.card.set_amplitude(int(channel_index) + 1, amplitude)
      self._print("  Done!")

      self._print(f"  Transferring waveforms...  0%", end = "\r")
      try:
        for segment_index in range(number_of_segments):
          signals = []
          segment_length = segment_lengths[segment_index]
          # self._print(f"\n  {number_of_channels}\n")
          for channel_index in range(number_of_channels):
            signals.append(segments[segment_index, :segment_length, channel_index])
          # self._print(f"  {signals}")
          self.card.array_to_device(signals, segment_index)
          self._print(f"  Transferring waveforms...  {(segment_index + 1)*100//number_of_segments}%", end = "\r")
        self._print(f"  Transferring waveforms...  Done!")
      except Exception as e:
        # self._print(f"\n  Error :(\n  {e}")
        import traceback
        self._print(traceback.format_exc())
        self._print("Release")
        raise Exception

      self._print(f"  Transferring sequence...   0%", end = "\r")
      try:
        sequence_index = 0
        sequence_size = len(sequence["step"])
        for step, next_step, segment, loops, end_of_sequence, loop_until_trigger in zip(sequence["step"], sequence["next_step"], sequence["segment"], sequence["loops"], sequence["end_of_sequence"], sequence["loop_until_trigger"]):
          self.card.set_step_instruction(int(step), int(segment), int(loops), int(next_step), bool(loop_until_trigger), bool(end_of_sequence))
          sequence_index += 1
          self._print(f"  Transferring sequence...   {(sequence_index + 1)*100//sequence_size}%", end = "\r")
        self._print(f"  Transferring sequence...   Done!")
      except Exception:
        self._print("  Error :(")
        self._print("Release")
        raise Exception

      self.card.set_start_step(start_steps[0])
      if not software_trigger:
        self.card.trigger_coupling_use_dc(trigger_port)
        self.card.use_trigger_positive_edge(trigger_port, trigger_level, re_arm_threshold = re_arm_level)
        self.card.set_sufficient_triggers(**{f"external_{trigger_port}":True})
        self.card.arm()
        self._print("  Card armed.")
      
      if software_trigger:
        self.card.arm()
        self.card.force_trigger()
        self._print("  Card software triggered.")

      # End of card programming, let the main worker know that the shot can start.
      self._print("Release")

      # Poll card status during the shot.
      ready = True
      for start_step in start_steps[1:]:
        # Wait until the card starts running before telling it where the start of the next sequence is.
        while ready:
          # Check for a request to stop the sequence.
          if self.manual_queue.qsize() > 0:
            raise Exception()
          
          ready = "Ready" in self.card.get_status_information()
          tm.sleep(self.wait_time)

        # Tell the card where the start of the next sequence is.
        self.card.set_start_step(start_step)

        # Wait until the card stops running before re-arming it.
        while not ready:
          # Check for a request to stop the sequence.
          if self.manual_queue.qsize() > 0:
            raise Exception()
          
          ready = "Ready" in self.card.get_status_information()
          tm.sleep(self.wait_time)
        self.card.arm()

      # Keep the thread running until it is requested to stop. Otherwise the card will be disabled and that's no good.
      self.manual_queue.get()
      
    except Exception:
      self.manual_queue.get()
    self.card.close()

#  =============================================================================================================
#  __          __        _               =======================================================================
#  \ \        / /       | |              =======================================================================
#   \ \  /\  / /__  _ __| | _____ _ __   =======================================================================
#    \ \/  \/ / _ \| '__| |/ / _ \ '__|  =======================================================================
#     \  /\  / (_) | |  |   <  __/ |     =======================================================================
#      \/  \/ \___/|_|  |_|\_\___|_|     =======================================================================
#                                        =======================================================================
#                                        =======================================================================
#  =============================================================================================================

class SpectrumAwgWorker(blacs.tab_base_classes.Worker):
  """
  The main worker thread for the Spectrum AWG.
  It implements the front-panel control of the :obj:`SpectrumAwgTab`, as well as handles moving to the buffered mode by hand-passing control of the card to a :obj:`SpectrumAwgWorkerMidFlight`.
  """
  
  amplitude_min   = 80e-3
  """
  The smallest acceptable amplitude in volts for any AWG channel.
  """

  sample_rate_min = 50
  """
  The smallest acceptable sample rate in MHz for the card.
  """

  def init(self):
    """
    Initialises the manual front-panel mode for the :obj:`SpectrumAwgTab`.
    """

    # Wait to get control of the AWG from any other thread using it.
    self.card = None
    while self.card is None:
      try:
        self.card = sc.Card(bytes(self.BLACS_connection, 'utf-8'))
        self.card.get_max_sample_rate()
      except Exception:
        self.card.close()
        self.card = None
        tm.sleep(TIME_OUT)

    # Initialise card.
    self.card.stop()
    self.card.reset()
    self.card_name          = self.card.get_name()
    self.number_of_channels = self.card.get_number_of_channels_per_front_end_module()
    time_string             = dtm.datetime.now().strftime("%Y%m%dT%H%M%S")

    # Print card information.
    print(f"{time_string}> Initialised AWG card:")
    print(f"  Model: {self.card_name}")
    print(f"  Channels: {self.number_of_channels}")
    print(f"  Memory: {self.card.get_max_memory_size()/(2**30)} GiB")

    # Set up front panel synthesiser mode for the SpectrumAwgTab.
    self.front_panel_previous = {}
    self.manual_waveform    = ["" for channel_index in range(self.number_of_channels)]
    self.manual_waveforms   = ["Sine", "Square", "Sawtooth"]
    self.manual_frequencies = [0 for channel_index in range(self.number_of_channels)]
    self.segment_size       = 512
    self.signals            = [None for channel_index in range(self.number_of_channels)]

    # Set up multiprocessing for programming the card during a shot.
    self.worker_mid_flight = None
    self.manual_queue = mp.Queue()

  def program_manual(self, front_panel_values):
    """
    Respond to user input from the :obj:`SpectrumAwgTab` GUI.

    PARAMETERS
    ----------
    front_panel_values : :obj:`dict`
      The current state of the GUI.
    """

    # Handle a change in sample rate or segment size, including resetting the waveforms.
    do_reset = False
    for key, value in front_panel_values.items():
      # Only look for changes in front panel
      if key in self.front_panel_previous:
        if self.front_panel_previous[key] == value:
          continue
      
      if key == "Sample rate":
        self.card.stop()
        if value < self.sample_rate_min:
          value = self.sample_rate_min
        self.card.set_sample_rate(value, "M")
        do_reset = True
      if key == "Segment size":
        self.card.stop()
        self.segment_size = max(int(round(value/32)*32), 32)
        self.card.set_memory_size(self.segment_size)
        front_panel_values[key] = self.segment_size
        do_reset = True
      for channel_index in range(self.number_of_channels):
        if key == f"Frequency:{channel_index}":
          self.manual_frequencies[channel_index] = value
          do_reset = True
    if do_reset:
      for channel_index in range(self.number_of_channels):
        self.manual_waveform[channel_index] = ""
        self.signals = [None for channel_index in range(self.number_of_channels)]
    
    # Make sure the waveform buttons act as radio buttons.
    waveform_decided = [False for channel_index in range(self.number_of_channels)]
    for key, value in front_panel_values.items():
      key_split = key.split(":")
      if len(key_split) < 2:
        continue
      key_base = key_split[0]
      key_channel_index = int(key_split[1])
      if key_base in self.manual_waveforms:
        if value:
          waveform_decided[key_channel_index] = True
    for channel_index in range(self.number_of_channels):
      if not waveform_decided[channel_index]:
        if self.manual_waveform[channel_index] in self.manual_waveforms:
          front_panel_values[f"{self.manual_waveform[channel_index]}:{channel_index}"] = True
        else:
          front_panel_values[f"Sine:{channel_index}"] = True
          self.manual_waveform[channel_index] = "Sine"

    # Handle a change in channel waveforms.
    for key, value in front_panel_values.items():
      key_split = key.split(":")
      if len(key_split) < 2:
        continue
      key_base = key_split[0]
      key_channel_index = int(key_split[1])
      if key_base in self.manual_waveforms:
        if value and self.manual_waveform[key_channel_index] != key_base:
          self.manual_waveform[key_channel_index] = key_base
          for waveform in self.manual_waveforms:
            if waveform != key_base:
              front_panel_values[f"{waveform}:{key_channel_index}"] = False

          signal_length = self.segment_size
          sample_rate   = self.card.get_sample_rate()
          time_end      = signal_length/sample_rate
          self.manual_frequencies[key_channel_index] = np.round(self.manual_frequencies[key_channel_index]*1e6*time_end)/(time_end*1e6)
          front_panel_values[f"Frequency:{key_channel_index}"] = self.manual_frequencies[key_channel_index]
          phase = self.manual_frequencies[key_channel_index]*1e6*math.tau*np.arange(0, signal_length)/sample_rate
          if self.manual_waveform[key_channel_index] == "Sine":
            self.signals[key_channel_index] = np.sin(phase)
          if self.manual_waveform[key_channel_index] == "Square":
            self.signals[key_channel_index] = np.sign(np.sin(phase))/math.sqrt(2)
          if self.manual_waveform[key_channel_index] == "Sawtooth":
            self.signals[key_channel_index] = 2*np.fmod(phase/math.tau, 1) - 1

          signals_are_defined = True
          for signal in self.signals:
            if signal is None:
              signals_are_defined = False
          if signals_are_defined:
            self.card.stop()
            self.card.set_channels_enable(**{f"channel_{channel_index}":True for channel_index in range(self.number_of_channels)})
            self.card.start()
            self.card.stop()

            self.card.set_memory_size(signal_length)
            self.card.array_to_device(self.signals)
            self.manual_waveform[key_channel_index] = key_base

            if front_panel_values["Enable"]:
              self.card.arm()
              self.card.force_trigger()
            for channel_index in range(self.number_of_channels):
              if front_panel_values[f"Output enable:{channel_index}"]:
                self.card.output_enable(channel_index)
    
    # Handle a change in card properties.
    for key, value in front_panel_values.items():
      valid_keys = ["Identify", "Enable"]
      for channel_index in range(self.number_of_channels):
        valid_keys += [f"Amplitude:{channel_index}", f"Output enable:{channel_index}"]
      if key not in valid_keys:
        continue
      if key in self.front_panel_previous:
        if self.front_panel_previous[key] == value:
          continue
      if key == "Identify":
        self.card.set_card_identification(int(value))
      if key == "Enable":
        if value:
          if "Ready" in self.card.get_status_information():
            self.card.arm()
            self.card.force_trigger()
        else:
          self.card.stop()
          self.card.execute_commands(disable_trigger = True)
      
      # Handle a change in channel properties.
      for channel_index in range(self.number_of_channels):
        if key == f"Amplitude:{channel_index}":
          if value < self.amplitude_min:
            value = self.amplitude_min
          self.card.set_amplitude(channel_index, value)
        if key == f"Output enable:{channel_index}":
          if value:
            self.card.output_enable(channel_index)
          else:
            self.card.output_disable(channel_index)
    
    self.front_panel_previous = front_panel_values.copy()
    return front_panel_values 

  def transition_to_buffered(self, device_name, h5file, initial_values, fresh):
    """
    Starts a thread of :obj:`SpectrumAwgWorkerMidFlight` which programs the AWG and handles in mid-shot.

    PARAMETERS
    ----------
    device_name : :obj:`str`
      Name of device in :obj:`labscript`.
    h5file : :obj:`str`
      Path to the :obj:`h5py.File` generated by :meth:`SpectrumAwg.generate_code`.
    initial_values :
      Unused.
    fresh :
      Unused.
    """
    time_string = dtm.datetime.now().strftime("%Y%m%dT%H%M%S")
    print(f"{time_string}> Preparing for shot:")
    self.card.close()
    
    # Create thread to handle card mid-shot.
    done_queue = mp.Queue()
    self.worker_mid_flight = SpectrumAwgWorkerMidFlight(done_queue, self.manual_queue, h5file, device_name, self.BLACS_connection)
    self.worker_mid_flight.start()

    # Print out any messages from the thread, and end this method when it calls "Release".
    while True:
      arguments, keyword_arguments = done_queue.get()
      if "Release" in arguments:
        break
      print(*arguments, **keyword_arguments)

    return initial_values

  def transition_to_manual(self, abort = False):
    """
    Ends the shot and takes back control of the AWG into this thread.

    PARAMETERS
    ----------
    abort :
      Unused.
    """

    # Kill worker_mid_flight thread.
    if self.worker_mid_flight is not None:
      self.manual_queue.put("Manual")
      print("  Waiting for mid-flight worker...")
      self.worker_mid_flight.join()
      print("  Transferred card from mid-flight worker to main worker.")
      self.worker_mid_flight = None

    # Take control of AWG.
    self.card = None
    while self.card is None:
      try:
        self.card = sc.Card(bytes(self.BLACS_connection, 'utf-8'))
        self.card.get_max_sample_rate()
      except Exception:
        self.card.close()
        self.card = None
        tm.sleep(TIME_OUT)

    # Reset AWG for manual control.
    self.card.stop()
    self.card.reset()
    self.card.use_mode_single()
    front_panel_reset = self.front_panel_previous
    self.front_panel_previous = {}
    self.program_manual(front_panel_reset)
    return True

  def abort_transition_to_buffered(self):
    """
    Calls :obj:`transition_to_manual`.
    """
    return self.transition_to_manual(True)
      
  def abort_buffered(self):
    """
    Calls :obj:`transition_to_manual`.
    """
    return self.transition_to_manual(True)

  def shutdown(self):
    """
    If in buffered mode, safely kills the :obj:`SpectrumAwgWorkerMidFlight` thread.
    If in manual mode, closes the card handle.
    """
    if self.worker_mid_flight is not None:
      self.manual_queue.put("Manual")
      self.worker_mid_flight.join()
      self.worker_mid_flight = None
    else:
      self.card.identification_led_disable()
      self.card.reset()
      self.card.close()

  def check_remote_values(self):
    """
    Polls the card and sees if its status matches that of the :obj:`SpectrumAwgTab`.

    RETURNS
    -------
    remote_values : :obj:`dict`
      Current status of the card.
    """
    remote_values = {
      "Identify"    : self.card.get_card_identification() != 0,
      "Enable"      : "Ready" not in self.card.get_status_information(),
      "Sample rate" : self.card.get_sample_rate()/1e6,
    }

    if remote_values["Sample rate"] < self.sample_rate_min:
      remote_values["Sample rate"] = self.sample_rate_min

    for channel_index in range(self.number_of_channels):
      remote_values[f"Output enable:{channel_index}"] = self.card.get_output_enable(channel_index) != 0
      remote_values[f"Amplitude:{channel_index}"]     = self.card.get_amplitude(channel_index)
      if remote_values[f"Amplitude:{channel_index}"] < self.amplitude_min:
        remote_values[f"Amplitude:{channel_index}"] = self.amplitude_min
    return remote_values

#  =============================================================================================================
#   ____  _                   _____ _    _ _____   =============================================================
#  |  _ \| |                 / ____| |  | |_   _|  =============================================================
#  | |_) | | __ _  ___ ___  | |  __| |  | | | |    =============================================================
#  |  _ <| |/ _` |/ __/ __| | | |_ | |  | | | |    =============================================================
#  | |_) | | (_| | (__\__ \ | |__| | |__| |_| |_   =============================================================
#  |____/|_|\__,_|\___|___/  \_____|\____/|_____|  =============================================================
#                                                  =============================================================
#                                                  =============================================================
#  =============================================================================================================

@labscript_devices.BLACS_tab
class SpectrumAwgTab(blacs.device_base_class.DeviceTab):
  """
  A GUI for some simple functionality of the AWG.
  The purpose of this is to be able to test to see if the card is working, rather than to actually run an experiment.

  The top panel controls the card itself:

  * :obj:`"Enable"` starts and stops a loop from running.
  * :obj:`"Identify"` toggles the flashing identification mode of the status LED.
  * :obj:`"Sample rate"` sets the sample rate of the card in MHz.
  * :obj:`"Segment size"` sets the size of the waveforms being played in manual mode in samples.

  The subsequent panels control properties of individual channels:

  * :obj:`"Amplitude"` sets the peak value of the waveform in volts.
  * :obj:`"Frequency"` sets the frequency of the waveform in MHz.
    Note that the frequency will be automatically changed so that a whole number of periods fit within the whole waveform.
  * :obj:`"Output enable"` enables a specific chanel.
  * :obj:`"Sawtooth"`, :obj:`"Square"` and :obj:`"Sine"` is a selection between three waveforms that the chanel can output.

  """
  def initialise_GUI(self):
    """
    Creates the GUI for the card.
    Does some tricks to get correct device names etc.
    """

    card = None
    while card is None:
      try:
        card = sc.Card(bytes(self.BLACS_connection, 'utf-8'))
        card.get_max_sample_rate()
      except Exception:
        card.close()
        card = None
        tm.sleep(TIME_OUT)
    sample_rate_max     = card.get_max_sample_rate()
    memory_size_max     = card.get_max_memory_size()
    number_of_channels  = card.get_number_of_channels_per_front_end_module()
    card.close()

    self.supports_remote_value_check(True)
    self.create_worker("worker", SpectrumAwgWorker, {"BLACS_connection":self.BLACS_connection})
    self.primary_worker = "worker"

    card_digital_io = {"Identify":{}, "Enable":{}}
    channel_digital_io = [{f"Output enable:{channel_index}":{}, f"Sine:{channel_index}":{}, f"Square:{channel_index}":{}, f"Sawtooth:{channel_index}":{}} for channel_index in range(number_of_channels)]
    digital_io = {**card_digital_io}
    for channel_index in range(number_of_channels):
      digital_io = {**digital_io, **channel_digital_io[channel_index]}
    self.create_digital_outputs(digital_io)
    digital_widgets = self.create_digital_widgets(digital_io)
    card_digital_widgets = {}
    channel_digital_widgets = []
    for channel_index in range(number_of_channels):
      channel_digital_widgets.append({})
    for key, value in digital_widgets.items():
      if key in card_digital_io:
        card_digital_widgets[key] = value
      for channel_index in range(number_of_channels):
        if key in channel_digital_io[channel_index]:
          channel_digital_widgets[channel_index][key] = value

    card_analog_io = {
      "Sample rate" : {
        "base_unit" : "MS/s",
        "min"       : 50,
        "max"       : sample_rate_max/1e6,
        "step"      : 1/1e6,
        "decimals"  : 6
      },
      "Segment size" : {
        "base_unit" : "Samples",
        "min"       : 32,
        "max"       : memory_size_max,
        "step"      : 32,
        "decimals"  : 0
      },
    }
    channel_analog_io = []
    for channel_index in range(number_of_channels):
      channel_analog_io.append(
        {
          f"Amplitude:{channel_index}":{
            "base_unit"   : "V",
            "min"       : 80e-3,
            "max"       : 2.5,
            "step"      : 1/1e3,
            "decimals"  : 3
          },
          f"Frequency:{channel_index}":{
            "base_unit"   : "MHz",
            "min"       : 0,
            "max"       : 312.5,
            "step"      : 1/1e3,
            "decimals"  : 3
          }
        }
      )

    analog_io = {**card_analog_io}
    for channel_index in range(number_of_channels):
      analog_io = {**analog_io, **channel_analog_io[channel_index]}
    self.create_analog_outputs(analog_io)
    analog_widgets = self.create_analog_widgets(analog_io)
    card_analog_widgets = {}
    channel_analog_widgets = []
    for channel_index in range(number_of_channels):
      channel_analog_widgets.append({})
    for key, value in analog_widgets.items():
      if key in card_analog_io:
        card_analog_widgets[key] = value
      for channel_index in range(number_of_channels):
        if key in channel_analog_io[channel_index]:
          channel_analog_widgets[channel_index][key] = value
    
    card_widgets = {**card_digital_widgets, **card_analog_widgets}
    channel_widgets = []
    for channel_index in range(number_of_channels):
      channel_widgets.append({**(channel_digital_widgets[channel_index]), **(channel_analog_widgets[channel_index])})

    block_names = [f"Channel {channel_index}" for channel_index in range(number_of_channels)]
    for channel_index in range(number_of_channels):
      block_names[channel_index] = None
      try:
        child = self.get_child_from_connection_table(self.device_name, f"{channel_index}")
        if child is not None:
          block_names[channel_index] = child.name
          
          if child.device_class == "SpectrumAwgOutCopy":
            server = self.get_child_from_connection_table(self.device_name, f"{channel_index - 1}")
            if server is not None:
              block_names[channel_index] = f"{child.name} | Buffer mode: Copy of {server.name}."
      except Exception:
        pass
    widgets_channels = []
    for block_name, channel_index in zip(block_names, range(number_of_channels)):
      if block_name is not None:
        widgets_channels.append((block_name, channel_widgets[channel_index])) 
    widgets = [(self.device_name, card_widgets)] + widgets_channels
    self.auto_place_widgets(*widgets)

  def restart(self, *args):
    """
    Makes sure that the :obj:`SpectrumAwgWorkerMidFlight` is killed when the refresh button is pressed.
    Otherwise BLACs freezes.
    """
    self.workers["worker"][1].put(("shutdown", [], {}))
    return super().restart(*args)
  
#  =============================================================================================================
#  __      ___                          ========================================================================
#  \ \    / (_)                         ========================================================================
#   \ \  / / _  _____      _____ _ __   ========================================================================
#    \ \/ / | |/ _ \ \ /\ / / _ \ '__|  ========================================================================
#     \  /  | |  __/\ V  V /  __/ |     ========================================================================
#      \/   |_|\___| \_/\_/ \___|_|     ========================================================================
#                                       ========================================================================
#                                       ========================================================================
#  =============================================================================================================

@labscript_devices.runviewer_parser
class SpectrumAwgViewer(object):
  """
  Currently doesn't do anything.
  Sampling the AWG sequence at full resolution is very memory intensive.
  I have written and commented out some code that subsamples the AWG waveform, although this too may get large fast for a sufficiently complicated sequence.

  .. warning::

    Uncomment at own risk!

    
  """
  def __init__(self, path, device):
    self.path = path
    self.name = device.name
    self.device = device

  # def get_traces(self, add_trace, clock=None):
  #   data              = h5py.File(self.path, "r")
  #   group             = data[f"devices/{self.name}"]
  #   segments          = np.asarray(group["SEGMENTS"])
  #   segment_lengths   = np.asarray(group["SEGMENTS"].attrs["LENGTHS"])
  #   connections       = np.asarray(group["SEGMENTS"].attrs["CONNECTIONS"])
  #   sample_rate       = np.asarray(group["SEGMENTS"].attrs["SAMPLE_RATE"])
  #   copies            = np.asarray(group["SEGMENTS"].attrs["COPIES"])
  #   differentials     = np.asarray(group["SEGMENTS"].attrs["DIFFERENTIALS"])
  #   amplitudes        = np.asarray(group["SEGMENTS"].attrs["AMPLITUDES"])
  #   amplitude_indices = np.asarray(group["SEGMENTS"].attrs["AMPLITUDE_INDICES"])
  #   sequence_group    = group["SEQUENCE"]
  #   start_steps       = np.asarray(sequence_group.attrs["START_STEPS"])
  #   software_trigger  = sequence_group.attrs["SOFTWARE_TRIGGER"]
  #   trigger_port      = int(sequence_group.attrs["TRIGGER_PORT"])
  #   trigger_level     = int(sequence_group.attrs["TRIGGER_LEVEL"])
  #   re_arm_level      = int(sequence_group.attrs["RE_ARM_LEVEL"])
  #   sequence          = {
  #     "step":               np.asarray(sequence_group["STEP"]),
  #     "next_step":          np.asarray(sequence_group["NEXT_STEP"]),
  #     "segment":            np.asarray(sequence_group["SEGMENT"]),
  #     "loops":              np.asarray(sequence_group["LOOPS"]),
  #     "end_of_sequence":    np.asarray(sequence_group["END_OF_SEQUENCE"]),
  #     "loop_until_trigger": np.asarray(sequence_group["LOOP_UNTIL_TRIGGER"])
  #   }
  #   data.close()

  #   stride = 64
  #   time = np.arange(0, 15, stride/sample_rate)#1/sample_rate)
  #   signals = [np.zeros_like(time) + channel_index for channel_index in connections]
  #   signal_names = [f"{self.name}: Channel {channel_index}" for channel_index in connections]
  #   # signal_amplitudes = []
  #   # for channel_index in connections:
  #   #   for amplitude_index, amplitude in zip(amplitude_indices, amplitudes):
  #   #     if amplitude_index == channel_index:
  #   #       self.card.output_enable(int(channel_index))
  #   #       self.card.set_amplitude(int(channel_index), amplitude)
  #   #     elif amplitude_index == channel_index + 1:
  #   #       if (channel_index in copies) or (channel_index in differentials):
  #   #         self.card.output_enable(int(channel_index) + 1)
  #   #         self.card.set_amplitude(int(channel_index) + 1, amplitude)

  #   sequence_index  = 0
  #   time_index      = 0
  #   sequence_size   = len(sequence["step"])
  #   for step, next_step, segment, loops, end_of_sequence, loop_until_trigger in zip(sequence["step"], sequence["next_step"], sequence["segment"], sequence["loops"], sequence["end_of_sequence"], sequence["loop_until_trigger"]):
  #     step_size = int(segment_lengths[segment])//stride
  #     # self.card.set_step_instruction(int(step), int(segment), int(loops), int(next_step), bool(loop_until_trigger), bool(end_of_sequence))
  #     for loop in range(loops):
  #       for channel_index, signal in enumerate(signals):
  #         signal[time_index:(time_index + step_size)] = segments[segment, ::stride, channel_index]
  #       time_index += step_size
  #     sequence_index += 1

  #   # time = np.arange(0, 10, 0.001)
  #   # clock = (time, np.sin(math.tau*time))
  #   # add_trace("AWG clock", clock, self.name, "0")
  #   # traces = {"AWG clock":clock}

  #   clocks = [(time, signal) for signal in signals]
  #   traces = {}
  #   done = False
  #   for clock, signal_name, connection in zip(clocks, signal_names, connections):
  #     if done:
  #       add_trace(signal_name, clock, self.name, f"{connection}")
  #       traces[signal_name] = clock
  #     done = True
  #   return traces