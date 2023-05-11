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

from collections.abc import Callable, Iterable, Mapping
from typing import Any
import numpy as np
import math
import spectrum_card as sc
import time as tm
import datetime as dtm
import multiprocessing as mp

import qtutils.qt.QtWidgets as qw
import qtutils.qt.QtGui as qg

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
  def __init__(self, name, parent_device, connection):
    super().__init__(name, parent_device, connection, default_value = 0)

  def loop(self, time, wave:np.ndarray):
    if wave.size % 32:
      raise Exception("Waveform length must be a multiple of 32.")
    
    self.parent_device.loop(int(self.connection), time, wave)
    return time

  def wait(self, time):
    self.parent_device.wait(int(self.connection), time)
    return time
  
  def init_amplitude(self, amplitude):
    self.parent_device.init_amplitude(int(self.connection), amplitude)
  
class SpectrumAwgOutCopy(labscript.Output):
  def __init__(self, name, parent_device, connection, copied_device:SpectrumAwgOut, differential = False):
    super().__init__(name, parent_device, connection, default_value = 0)
    self.differential = differential
    self.copied_device = copied_device

    self.parent_device.make_copy(copied_device.connection, connection, differential)

  def init_amplitude(self, amplitude):
    self.parent_device.init_amplitude(int(self.connection), amplitude)

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
  description = "Spectrum AWG"
  clock_limit = 1e6
  trigger_delay =  4.781e-6#(7)s, from 2023-04-13 Alex Tritt

  allowed_children = [SpectrumAwgOut, SpectrumAwgOutCopy]

  def __init__(self, name, parent_device, BLACS_connection = "/dev/spcm0", **kwargs):
    self.BLACS_connection = BLACS_connection
    labscript.Device.__init__(self, name, None, **kwargs)

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
    labscript.Device.generate_code(self, hdf5_file)
    
    if len(self.instructions) == 0:
      return
    
    if self.sample_rate is None:
      raise Exception(f"Please set the sample rate for {self.name}.")

    self.instructions.sort(key = lambda instruction : instruction["time"])
    # print("Instructions:", self.instructions)

    connections = []
    for instruction in self.instructions:
      if instruction["connection"] not in connections:
        connections.append(instruction["connection"])
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

  def loop(self, connection, time, wave):
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
  def wait(self, connection, time):
    self.instructions.append(
      {
        "instruction" : "wait",
        "connection"  : connection,
        "time"        : time
      }
    )

  def make_copy(self, server, client, differential):
    server = int(server)
    client = int(client)
    if client - server != 1:
      raise Exception(f"A copy of channel {server} must be placed on channel {server + 1}.")
    
    if differential:
      self.differentials.append(server)
    else:
      self.copies.append(server)

  def init_sample_rate(self, sample_rate):
    self.sample_rate = sample_rate

  def init_amplitude(self, connection, amplitude):
    self.amplitudes.append([connection, amplitude])

  def init_trigger(self, port, level, re_arm_level = None):
    if re_arm_level is None:
      re_arm_level = 0.5*level
    self.software_trigger = False
    self.trigger_port     = port
    self.trigger_level    = float(level)
    self.re_arm_level     = float(re_arm_level)

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
  wait_time = 1e-6
  def __init__(self, done_queue:mp.Queue, manual_queue:mp.Queue, h5file:str, device_name:str, address:str):
    self.done_queue   = done_queue
    self.manual_queue = manual_queue
    self.h5file       = h5file
    self.address      = address
    self.device_name  = device_name
    super().__init__()

  def print(self, *arguments, **keyword_arguments):
    # print(*arguments, **keyword_arguments)
    self.done_queue.put(([*arguments], {**keyword_arguments}))
  
  def run(self):
    self.card = None
    while self.card is None:
      try:
        self.card = sc.Card(bytes(self.address, 'utf-8'))
        self.card.get_max_sample_rate()
      except Exception:
        self.card.close()
        self.card = None
        tm.sleep(TIME_OUT)
    
    try:
      self.print("  Transferred card from main worker to mid-flight worker.")

      self.print(f"  Reading from HDF5...      ", end = " ")
      data              = h5py.File(self.h5file, "r")
      group             = data[f"devices/{self.device_name}"]
      segments          = np.asarray(group["SEGMENTS"])
      segment_lengths   = np.asarray(group["SEGMENTS"].attrs["LENGTHS"])
      connections       = np.asarray(group["SEGMENTS"].attrs["CONNECTIONS"])
      sample_rate       = np.asarray(group["SEGMENTS"].attrs["SAMPLE_RATE"])
      copies            = np.asarray(group["SEGMENTS"].attrs["COPIES"])
      differentials     = np.asarray(group["SEGMENTS"].attrs["DIFFERENTIALS"])
      amplitudes        = np.asarray(group["SEGMENTS"].attrs["AMPLITUDES"])
      amplitude_indices = np.asarray(group["SEGMENTS"].attrs["AMPLITUDE_INDICES"])
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
      self.print("Done!")

      self.print(f"  Setting card parameters...", end = " ")
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
      number_of_segments_round = 2**(math.ceil(math.log2(number_of_segments)))

      self.card.set_number_of_segments(number_of_segments_round)
      self.card.set_memory_size(number_of_segments_round*segment_lengths.max())

      self.card.set_channels_enable(**{f"channel_{channel_index}":True for channel_index in connections})
      for channel_index in connections:
        for amplitude_index, amplitude in zip(amplitude_indices, amplitudes):
          if amplitude_index == channel_index:
            self.card.output_enable(int(channel_index))
            self.card.set_amplitude(int(channel_index), amplitude)
          elif amplitude_index == channel_index + 1:
            if (channel_index in copies) or (channel_index in differentials):
              self.card.output_enable(int(channel_index) + 1)
              self.card.set_amplitude(int(channel_index) + 1, amplitude)
      self.print("Done!")

      self.print(f"  Transferring waveforms...  0%", end = "\r")
      for segment_index in range(number_of_segments):
        signals = []
        segment_length = segment_lengths[segment_index]
        for channel_index in range(number_of_channels):
          signals.append(segments[segment_index, :segment_length, channel_index])
        self.card.array_to_device(signals, segment_index)
        self.print(f"  Transferring waveforms...  {(segment_index + 1)*100//number_of_segments}%", end = "\r")
      self.print(f"  Transferring waveforms...  Done!")

      self.print(f"  Transferring sequence...   0%", end = "\r")
      sequence_index = 0
      sequence_size = len(sequence["step"])
      for step, next_step, segment, loops, end_of_sequence, loop_until_trigger in zip(sequence["step"], sequence["next_step"], sequence["segment"], sequence["loops"], sequence["end_of_sequence"], sequence["loop_until_trigger"]):
        # print(step, next_step, segment, loops, end_of_sequence, loop_until_trigger)
        self.card.set_step_instruction(int(step), int(segment), int(loops), int(next_step), bool(loop_until_trigger), bool(end_of_sequence))
        sequence_index += 1
        self.print(f"  Transferring sequence...   {(sequence_index + 1)*100//sequence_size}%", end = "\r")
      self.print(f"  Transferring sequence...   Done!")

      self.card.set_start_step(start_steps[0])
      if not software_trigger:
        self.card.trigger_coupling_use_dc(trigger_port)
        self.card.use_trigger_positive_edge(trigger_port, trigger_level, re_arm_threshold = re_arm_level)
        self.card.set_sufficient_triggers(**{f"external_{trigger_port}":True})
        self.card.arm()
        self.print("  Card armed.")
      
      if software_trigger:
        self.card.arm()
        self.card.force_trigger()
        self.print("  Card software triggered.")

      self.print("Release")

      # for index in range(20):
      #   print(self.card.get_status_information())
      #   tm.sleep(1)
      ready = True
      for start_step in start_steps[1:]:
        while ready:
          if self.manual_queue.qsize() > 0:
            raise Exception()
          ready = "Ready" in self.card.get_status_information()
          tm.sleep(self.wait_time)
        # print(self.card.get_status_information())
        self.card.set_start_step(start_step)
        while not ready:
          if self.manual_queue.qsize() > 0:
            raise Exception()
          ready = "Ready" in self.card.get_status_information()
          tm.sleep(self.wait_time)
        # print(self.card.get_status_information())
        
        # print(self.card.get_start_step())
        self.card.arm()
        # print(self.card.get_status_information())
      self.manual_queue.get()
    except Exception:
      pass
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
  
  amplitude_min   = 80e-3
  sample_rate_min = 50

  # def __init__(self, *args, **kwargs):
  #   self.manual_queue = mp.Queue()
  #   super().__init__(*args, **kwargs)

  # def start(self, *args, **kwargs):
  #   to_queue, from_queue = super().start(*args, **kwargs)
  #   self.manual_queue = to_queue
  #   return to_queue, from_queue

  def init(self):
    # self.BLACS_connection   = workerargs["connection"]
    self.card = None
    while self.card is None:
      try:
        self.card = sc.Card(bytes(self.BLACS_connection, 'utf-8'))
        self.card.get_max_sample_rate()
      except Exception:
        self.card.close()
        self.card = None
        tm.sleep(TIME_OUT)

    self.card.stop()
    self.card.reset()
    self.card_name          = self.card.get_name()
    self.number_of_channels = self.card.get_number_of_channels_per_front_end_module()
    time_string             = dtm.datetime.now().strftime("%Y%m%dT%H%M%S")

    self.worker_mid_flight = None
    self.manual_queue = mp.Queue()

    print(f"{time_string}> Initialised AWG card:")
    print(f"  Model: {self.card_name}")
    print(f"  Channels: {self.number_of_channels}")
    print(f"  Memory: {self.card.get_max_memory_size()/(2**30)} GiB")
    # print(f"Temperatures: FPGA: {self.card.get_temperature_base()} degC, Amplifier: {self.card.get_temperature_module_1()} degC")
    self.front_panel_previous = {}

    self.manual_waveform  = ["" for channel_index in range(self.number_of_channels)]
    self.manual_waveforms = ["Sine", "Square", "Sawtooth"]
    self.segment_size     = 512
    self.signals          = [None for channel_index in range(self.number_of_channels)]

  def program_manual(self, front_panel_values):
    for key, value in front_panel_values.items():
      if key not in ["Sample rate", "Segment size"]:
        continue
      if key in self.front_panel_previous:
        if self.front_panel_previous[key] == value:
          continue
      if key == "Sample rate":
        self.card.stop()
        if value < self.sample_rate_min:
          value = self.sample_rate_min
        self.card.set_sample_rate(value, "M")
        for channel_index in range(self.number_of_channels):
          self.manual_waveform[channel_index] = ""
          self.signals = [None for channel_index in range(self.number_of_channels)]
      if key == "Segment size":
        self.card.stop()
        self.segment_size = max(int(round(value/32)*32), 32)
        self.card.set_memory_size(self.segment_size)
        front_panel_values[key] = self.segment_size
        for channel_index in range(self.number_of_channels):
          self.manual_waveform[channel_index] = ""
          self.signals = [None for channel_index in range(self.number_of_channels)]
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
    for key, value in front_panel_values.items():
      key_split = key.split(":")
      if len(key_split) < 2:
        continue
      key_base = key_split[0]
      key_channel_index = int(key_split[1])
      if key_base in self.manual_waveforms:
        if value and self.manual_waveform[key_channel_index] != key_base:
          for waveform in self.manual_waveforms:
            if waveform != key_base:
              front_panel_values[f"{waveform}:{key_channel_index}"] = False

          signal_length = self.segment_size
          if key_base == "Sine":
            self.signals[key_channel_index] = np.sin(math.tau*np.arange(0, signal_length)/signal_length)
          if key_base == "Square":
            self.signals[key_channel_index] = np.sign(np.sin(math.tau*np.arange(0, signal_length)/signal_length))
          if key_base == "Sawtooth":
            self.signals[key_channel_index] = np.linspace(-1, 1, signal_length)

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
    time_string = dtm.datetime.now().strftime("%Y%m%dT%H%M%S")
    print(f"{time_string}> Preparing for shot:")

    done_queue = mp.Queue()

    self.card.close()
    self.worker_mid_flight = SpectrumAwgWorkerMidFlight(done_queue, self.manual_queue, h5file, device_name, self.BLACS_connection)
    self.worker_mid_flight.start()

    while True:
      arguments, keyword_arguments = done_queue.get()
      if "Release" in arguments:
        break
      print(*arguments, **keyword_arguments)

    return initial_values

  def transition_to_manual(self, abort = False):
    if self.worker_mid_flight is not None:
      self.manual_queue.put("Manual")
      print("  Waiting for mid-flight worker...")
      self.worker_mid_flight.join()
      print("  Transferred card from mid-flight worker to main worker.")
      self.worker_mid_flight = None
    self.card = None
    while self.card is None:
      try:
        self.card = sc.Card(bytes(self.BLACS_connection, 'utf-8'))
        self.card.get_max_sample_rate()
      except Exception:
        self.card.close()
        self.card = None
        tm.sleep(TIME_OUT)
    self.card.stop()
    self.card.reset()
    self.card.use_mode_single()
    front_panel_reset = self.front_panel_previous
    self.front_panel_previous = {}
    self.program_manual(front_panel_reset)
    return True

  def abort_transition_to_buffered(self):
    return self.transition_to_manual(True)
      
  def abort_buffered(self):
    return self.transition_to_manual(True)
  
  # def __del__(self):
  #   self.shutdown()

  def shutdown(self):
    if self.worker_mid_flight is not None:
      self.manual_queue.put("Manual")
      self.worker_mid_flight.join()
      self.worker_mid_flight = None
    else:
      self.card.identification_led_disable()
      self.card.reset()
      self.card.close()

  def check_remote_values(self):
    front_panel_values = {
      "Identify"    : self.card.get_card_identification() != 0,
      "Enable"      : "Ready" not in self.card.get_status_information(),
      "Sample rate" : self.card.get_sample_rate()/1e6,
    }

    if front_panel_values["Sample rate"] < self.sample_rate_min:
      front_panel_values["Sample rate"] = self.sample_rate_min

    for channel_index in range(self.number_of_channels):
      front_panel_values[f"Output enable:{channel_index}"] = self.card.get_output_enable(channel_index) != 0
      front_panel_values[f"Amplitude:{channel_index}"]     = self.card.get_amplitude(channel_index)
      if front_panel_values[f"Amplitude:{channel_index}"] < self.amplitude_min:
        front_panel_values[f"Amplitude:{channel_index}"] = self.amplitude_min
    return front_panel_values

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
  def initialise_GUI(self):
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
      channel_analog_io.append({
        f"Amplitude:{channel_index}":{
        "base_unit"   : "V",
          "min"       : 80e-3,
          "max"       : 2.5,
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
    self.workers["worker"][1].put(("shutdown", [], {}))
    # tm.sleep(1)
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