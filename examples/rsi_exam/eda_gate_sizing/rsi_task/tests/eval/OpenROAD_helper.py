#BSD 3-Clause License
#
#Copyright (c) 2024, ASU-VDA-Lab
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import openroad as ord
from openroad import Tech, Design
import os, odb
from pathlib import Path
from collections import defaultdict

def load_design(design_name, verilog = False):
  tech = Tech()
  libDir = Path("../../platform/ASAP7/lib/")
  lefDir = Path("../../platform/ASAP7/lef/")
  designDir = Path("../../design/%s"%design_name)

  # Read technology files
  libFiles = sorted(libDir.glob('*.lib'))
  lefFiles = sorted(lefDir.glob('*.lef'))
  for libFile in libFiles:
    tech.readLiberty(libFile.as_posix())
  tech.readLef("%s/%s"%(lefDir.as_posix(), "asap7_tech_1x_201209.lef"))
  for lefFile in lefFiles:
    tech.readLef(lefFile.as_posix())
  design = Design(tech)

  # Read design files
  if verilog:
    verilogFile = "%s/%s.v"%(designDir.as_posix(), design_name)
    design.readVerilog(verilogFile)
    design.link(design_name)
  else:
    defFile = "%s/%s.def"%(designDir.as_posix(), design_name)
    design.readDef(defFile)

  # Read the SDC file, SPEF file, and set the clocks
  sdcFile = "%s/%s.sdc"%(designDir.as_posix(), design_name)
  design.evalTclString("read_sdc %s"%sdcFile)
  # setRC.tcl assumes fF/kOhm, but OpenSTA takes its global command units from the
  # FIRST Liberty read and this platform mixes fF/ps (asap7sc7p5t_*) with pF/ns
  # (sram_*). The load order depends on filesystem enumeration: a bind-mounted
  # ext4 directory returns sram_* first, while the same files baked into an image
  # layer return asap7sc7p5t_* first. Sorting above pins today's order; this check
  # is what keeps it correct if a library sorting ahead of asap7sc7p5t_* is added.
  cap_unit = float(design.evalTclString("sta::capacitance_ui_sta 1.0"))
  time_unit = float(design.evalTclString("sta::time_ui_sta 1.0"))
  print("[units] first_liberty=%s cap_ui=%g time_ui=%g"
        % (libFiles[0].name, cap_unit, time_unit), flush=True)
  if abs(cap_unit - 1e-15) > 1e-18 or abs(time_unit - 1e-12) > 1e-15:
    raise RuntimeError(
      "OpenSTA command units are not fF/ps (cap=%g F, time=%g s); first Liberty "
      "read was %s. setRC.tcl and the TNS conversion both assume fF/ps."
      % (cap_unit, time_unit, libFiles[0].name))
  design.evalTclString("source ../../platform/ASAP7/setRC.tcl")
  
  # Global connect
  VDDNet = design.getBlock().findNet("VDD")
  if VDDNet is None:
    VDDNet = odb.dbNet_create(design.getBlock(), "VDD")
  VDDNet.setSpecial()
  VDDNet.setSigType("POWER")
  VSSNet = design.getBlock().findNet("VSS")
  if VSSNet is None:
    VSSNet = odb.dbNet_create(design.getBlock(), "VSS")
  VSSNet.setSpecial()
  VSSNet.setSigType("GROUND")
  design.getBlock().addGlobalConnect(None, ".*", "VDD", VDDNet, True)
  design.getBlock().addGlobalConnect(None, ".*", "VSS", VSSNet, True)
  design.getBlock().globalConnect()

  return tech, design

#################################################
# Get the load pin capacitance of an output pin #
#################################################
def get_output_load_pin_cap(pin, corner, timing):
  # first check if the pin is an output pin
  if pin.isOutputSignal():
    output_load_pin_cap = 0
    # loop through all pins connected to this pin (including this pin)
    for net_pin in pin.getNet().getITerms():
      # get the downstream input pin caps
      if net_pin.isInputSignal():
        output_load_pin_cap += timing.getPortCap(net_pin, corner, timing.Max)
    return output_load_pin_cap
  else:
    return -1

############################
# Get a dict of equivcells #
############################
def build_libcell_dict(filename):
  id_to_names = defaultdict(list)
  with open(filename, "r") as file:
    for line in file:
      line = line.split(",")
      id_to_names[line[1][:-1]].append(line[0])
  
  libcell_dict = dict()
  for id, names in id_to_names.items():
    for name in names:
      libcell_dict[name] = names

  return libcell_dict  
