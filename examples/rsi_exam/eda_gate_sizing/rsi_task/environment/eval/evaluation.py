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

from openroad import Tech, Design, Timing
import openroad as ord
import odb
from check_validity import check_validity
from collections import defaultdict
from pathlib import Path
import os, argparse, sys

def swap_libcell(filePath: str, design: Design):
  # Build original instance name : libcell type map
  changeTypeDict = defaultdict()
  with open(filePath, "r") as file:
    for line in file:
      line = line.split()
      changeTypeDict[line[0]] = line[1]

  # Start examine the correctness of the result
  db = ord.get_db()
  for inst_name in changeTypeDict:
    inst = design.getBlock().findInst(inst_name)
    if inst == None:
      print("Error: Instance \"%s\" is not in the design"%inst.getName())
      return False
    inst.swapMaster(db.findMaster(changeTypeDict[inst.getName()]))
  return True
 
def ICCAD_evaluation(filePath: str, design: Design, timing: Timing):
  if check_validity(filePath, design, timing):
    # We only have one corner in this contest
    corner = timing.getCorners()[0]
    leakageBeforeSwap = 0
    for inst in design.getBlock().getInsts():
      leakageBeforeSwap += timing.staticPower(inst, corner)
    leakageBeforeSwap *= 1000000
    timing.resetTiming()
    if swap_libcell(filePath, design):
      # Legalization
      site = design.getBlock().getRows()[0].getSite()
      max_disp_x = int((design.getBlock().getBBox().xMax() - design.getBlock().getBBox().xMin()) / site.getWidth())
      max_disp_y = int((design.getBlock().getBBox().yMax() - design.getBlock().getBBox().yMin()) / site.getHeight())
      print("Legalizing...")
      design.getOpendp().detailedPlacement(max_disp_x, max_disp_y, "", False)
      # Global Route and Estimate Global Route RC
      signal_low_layer = design.getTech().getDB().getTech().findLayer("M1").getRoutingLevel()
      signal_high_layer = design.getTech().getDB().getTech().findLayer("M7").getRoutingLevel()
      clk_low_layer = design.getTech().getDB().getTech().findLayer("M1").getRoutingLevel()
      clk_high_layer = design.getTech().getDB().getTech().findLayer("M7").getRoutingLevel()
      grt = design.getGlobalRouter()
      grt.clear()
      grt.setAllowCongestion(True)
      grt.setMinRoutingLayer(signal_low_layer)
      grt.setMaxRoutingLayer(signal_high_layer)
      grt.setMinLayerForClock(clk_low_layer)
      grt.setMaxLayerForClock(clk_high_layer)
      grt.setAdjustment(0.5)
      grt.setVerbose(False)
      print("Run Global Routing...")
      grt.globalRoute(False)
      design.evalTclString("estimate_parasitics -global_routing")
      # Start Evaluation
      tns, slew, cap, leakage = 0, 0, 0, 0
      # Penalties are subject to change
      tnsPenalty, slewPenalty, capPenalty = 10, 20, 20

      # Get all timing metrices
      design.evalTclString("report_tns > tns_evaluation_temp.txt")
      with open ("tns_evaluation_temp.txt", "r") as file:
        for line in file:
          tns = float(line.split()[1]) / 1000
      for pin_ in design.getBlock().getITerms():
        if pin_.getNet() != None:
          if pin_.getNet().getSigType() != 'POWER' and pin_.getNet().getSigType() != 'GROUND' and pin_.getNet().getSigType() != 'CLOCK':
            library_cell_pin = [MTerm for MTerm in pin_.getInst().getMaster().getMTerms() if (pin_.getInst().getName() + "/" + MTerm.getName()) == pin_.getName()][0]       
            if timing.getMaxSlewLimit(library_cell_pin) < timing.getPinSlew(pin_):
              diff = abs(timing.getMaxSlewLimit(library_cell_pin) - timing.getPinSlew(pin_)) * 1000000000
              slew += diff
            if pin_.isOutputSignal():
              if timing.getMaxCapLimit(library_cell_pin) < timing.getNetCap(pin_.getNet(), corner, timing.Max):
                diff = abs(timing.getMaxCapLimit(library_cell_pin) - timing.getNetCap(pin_.getNet(), corner, timing.Max)) * 1000000000000000
                cap += diff
          
      os.remove("tns_evaluation_temp.txt")
      for inst in design.getBlock().getInsts():
        leakage += timing.staticPower(inst, corner)
      leakage *= 1000000
      leakage -= leakageBeforeSwap
      # Adjust penalties
      tnsPenalty = 0 if tns >= 0.0 else tnsPenalty
      slewPenalty = 0 if slew == 0 else slewPenalty
      capPenalty = 0 if cap == 0 else capPenalty
     
      score = leakage + tnsPenalty * abs(tns) + slewPenalty * abs(slew) + capPenalty * abs(cap)
      print("===================================================")
      print("TNS: %f ns"%(tns))
      if slewPenalty != 0:
        print("Total slew violation difference: %f ns"%(slew))
      else:
        print("No slew violation")
      if capPenalty != 0:
        print("Total load capacitance violation difference: %f fF"%(cap))
      else:
        print("No load capacitance violation")
      print("Leakage power difference: %f uW"%(leakage))
      print("Score: %f"%score)
      print("Require runtime in official score calculation")
      print("===================================================")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Output file path and the design name")
  parser.add_argument("--file_path", type = Path, default="./file", action = "store")
  parser.add_argument("--design_name", type = str, default="NaN", action = "store")
  parser.add_argument("--dump_def", default = False, action = "store_true")
  pyargs = parser.parse_args()
  
  sys.path.append("../example/")
  from OpenROAD_helper import load_design

  tech, design = load_design(pyargs.design_name, False)
  timing = Timing(design)
  ICCAD_evaluation(pyargs.file_path, design, timing)
  
  if pyargs.dump_def:
    odb.write_def(design.getBlock(), "%s.def"%pyargs.design_name)
