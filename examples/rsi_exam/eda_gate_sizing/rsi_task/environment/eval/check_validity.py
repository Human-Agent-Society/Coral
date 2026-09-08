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

from openroad import Design, Timing
import openroad as ord
from collections import defaultdict

def check_validity(file_path: str, design: Design, timing: Timing):
  # Build original instance name : libcell type map
  changeTypeDict = defaultdict()
  with open(file_path, "r") as file:
    for line in file:
      line = line.split()
      changeTypeDict[line[0]] = line[1]

  # Start examine the correctness of the result
  db = ord.get_db()
  block = design.getBlock()
  timing.makeEquivCells()
  for instName, libcellName in changeTypeDict.items():
    inst = block.findInst(instName)
    if inst == None:
      print("Error: Instance \"%s\" not found."%instName)
      return False
    correctMasters = timing.equivCells(inst.getMaster()) if not (design.isSequential(inst.getMaster()) or inst.getMaster().isBlock()) else [inst.getMaster()]
    if libcellName not in [correctMaster.getName() for correctMaster in correctMasters]:
      correctMasters = ", ".join([correctMaster.getName() for correctMaster in correctMasters])
      print("Error: Instance \"%s\" should be using the following library cells: %s, but found intending to switch to: %s"%(instName, correctMasters, libcellName))
      return False
  return True

