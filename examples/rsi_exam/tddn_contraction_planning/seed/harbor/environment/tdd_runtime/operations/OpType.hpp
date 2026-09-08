#pragma once

#include <cstdint>
#include <functional>
#include <iostream>
#include <string>

namespace qc {
// Natively supported operations of the QFR library
enum OpType : std::uint8_t {
  None,
  // Standard Operations
  GPhase,
  I,
  H,
  X,
  Y,
  Z,
  S,
  Sdag,
  T,
  Tdag,
  V,
  Vdag,
  U3,
  U2,
  Phase,
  SX,
  SXdag,
  RX,
  RY,
  RZ,
  SWAP,
  iSWAP, // NOLINT (readability-identifier-naming)
  Peres,
  Peresdag,
  DCX,
  ECR,
  RXX,
  RYY,
  RZZ,
  RZX,
  XXminusYY,
  XXplusYY,
  // Compound Operation
  Compound,
  // Non Unitary Operations
  Measure,
  Reset,
  Snapshot,
  ShowProbabilities,
  Barrier,
  Teleportation,
  // Classically-controlled Operation
  ClassicControlled,
  // Noise operations
  ATrue,
  AFalse,
  MultiATrue,
  MultiAFalse,
  // Number of OpTypes
  OpCount
};

inline std::string toString(const OpType& opType) {
  switch (opType) {
  case None:
    return "none";
  case GPhase:
    return "gphase";
  case I:
    return "i";
  case H:
    return "h";
  case X:
    return "x";
  case Y:
    return "y";
  case Z:
    return "z";
  case S:
    return "s";
  case Sdag:
    return "sdg";
  case T:
    return "t";
  case Tdag:
    return "tdg";
  case V:
    return "v";
  case Vdag:
    return "vdg";
  case U3:
    return "u3";
  case U2:
    return "u2";
  case Phase:
    return "p";
  case SX:
    return "sx";
  case SXdag:
    return "sxdg";
  case RX:
    return "rx";
  case RY:
    return "ry";
  case RZ:
    return "rz";
  case SWAP:
    return "swap";
  case iSWAP:
    return "iswap";
  case Peres:
    return "peres";
  case Peresdag:
    return "peresdg";
  case DCX:
    return "dcx";
  case ECR:
    return "ecr";
  case RXX:
    return "rxx";
  case RYY:
    return "ryy";
  case RZZ:
    return "rzz";
  case RZX:
    return "rzx";
  case XXminusYY:
    return "xx_minus_yy";
  case XXplusYY:
    return "xx_plus_yy";
  case Compound:
    return "compound";
  case Measure:
    return "measure";
  case Reset:
    return "reset";
  case Snapshot:
    return "snapshot";
  case ShowProbabilities:
    return "show probabilities";
  case Barrier:
    return "barrier";
  case Teleportation:
    return "teleportation";
  case ClassicControlled:
    return "classic controlled";
  default:
    throw std::invalid_argument("Invalid OpType!");
  }
}

inline bool isTwoQubitGate(const OpType& opType) {
  switch (opType) {
  case SWAP:
  case iSWAP:
  case Peres:
  case Peresdag:
  case DCX:
  case ECR:
  case RXX:
  case RYY:
  case RZZ:
  case RZX:
  case XXminusYY:
  case XXplusYY:
    return true;
  default:
    return false;
  }
}

inline std::ostream& operator<<(std::ostream& out, OpType& opType) {
  out << toString(opType);
  return out;
}

const inline static std::unordered_map<std::string, qc::OpType>
    OP_NAME_TO_TYPE = {
        {"none", OpType::None},
        {"gphase", OpType::GPhase},
        {"i", OpType::I},
        {"id", OpType::I},
        {"h", OpType::H},
        {"ch", OpType::H},
        {"x", OpType::X},
        {"cnot", OpType::X},
        {"cx", OpType::X},
        {"mcx", OpType::X},
        {"y", OpType::Y},
        {"cy", OpType::Y},
        {"z", OpType::Z},
        {"cz", OpType::Z},
        {"s", OpType::S},
        {"cs", OpType::S},
        {"sdg", OpType::Sdag},
        {"csdg", OpType::Sdag},
        {"t", OpType::T},
        {"ct", OpType::T},
        {"tdg", OpType::Tdag},
        {"ctdg", OpType::Tdag},
        {"v", OpType::V},
        {"vdg", OpType::Vdag},
        {"u", OpType::U3},
        {"cu", OpType::U3},
        {"u3", OpType::U3},
        {"cu3", OpType::U3},
        {"u2", OpType::U2},
        {"cu2", OpType::U2},
        {"p", OpType::Phase},
        {"cp", OpType::Phase},
        {"mcp", OpType::Phase},
        {"phase", OpType::Phase},
        {"cphase", OpType::Phase},
        {"mcphase", OpType::Phase},
        {"u1", OpType::Phase},
        {"cu1", OpType::Phase},
        {"sx", OpType::SX},
        {"csx", OpType::SX},
        {"sxdg", OpType::SXdag},
        {"csxdg", OpType::SXdag},
        {"rx", OpType::RX},
        {"crx", OpType::RX},
        {"ry", OpType::RY},
        {"cry", OpType::RY},
        {"rz", OpType::RZ},
        {"crz", OpType::RZ},
        {"swap", OpType::SWAP},
        {"cswap", OpType::SWAP},
        {"iswap", OpType::iSWAP},
        {"peres", OpType::Peres},
        {"peresdg", OpType::Peresdag},
        {"dcx", OpType::DCX},
        {"ecr", OpType::ECR},
        {"rxx", OpType::RXX},
        {"ryy", OpType::RYY},
        {"rzz", OpType::RZZ},
        {"rzx", OpType::RZX},
        {"xx_minus_yy", OpType::XXminusYY},
        {"xx_plus_yy", OpType::XXplusYY},
        {"measure", OpType::Measure},
        {"reset", OpType::Reset},
        {"snapshot", OpType::Snapshot},
        {"show probabilities", OpType::ShowProbabilities},
        {"barrier", OpType::Barrier},
        {"teleportation", OpType::Teleportation},
        {"classic controlled", OpType::ClassicControlled},
        {"compound", OpType::Compound},
};

[[nodiscard]] inline OpType opTypeFromString(const std::string& opType) {
  // try to find the operation type in the map of known operation types and
  // return it if found or throw an exception otherwise.
  if (const auto it = OP_NAME_TO_TYPE.find(opType);
      it != OP_NAME_TO_TYPE.end()) {
    return OP_NAME_TO_TYPE.at(opType);
  }
  throw std::invalid_argument("Unsupported operation type: " + opType);
}

inline std::istream& operator>>(std::istream& in, OpType& opType) {
  std::string opTypeStr;
  in >> opTypeStr;

  if (opTypeStr.empty()) {
    in.setstate(std::istream::failbit);
    return in;
  }

  opType = opTypeFromString(opTypeStr);
  return in;
}

} // namespace qc
