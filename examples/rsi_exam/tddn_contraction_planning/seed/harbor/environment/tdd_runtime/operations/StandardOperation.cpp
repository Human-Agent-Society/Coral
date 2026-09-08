#include "StandardOperation.hpp"

#include <cassert>
#include <sstream>
#include <variant>

namespace qc {
/***
 * Protected Methods
 ***/
OpType StandardOperation::parseU3(fp& theta, fp& phi, fp& lambda) {
  if (std::abs(theta) < PARAMETER_TOLERANCE &&
      std::abs(phi) < PARAMETER_TOLERANCE) {
    parameter = {lambda};
    return parseU1(parameter[0]);
  }

  if (std::abs(theta - PI_2) < PARAMETER_TOLERANCE) {
    parameter = {phi, lambda};
    return parseU2(parameter[0], parameter[1]);
  }

  if (std::abs(lambda) < PARAMETER_TOLERANCE) {
    lambda = 0.L;
    if (std::abs(phi) < PARAMETER_TOLERANCE) {
      checkInteger(theta);
      checkFractionPi(theta);
      parameter = {theta};
      return RY;
    }
  }

  if (std::abs(lambda - PI_2) < PARAMETER_TOLERANCE) {
    lambda = PI_2;
    if (std::abs(phi + PI_2) < PARAMETER_TOLERANCE) {
      checkInteger(theta);
      checkFractionPi(theta);
      parameter = {theta};
      return RX;
    }

    if (std::abs(phi - PI_2) < PARAMETER_TOLERANCE) {
      phi = PI_2;
      if (std::abs(theta - PI) < PARAMETER_TOLERANCE) {
        parameter.clear();
        return Y;
      }
    }
  }

  if (std::abs(lambda + PI_2) < PARAMETER_TOLERANCE) {
    lambda = -PI_2;
    if (std::abs(phi - PI_2) < PARAMETER_TOLERANCE) {
      phi = PI_2;
      parameter = {-theta};
      return RX;
    }
  }

  if (std::abs(lambda - PI) < PARAMETER_TOLERANCE) {
    lambda = PI;
    if (std::abs(phi) < PARAMETER_TOLERANCE) {
      phi = 0.L;
      if (std::abs(theta - PI) < PARAMETER_TOLERANCE) {
        parameter.clear();
        return X;
      }
    }
  }

  // parse a real u3 gate
  checkInteger(lambda);
  checkFractionPi(lambda);
  checkInteger(phi);
  checkFractionPi(phi);
  checkInteger(theta);
  checkFractionPi(theta);

  return U3;
}

OpType StandardOperation::parseU2(fp& phi, fp& lambda) {
  if (std::abs(phi) < PARAMETER_TOLERANCE) {
    phi = 0.L;
    if (std::abs(std::abs(lambda) - PI) < PARAMETER_TOLERANCE) {
      parameter.clear();
      return H;
    }
    if (std::abs(lambda) < PARAMETER_TOLERANCE) {
      parameter = {PI_2};
      return RY;
    }
  }

  if (std::abs(lambda - PI_2) < PARAMETER_TOLERANCE) {
    lambda = PI_2;
    if (std::abs(phi + PI_2) < PARAMETER_TOLERANCE) {
      parameter.clear();
      return V;
    }
  }

  if (std::abs(lambda + PI_2) < PARAMETER_TOLERANCE) {
    lambda = -PI_2;
    if (std::abs(phi - PI_2) < PARAMETER_TOLERANCE) {
      parameter.clear();
      return Vdag;
    }
  }

  checkInteger(lambda);
  checkFractionPi(lambda);
  checkInteger(phi);
  checkFractionPi(phi);

  return U2;
}

OpType StandardOperation::parseU1(fp& lambda) {
  if (std::abs(lambda) < PARAMETER_TOLERANCE) {
    parameter.clear();
    return I;
  }
  const bool sign = std::signbit(lambda);

  if (std::abs(std::abs(lambda) - PI) < PARAMETER_TOLERANCE) {
    parameter.clear();
    return Z;
  }

  if (std::abs(std::abs(lambda) - PI_2) < PARAMETER_TOLERANCE) {
    parameter.clear();
    return sign ? Sdag : S;
  }

  if (std::abs(std::abs(lambda) - PI_4) < PARAMETER_TOLERANCE) {
    parameter.clear();
    return sign ? Tdag : T;
  }

  checkInteger(lambda);
  checkFractionPi(lambda);

  return Phase;
}

void StandardOperation::checkUgate() {
  if (parameter.empty()) {
    return;
  }
  if (type == Phase) {
    assert(parameter.size() == 1);
    type = parseU1(parameter.at(0));
  } else if (type == U2) {
    assert(parameter.size() == 2);
    type = parseU2(parameter.at(0), parameter.at(1));
  } else if (type == U3) {
    assert(parameter.size() == 3);
    type = parseU3(parameter.at(0), parameter.at(1), parameter.at(2));
  }
}

void StandardOperation::setup(const std::size_t nq, const Qubit startingQubit) {
  nqubits = nq;
  startQubit = startingQubit;
  checkUgate();
  setName();
}

/***
 * Constructors
 ***/
StandardOperation::StandardOperation(const std::size_t nq, const Qubit target,
                                     const OpType g, std::vector<fp> params,
                                     const Qubit startingQubit) {
  type = g;
  parameter = std::move(params);
  setup(nq, startingQubit);
  targets.emplace_back(target);
}

StandardOperation::StandardOperation(const std::size_t nq, const Targets& targ,
                                     const OpType g, std::vector<fp> params,
                                     const Qubit startingQubit) {
  type = g;
  parameter = std::move(params);
  setup(nq, startingQubit);
  targets = targ;
}

StandardOperation::StandardOperation(const std::size_t nq,
                                     const Control control, const Qubit target,
                                     const OpType g,
                                     const std::vector<fp>& params,
                                     const Qubit startingQubit)
    : StandardOperation(nq, target, g, params, startingQubit) {
  controls.insert(control);
}

StandardOperation::StandardOperation(const std::size_t nq,
                                     const Control control, const Targets& targ,
                                     const OpType g,
                                     const std::vector<fp>& params,
                                     const Qubit startingQubit)
    : StandardOperation(nq, targ, g, params, startingQubit) {
  controls.insert(control);
}

StandardOperation::StandardOperation(const std::size_t nq, const Controls& c,
                                     const Qubit target, const OpType g,
                                     const std::vector<fp>& params,
                                     const Qubit startingQubit)
    : StandardOperation(nq, target, g, params, startingQubit) {
  controls = c;
}

StandardOperation::StandardOperation(const std::size_t nq, const Controls& c,
                                     const Targets& targ, const OpType g,
                                     const std::vector<fp>& params,
                                     const Qubit startingQubit)
    : StandardOperation(nq, targ, g, params, startingQubit) {
  controls = c;
}

// MCT Constructor
StandardOperation::StandardOperation(const std::size_t nq, const Controls& c,
                                     const Qubit target,
                                     const Qubit startingQubit)
    : StandardOperation(nq, c, target, X, {}, startingQubit) {}

// MCF (cSWAP), Peres, parameterized two target Constructor
StandardOperation::StandardOperation(const std::size_t nq, const Controls& c,
                                     const Qubit target0, const Qubit target1,
                                     const OpType g,
                                     const std::vector<fp>& params,
                                     const Qubit startingQubit)
    : StandardOperation(nq, c, {target0, target1}, g, params, startingQubit) {}

/***
 * Public Methods
 ***/
void StandardOperation::dumpOpenQASM(
    std::ostream& of, const RegisterNames& qreg,
    [[maybe_unused]] const RegisterNames& creg) const {
  std::ostringstream op;
  op << std::setprecision(std::numeric_limits<fp>::digits10);
  if ((controls.size() > 1 && type != X) || controls.size() > 2) {
    std::cout << "[WARNING] Multiple controlled gates are not natively "
                 "supported by OpenQASM. "
              << "However, this library can parse .qasm files with multiple "
                 "controlled gates (e.g., cccx) correctly. "
              << "Thus, while not valid vanilla OpenQASM, the dumped file will "
                 "work with this library. "
              << std::endl;
  }

  // safe the numbers of controls as a prefix to the operation name
  op << std::string(controls.size(), 'c');

  switch (type) {
  case GPhase:
    op << "gphase(" << parameter.at(0) << ")";
    break;
  case I:
    op << "id";
    break;
  case H:
    op << "h";
    break;
  case X:
    op << "x";
    break;
  case Y:
    op << "y";
    break;
  case Z:
    op << "z";
    break;
  case S:
    if (!controls.empty()) {
      op << "p(pi/2)";
    } else {
      op << "s";
    }
    break;
  case Sdag:
    if (!controls.empty()) {
      op << "p(-pi/2)";
    } else {
      op << "sdg";
    }
    break;
  case T:
    if (!controls.empty()) {
      op << "p(pi/4)";
    } else {
      op << "t";
    }
    break;
  case Tdag:
    if (!controls.empty()) {
      op << "p(-pi/4)";
    } else {
      op << "tdg";
    }
    break;
  case V:
    op << "u3(pi/2,-pi/2,pi/2)";
    break;
  case Vdag:
    op << "u3(pi/2,pi/2,-pi/2)";
    break;
  case U3:
    op << "u3(" << parameter[0] << "," << parameter[1] << "," << parameter[2]
       << ")";
    break;
  case U2:
    op << "u3(pi/2," << parameter[0] << "," << parameter[1] << ")";
    break;
  case Phase:
    op << "p(" << parameter[0] << ")";
    break;
  case SX:
    op << "sx";
    break;
  case SXdag:
    op << "sxdg";
    break;
  case RX:
    op << "rx(" << parameter[0] << ")";
    break;
  case RY:
    op << "ry(" << parameter[0] << ")";
    break;
  case RZ:
    op << "rz(" << parameter[0] << ")";
    break;
  case DCX:
    op << "dcx";
    break;
  case ECR:
    op << "ecr";
    break;
  case RXX:
    op << "rxx(" << parameter[0] << ")";
    break;
  case RYY:
    op << "ryy(" << parameter[0] << ")";
    break;
  case RZZ:
    op << "rzz(" << parameter[0] << ")";
    break;
  case RZX:
    op << "rzx(" << parameter[0] << ")";
    break;
  case XXminusYY:
    op << "xx_minus_yy(" << parameter[0] << "," << parameter[1] << ")";
    break;
  case XXplusYY:
    op << "xx_plus_yy(" << parameter[0] << "," << parameter[1] << ")";
    break;
  case SWAP:
    op << "swap";
    break;
  case iSWAP:
    op << "iswap";
    break;
  case Peres:
    of << op.str() << "cx";
    for (const auto& c : controls) {
      of << " " << qreg[c.qubit].second << ",";
    }
    of << " " << qreg[targets[1]].second << ", " << qreg[targets[0]].second
       << ";\n";

    of << op.str() << "x";
    for (const auto& c : controls) {
      of << " " << qreg[c.qubit].second << ",";
    }
    of << " " << qreg[targets[1]].second << ";\n";
    return;
  case Peresdag:
    of << op.str() << "x";
    for (const auto& c : controls) {
      of << " " << qreg[c.qubit].second << ",";
    }
    of << " " << qreg[targets[1]].second << ";\n";

    of << op.str() << "cx";
    for (const auto& c : controls) {
      of << " " << qreg[c.qubit].second << ",";
    }
    of << " " << qreg[targets[1]].second << ", " << qreg[targets[0]].second
       << ";\n";
    return;
  case Teleportation:
    dumpOpenQASMTeleportation(of, qreg);
    return;
  default:
    std::cerr << "gate type (index) " << static_cast<int>(type)
              << " could not be converted to OpenQASM" << std::endl;
  }

  // apply X operations to negate the respective controls
  for (const auto& c : controls) {
    if (c.type == Control::Type::Neg) {
      of << "x " << qreg[c.qubit].second << ";\n";
    }
  }
  // apply the operation
  of << op.str();
  // add controls and targets of the operation
  for (const auto& c : controls) {
    of << " " << qreg[c.qubit].second << ",";
  }
  if (!targets.empty()) {
    for (const auto& t : targets) {
      of << " " << qreg[t].second << ",";
    }
    of.seekp(-1, std::ios_base::cur);
  }
  of << ";\n";
  // apply X operations to negate the respective controls again
  for (const auto& c : controls) {
    if (c.type == Control::Type::Neg) {
      of << "x " << qreg[c.qubit].second << ";\n";
    }
  }
}

void StandardOperation::dumpOpenQASMTeleportation(
    std::ostream& of, const RegisterNames& qreg) const {
  if (!controls.empty() || targets.size() != 3) {
    std::cerr << "controls = ";
    for (const auto& c : controls) {
      std::cerr << qreg.at(c.qubit).second << " ";
    }
    std::cerr << "\ntargets = ";
    for (const auto& t : targets) {
      std::cerr << qreg.at(t).second << " ";
    }
    std::cerr << "\n";

    throw QFRException("Teleportation needs three targets");
  }
  /*
                                      ░      ┌───┐ ░ ┌─┐    ░
                  |ψ⟩ q_0: ───────────░───■──┤ H ├─░─┤M├────░─────────────── |0⟩
     or |1⟩ ┌───┐      ░ ┌─┴─┐└───┘ ░ └╥┘┌─┐ ░ |0⟩ a_0: ┤ H ├──■───░─┤ X
     ├──────░──╫─┤M├─░─────────────── |0⟩ or |1⟩ └───┘┌─┴─┐ ░ └───┘      ░  ║
     └╥┘ ░  ┌───┐  ┌───┐ |0⟩ a_1: ─────┤ X ├─░────────────░──╫──╫──░──┤ X ├──┤ Z
     ├─ |ψ⟩ └───┘ ░            ░  ║  ║  ░  └─┬─┘  └─┬─┘ ║  ║    ┌──┴──┐   │
                bitflip: 1/═══════════════════════════╩══╬════╡ = 1 ╞═══╪═══
                                                      0  ║    └─────┘┌──┴──┐
              phaseflip: 1/══════════════════════════════╩═══════════╡ = 1 ╞
                                                         0           └─────┘
          */
  of << "// teleport q_0, a_0, a_1; q_0 --> a_1  via a_0\n";
  of << "teleport " << qreg[targets[0]].second << ", "
     << qreg[targets[1]].second << ", " << qreg[targets[2]].second << ";\n";
}
} // namespace qc
