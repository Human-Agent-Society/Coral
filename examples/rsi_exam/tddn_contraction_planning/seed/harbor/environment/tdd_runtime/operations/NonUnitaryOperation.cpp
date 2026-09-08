/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for
 * more information.
 */

#include "NonUnitaryOperation.hpp"

#include <algorithm>
#include <cassert>
#include <utility>

namespace qc {
// Measurement constructor
NonUnitaryOperation::NonUnitaryOperation(const std::size_t nq,
                                         std::vector<Qubit> qubitRegister,
                                         std::vector<Bit> classicalRegister)
    : qubits(std::move(qubitRegister)), classics(std::move(classicalRegister)) {
  if (qubits.size() != classics.size()) {
    throw std::invalid_argument(
        "Sizes of qubit register and classical register do not match.");
  }
  // i-th qubit to be measured shall be measured into i-th classical register
  type = Measure;
  nqubits = nq;
  Operation::setName();
}
NonUnitaryOperation::NonUnitaryOperation(const std::size_t nq,
                                         const Qubit qubit, const Bit cbit) {
  type = Measure;
  nqubits = nq;
  qubits.emplace_back(qubit);
  classics.emplace_back(cbit);
  Operation::setName();
}

// Snapshot constructor
NonUnitaryOperation::NonUnitaryOperation(
    const std::size_t nq, const std::vector<Qubit>& qubitRegister,
    const std::size_t n)
    : NonUnitaryOperation(nq, qubitRegister, Snapshot) {
  parameter.emplace_back(static_cast<fp>(n));
}

// General constructor
NonUnitaryOperation::NonUnitaryOperation(
    const std::size_t nq, const std::vector<Qubit>& qubitRegister, OpType op) {
  type = op;
  nqubits = nq;
  targets = qubitRegister;
  std::sort(targets.begin(), targets.end());
  Operation::setName();
}

std::ostream& NonUnitaryOperation::printNonUnitary(
    std::ostream& os, const std::vector<Qubit>& q, const std::vector<Bit>& c,
    const Permutation& permutation) const {
  switch (type) {
  case Measure:
    printMeasurement(os, q, c, permutation);
    break;
  case Reset:
  case Barrier:
  case Snapshot:
    printResetBarrierOrSnapshot(os, q, permutation);
    break;
  case ShowProbabilities:
    os << name;
    break;
  default:
    std::cerr << "Non-unitary operation with invalid type " << type
              << " detected. Proceed with caution!" << std::endl;
    break;
  }
  return os;
}

void NonUnitaryOperation::dumpOpenQASM(std::ostream& of,
                                       const RegisterNames& qreg,
                                       const RegisterNames& creg) const {
  auto classicsIt = classics.cbegin();
  switch (type) {
  case Measure:
    if (isWholeQubitRegister(qreg, qubits.front(), qubits.back()) &&
        isWholeQubitRegister(qreg, classics.front(), classics.back())) {
      of << "measure " << qreg[qubits.front()].first << " -> "
         << creg[classics.front()].first << ";" << std::endl;
    } else {
      for (const auto& c : qubits) {
        of << "measure " << qreg[c].second << " -> " << creg[*classicsIt].second
           << ";" << std::endl;
        ++classicsIt;
      }
    }
    break;
  case Reset:
    if (isWholeQubitRegister(qreg, targets.front(), targets.back())) {
      of << "reset " << qreg[targets.front()].first << ";" << std::endl;
    } else {
      for (const auto& target : targets) {
        of << "reset " << qreg[target].second << ";" << std::endl;
      }
    }
    break;
  case Snapshot:
    if (!targets.empty()) {
      of << "snapshot(" << parameter[0] << ") ";

      for (unsigned int q = 0; q < targets.size(); ++q) {
        if (q > 0) {
          of << ", ";
        }
        of << qreg[targets[q]].second;
      }
      of << ";" << std::endl;
    }
    break;
  case ShowProbabilities:
    of << "show_probabilities;" << std::endl;
    break;
  case Barrier:
    if (isWholeQubitRegister(qreg, targets.front(), targets.back())) {
      of << "barrier " << qreg[targets.front()].first << ";" << std::endl;
    } else {
      for (const auto& target : targets) {
        of << "barrier " << qreg[target].second << ";" << std::endl;
      }
    }
    break;
  default:
    std::cerr << "Non-unitary operation with invalid type " << type
              << " detected. Proceed with caution!" << std::endl;
    break;
  }
}

bool NonUnitaryOperation::actsOn(Qubit i) const {
  if (type == Measure) {
    return std::any_of(qubits.cbegin(), qubits.cend(),
                       [&i](const auto& q) { return q == i; });
  }
  if (type == Reset) {
    return std::any_of(targets.cbegin(), targets.cend(),
                       [&i](const auto& t) { return t == i; });
  }
  // other non-unitary operations (e.g., barrier statements) may be ignored
  return false;
}

bool NonUnitaryOperation::equals(const Operation& op, const Permutation& perm1,
                                 const Permutation& perm2) const {
  if (const auto* nonunitary = dynamic_cast<const NonUnitaryOperation*>(&op)) {
    if (getType() != nonunitary->getType()) {
      return false;
    }

    if (getType() == Measure) {
      // check number of qubits to be measured
      const auto nq1 = qubits.size();
      const auto nq2 = nonunitary->qubits.size();
      if (nq1 != nq2) {
        return false;
      }

      // these are just sanity checks and should always be fulfilled
      assert(qubits.size() == classics.size());
      assert(nonunitary->qubits.size() == nonunitary->classics.size());

      std::set<std::pair<Qubit, Bit>> measurements1{};
      auto qubitIt1 = qubits.cbegin();
      auto classicIt1 = classics.cbegin();
      while (qubitIt1 != qubits.cend()) {
        if (perm1.empty()) {
          measurements1.emplace(*qubitIt1, *classicIt1);
        } else {
          measurements1.emplace(perm1.at(*qubitIt1), *classicIt1);
        }
        ++qubitIt1;
        ++classicIt1;
      }

      std::set<std::pair<Qubit, Bit>> measurements2{};
      auto qubitIt2 = nonunitary->qubits.cbegin();
      auto classicIt2 = nonunitary->classics.cbegin();
      while (qubitIt2 != nonunitary->qubits.cend()) {
        if (perm2.empty()) {
          measurements2.emplace(*qubitIt2, *classicIt2);
        } else {
          measurements2.emplace(perm2.at(*qubitIt2), *classicIt2);
        }
        ++qubitIt2;
        ++classicIt2;
      }

      return measurements1 == measurements2;
    }
    return Operation::equals(op, perm1, perm2);
  }
  return false;
}

void NonUnitaryOperation::addDepthContribution(
    std::vector<std::size_t>& depths) const {
  if (type == Measure || type == Reset) {
    Operation::addDepthContribution(depths);
  }
}

void NonUnitaryOperation::printMeasurement(
    std::ostream& os, const std::vector<Qubit>& q, const std::vector<Bit>& c,
    const Permutation& permutation) const {
  auto qubitIt = q.cbegin();
  auto classicIt = c.cbegin();
  os << name << "\t";
  if (permutation.empty()) {
    for (std::size_t i = 0; i < nqubits; ++i) {
      if (qubitIt != q.cend() && *qubitIt == i) {
        os << "\033[34m" << *classicIt << "\t"
           << "\033[0m";
        ++qubitIt;
        ++classicIt;
      } else {
        os << "|\t";
      }
    }
  } else {
    for (const auto& [physical, logical] : permutation) {
      if (qubitIt != q.cend() && *qubitIt == physical) {
        os << "\033[34m" << *classicIt << "\t"
           << "\033[0m";
        ++qubitIt;
        ++classicIt;
      } else {
        os << "|\t";
      }
    }
  }
}

void NonUnitaryOperation::printResetBarrierOrSnapshot(
    std::ostream& os, const std::vector<Qubit>& q,
    const Permutation& permutation) const {
  auto qubitIt = q.cbegin();
  os << name << "\t";
  if (permutation.empty()) {
    for (std::size_t i = 0; i < nqubits; ++i) {
      if (qubitIt != q.cend() && *qubitIt == i) {
        if (type == Reset) {
          os << "\033[31m"
             << "r\t"
             << "\033[0m";
        } else if (type == Barrier) {
          os << "\033[32m"
             << "b\t"
             << "\033[0m";
        } else {
          os << "\033[33m"
             << "s\t"
             << "\033[0m";
        }
        ++qubitIt;
      } else {
        os << "|\t";
      }
    }
  } else {
    for (const auto& [physical, logical] : permutation) {
      if (qubitIt != q.cend() && *qubitIt == physical) {
        if (type == Reset) {
          os << "\033[31m"
             << "r\t"
             << "\033[0m";
        } else if (type == Barrier) {
          os << "\033[32m"
             << "b\t"
             << "\033[0m";
        } else {
          os << "\033[33m"
             << "s\t"
             << "\033[0m";
        }
        ++qubitIt;
      } else {
        os << "|\t";
      }
    }
  }
  if (type == Snapshot) {
    os << "\tp: (" << q.size() << ") (" << parameter[0] << ")";
  }
}
} // namespace qc
