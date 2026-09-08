#pragma once

#include "../Definitions.hpp"
#include "OpType.hpp"
#include "../Permutation.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <vector>

namespace qc {
class Operation {
protected:
  Controls controls{};
  Targets targets{};
  std::vector<fp> parameter{};

  std::size_t nqubits = 0;
  Qubit startQubit = 0;
  OpType type = None;
  std::string name{};

  static bool isWholeQubitRegister(const RegisterNames& reg, std::size_t start,
                                   std::size_t end) {
    return !reg.empty() && reg[start].first == reg[end].first &&
           (start == 0 || reg[start].first != reg[start - 1].first) &&
           (end == reg.size() - 1 || reg[end].first != reg[end + 1].first);
  }

public:
  Operation() = default;

  Operation(const Operation& op) = delete;
  Operation(Operation&& op) noexcept = default;

  Operation& operator=(const Operation& op) = delete;

  Operation& operator=(Operation&& op) noexcept = default;

  // Virtual Destructor
  virtual ~Operation() = default;

  [[nodiscard]] virtual std::unique_ptr<Operation> clone() const = 0;

  // Getters
  [[nodiscard]] virtual const Targets& getTargets() const { return targets; }
  virtual Targets& getTargets() { return targets; }
  [[nodiscard]] virtual std::size_t getNtargets() const {
    return targets.size();
  }

  [[nodiscard]] virtual const Controls& getControls() const { return controls; }
  virtual Controls& getControls() { return controls; }
  [[nodiscard]] virtual std::size_t getNcontrols() const {
    return controls.size();
  }

  [[nodiscard]] std::size_t getNqubits() const { return nqubits; }

  [[nodiscard]] const std::vector<fp>& getParameter() const {
    return parameter;
  }
  std::vector<fp>& getParameter() { return parameter; }

  [[nodiscard]] const std::string& getName() const { return name; }
  [[nodiscard]] virtual OpType getType() const { return type; }

  [[nodiscard]] virtual Qubit getStartingQubit() const { return startQubit; }

  [[nodiscard]] virtual std::set<Qubit> getUsedQubits() const {
    const auto& opTargets = getTargets();
    const auto& opControls = getControls();
    std::set<Qubit> usedQubits = {opTargets.begin(), opTargets.end()};
    for (const auto& control : opControls) {
      usedQubits.insert(control.qubit);
    }
    return usedQubits;
  }

  // Setter
  virtual void setNqubits(const std::size_t nq) { nqubits = nq; }

  virtual void setTargets(const Targets& t) { targets = t; }

  virtual void setControls(const Controls& c) { controls = c; }

  virtual void setName();

  virtual void setGate(const OpType g) {
    type = g;
    setName();
  }

  virtual void setParameter(const std::vector<fp>& p) { parameter = p; }

  [[nodiscard]] inline virtual bool isUnitary() const { return true; }

  [[nodiscard]] inline virtual bool isStandardOperation() const {
    return false;
  }

  [[nodiscard]] inline virtual bool isCompoundOperation() const {
    return false;
  }

  [[nodiscard]] inline virtual bool isNonUnitaryOperation() const {
    return false;
  }

  [[nodiscard]] inline virtual bool isClassicControlledOperation() const {
    return false;
  }

  [[nodiscard]] inline virtual bool isSymbolicOperation() const {
    return false;
  }

  [[nodiscard]] inline virtual bool isControlled() const {
    return !controls.empty();
  }

  [[nodiscard]] inline virtual bool actsOn(const Qubit i) const {
    for (const auto& t : targets) {
      if (t == i) {
        return true;
      }
    }
    return controls.count(i) > 0;
  }

  virtual void addDepthContribution(std::vector<std::size_t>& depths) const;

  [[nodiscard]] virtual bool equals(const Operation& op,
                                    const Permutation& perm1,
                                    const Permutation& perm2) const;
  [[nodiscard]] virtual bool equals(const Operation& op) const {
    return equals(op, {}, {});
  }

  virtual std::ostream& printParameters(std::ostream& os) const;
  virtual std::ostream& print(std::ostream& os) const;
  virtual std::ostream& print(std::ostream& os,
                              const Permutation& permutation) const;

  friend std::ostream& operator<<(std::ostream& os, const Operation& op) {
    return op.print(os);
  }

  virtual void dumpOpenQASM(std::ostream& of, const RegisterNames& qreg,
                            const RegisterNames& creg) const = 0;
};
} // namespace qc
