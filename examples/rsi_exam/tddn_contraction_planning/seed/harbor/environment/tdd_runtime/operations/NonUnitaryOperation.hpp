#pragma once

#include "Operation.hpp"

namespace qc {

class NonUnitaryOperation final : public Operation {
protected:
  std::vector<Qubit>
      qubits{}; // vector for the qubits to measure (necessary since std::set
                // does not preserve the order of inserted elements)
  std::vector<Bit> classics{}; // vector for the classical bits to measure into

  std::ostream& printNonUnitary(std::ostream& os, const std::vector<Qubit>& q,
                                const std::vector<Bit>& c = {},
                                const Permutation& permutation = {}) const;
  void printMeasurement(std::ostream& os, const std::vector<Qubit>& q,
                        const std::vector<Bit>& c,
                        const Permutation& permutation) const;
  void printResetBarrierOrSnapshot(std::ostream& os,
                                   const std::vector<Qubit>& q,
                                   const Permutation& permutation) const;

public:
  // Measurement constructor
  NonUnitaryOperation(std::size_t nq, std::vector<Qubit> qubitRegister,
                      std::vector<Bit> classicalRegister);
  NonUnitaryOperation(std::size_t nq, Qubit qubit, Bit cbit);

  // Snapshot constructor
  NonUnitaryOperation(std::size_t nq, const std::vector<Qubit>& qubitRegister,
                      std::size_t n);

  // ShowProbabilities constructor
  explicit NonUnitaryOperation(const std::size_t nq) {
    nqubits = nq;
    type = ShowProbabilities;
  }

  // General constructor
  NonUnitaryOperation(std::size_t nq, const std::vector<Qubit>& qubitRegister,
                      OpType op = Reset);

  [[nodiscard]] std::unique_ptr<Operation> clone() const override {
    if (getType() == qc::Measure) {
      return std::make_unique<NonUnitaryOperation>(getNqubits(), getTargets(),
                                                   getClassics());
    }
    if (getType() == qc::Snapshot) {
      return std::make_unique<NonUnitaryOperation>(
          getNqubits(), getTargets(),
          static_cast<std::size_t>(getParameter().at(0)));
    }
    if (getType() == qc::ShowProbabilities) {
      return std::make_unique<NonUnitaryOperation>(getNqubits());
    }
    return std::make_unique<NonUnitaryOperation>(getNqubits(), getTargets(),
                                                 getType());
  }

  [[nodiscard]] bool isUnitary() const override { return false; }

  [[nodiscard]] bool isNonUnitaryOperation() const override { return true; }

  [[nodiscard]] const Targets& getTargets() const override {
    if (type == Measure) {
      return qubits;
    }
    return targets;
  }
  Targets& getTargets() override {
    if (type == Measure) {
      return qubits;
    }
    return targets;
  }
  [[nodiscard]] std::size_t getNtargets() const override {
    return getTargets().size();
  }

  [[nodiscard]] const std::vector<Bit>& getClassics() const { return classics; }
  std::vector<Bit>& getClassics() { return classics; }
  [[nodiscard]] size_t getNclassics() const { return classics.size(); }

  [[nodiscard]] bool actsOn(Qubit i) const override;

  void addDepthContribution(std::vector<std::size_t>& depths) const override;

  [[nodiscard]] bool equals(const Operation& op, const Permutation& perm1,
                            const Permutation& perm2) const override;
  [[nodiscard]] bool equals(const Operation& operation) const override {
    return equals(operation, {}, {});
  }

  std::ostream& print(std::ostream& os) const override {
    if (type == Measure) {
      return printNonUnitary(os, qubits, classics);
    }
    return printNonUnitary(os, targets);
  }
  std::ostream& print(std::ostream& os,
                      const Permutation& permutation) const override {
    if (type == Measure) {
      return printNonUnitary(os, qubits, classics, permutation);
    }
    return printNonUnitary(os, targets, {}, permutation);
  }

  void dumpOpenQASM(std::ostream& of, const RegisterNames& qreg,
                    const RegisterNames& creg) const override;

  [[nodiscard]] std::set<Qubit> getUsedQubits() const override {
    const auto& ts = getTargets();
    return {ts.begin(), ts.end()};
  }
};
} // namespace qc
