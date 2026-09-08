#pragma once

#include "Operation.hpp"

namespace qc {
class StandardOperation : public Operation {
protected:
  static void checkInteger(fp& ld) {
    const fp nearest = std::nearbyint(ld);
    if (std::abs(ld - nearest) < PARAMETER_TOLERANCE) {
      ld = nearest;
    }
  }

  static void checkFractionPi(fp& ld) {
    const fp div = PI / ld;
    const fp nearest = std::nearbyint(div);
    if (std::abs(div - nearest) < PARAMETER_TOLERANCE) {
      ld = PI / nearest;
    }
  }

  OpType parseU3(fp& theta, fp& phi, fp& lambda);
  OpType parseU2(fp& phi, fp& lambda);
  OpType parseU1(fp& lambda);

  void checkUgate();
  void setup(std::size_t nq, Qubit startingQubit = 0);

  void dumpOpenQASMTeleportation(std::ostream& of,
                                 const RegisterNames& qreg) const;

public:
  StandardOperation() = default;

  // Standard Constructors
  StandardOperation(std::size_t nq, Qubit target, OpType g,
                    std::vector<fp> params = {}, Qubit startingQubit = 0);
  StandardOperation(std::size_t nq, const Targets& targ, OpType g,
                    std::vector<fp> params = {}, Qubit startingQubit = 0);

  StandardOperation(std::size_t nq, Control control, Qubit target, OpType g,
                    const std::vector<fp>& params = {},
                    Qubit startingQubit = 0);
  StandardOperation(std::size_t nq, Control control, const Targets& targ,
                    OpType g, const std::vector<fp>& params = {},
                    Qubit startingQubit = 0);

  StandardOperation(std::size_t nq, const Controls& c, Qubit target, OpType g,
                    const std::vector<fp>& params = {},
                    Qubit startingQubit = 0);
  StandardOperation(std::size_t nq, const Controls& c, const Targets& targ,
                    OpType g, const std::vector<fp>& params = {},
                    Qubit startingQubit = 0);

  // MCT Constructor
  StandardOperation(std::size_t nq, const Controls& c, Qubit target,
                    Qubit startingQubit = 0);

  // MCF (cSWAP), Peres, parameterized two target Constructor
  StandardOperation(std::size_t nq, const Controls& c, Qubit target0,
                    Qubit target1, OpType g, const std::vector<fp>& params = {},
                    Qubit startingQubit = 0);

  [[nodiscard]] std::unique_ptr<Operation> clone() const override {
    return std::make_unique<StandardOperation>(
        getNqubits(), getControls(), getTargets(), getType(), getParameter(),
        getStartingQubit());
  }

  [[nodiscard]] bool isStandardOperation() const override { return true; }

  [[nodiscard]] bool equals(const Operation& op, const Permutation& perm1,
                            const Permutation& perm2) const override {
    return Operation::equals(op, perm1, perm2);
  }
  [[nodiscard]] bool equals(const Operation& operation) const override {
    return equals(operation, {}, {});
  }

  void dumpOpenQASM(std::ostream& of, const RegisterNames& qreg,
                    const RegisterNames& creg) const override;
};

} // namespace qc
