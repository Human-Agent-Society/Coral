#pragma once

#include "Definitions.hpp"
#include "GateMatrixDefinitions.hpp"
#include "Package.hpp"
#include "../operations/ClassicControlledOperation.hpp"
#include "../operations/CompoundOperation.hpp"
#include "../operations/NonUnitaryOperation.hpp"
#include "../operations/StandardOperation.hpp"

#include <variant>

namespace dd {
	// single-target Operations
	template <class Config>
	qc::MatrixDD getStandardOperationDD(const qc::StandardOperation* op,
		std::unique_ptr<dd::Package<Config>>& dd,
		const dd::Controls& controls,
		dd::Qubit target, bool inverse) {
		return mEdge::zero;
	}

	// two-target Operations
	template <class Config>
	qc::MatrixDD getStandardOperationDD(const qc::StandardOperation* op,
		std::unique_ptr<dd::Package<Config>>& dd,
		const dd::Controls& controls,
		dd::Qubit target0, dd::Qubit target1,
		bool inverse) {
		return mEdge::zero;
	}

	// The methods with a permutation parameter apply these Operations according to
	// the mapping specified by the permutation, e.g.
	//      if perm[0] = 1 and perm[1] = 0
	//      then cx 0 1 will be translated to cx perm[0] perm[1] == cx 1 0

	template <class Config>
	qc::MatrixDD getDD(const qc::Operation* op,
		std::unique_ptr<dd::Package<Config>>& dd,
		qc::Permutation& permutation, bool inverse = false) {
		return mEdge::zero;
	}

	template <class Config>
	qc::MatrixDD getDD(const qc::Operation* op,
		std::unique_ptr<dd::Package<Config>>& dd,
		bool inverse = false) {
		qc::Permutation perm{};
		return getDD(op, dd, perm, inverse);
	}

	template <class Config>
	qc::MatrixDD getInverseDD(const qc::Operation* op,
		std::unique_ptr<dd::Package<Config>>& dd) {
		return getDD(op, dd, true);
	}

	template <class Config>
	qc::MatrixDD getInverseDD(const qc::Operation* op,
		std::unique_ptr<dd::Package<Config>>& dd,
		qc::Permutation& permutation) {
		return getDD(op, dd, permutation, true);
	}

	template <class Config>
	void dumpTensor(qc::Operation* op, std::ostream& of,
		std::vector<std::size_t>& inds, std::size_t& gateIdx,
		std::unique_ptr<dd::Package<Config>>& dd);

	// apply swaps 'on' DD in order to change 'from' to 'to'
	// where |from| >= |to|
	template <class DDType, class Config>
	void changePermutation(DDType& on, qc::Permutation& from,
		const qc::Permutation& to,
		std::unique_ptr<dd::Package<Config>>& dd,
		bool regular = true) {
		return;
	}

} // namespace dd
