#include "Operations.hpp"

namespace dd {
	template <class Config>
	void dumpTensor(qc::Operation* op, std::ostream& of,
		std::vector<std::size_t>& inds, size_t& gateIdx,
		std::unique_ptr<dd::Package<Config>>& dd) {
		return;
	}

	template void
		dumpTensor<DDPackageConfig>(qc::Operation* op, std::ostream& of,
			std::vector<std::size_t>& inds, size_t& gateIdx,
			std::unique_ptr<dd::Package<DDPackageConfig>>& dd);
} // namespace dd
