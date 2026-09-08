#pragma once

#include "Definitions.hpp"



#include "Tdd.hpp"



#include <array>
#include <cstddef>
#include <iostream>
#include <utility>

namespace dd {

	/// Data structure for caching computed results
	/// \tparam LeftOperandType type of the operation's left operand
	/// \tparam RightOperandType type of the operation's right operand
	/// \tparam ResultType type of the operation's result
	/// \tparam NBUCKET number of hash buckets to use (has to be a power of two)
	template <class LeftOperandType, class RightOperandType, class ResultType,
		std::size_t NBUCKET = 16384>
	class ComputeTable {
	public:
		ComputeTable() = default;

		struct Entry {
			LeftOperandType leftOperand;
			RightOperandType rightOperand;
			ResultType result;
		};

		static constexpr std::size_t MASK = NBUCKET - 1;

		static std::size_t hash(const LeftOperandType& leftOperand,
			const RightOperandType& rightOperand) {
			const auto h1 = std::hash<LeftOperandType>{}(leftOperand);
			const auto h2 = std::hash<RightOperandType>{}(rightOperand);
			const auto hash = dd::combineHash(h1, h2);
			return hash & MASK;
		}

		// access functions
		[[nodiscard]] const auto& getTable() const { return table; }

		void insert(const LeftOperandType& leftOperand,
			const RightOperandType& rightOperand, const ResultType& result) {
			const auto key = hash(leftOperand, rightOperand);
			table[key] = { leftOperand, rightOperand, result };
			++count;
		}

		ResultType lookup(const LeftOperandType& leftOperand, const RightOperandType& rightOperand) {
			ResultType result{};
			lookups++;
			const auto key = hash(leftOperand, rightOperand);
			auto& entry = table[key];
			if (entry.result.p == nullptr) {
				return result;
			}
			if (entry.leftOperand != leftOperand) {
				return result;
			}
			if (entry.rightOperand != rightOperand) {
				return result;
			}
			hits++;
			return entry.result;
		}

		void clear() {
			if (count > 0) {
				for (auto& entry : table) {
					entry.result.p = nullptr;
				}
				count = 0;
			}
		}

		[[nodiscard]] fp hitRatio() const {
			return static_cast<fp>(hits) / static_cast<fp>(lookups);
		}

		std::ostream& printStatistics(std::ostream& os = std::cout) {
			os << "hits: " << hits << ", looks: " << lookups
				<< ", ratio: " << hitRatio() << std::endl;
			return os;
		}

	private:
		std::array<Entry, NBUCKET> table{};
		// compute table lookup statistics
		std::size_t hits = 0;
		std::size_t lookups = 0;
		std::size_t count = 0;
	};

	template <class LeftOperandType, class RightOperandType, class ResultType,
		std::size_t NBUCKET = 16384>
	class ComputeTable2 {
	public:
		ComputeTable2() = default;

		struct res {
			ResultType e;
			int cont_num;
		};

		struct Entry {
			LeftOperandType leftOperand;
			RightOperandType rightOperand;
			ResultType result;
			int cont_num;
			key_2_new_key_node* key_2_new_key1;
			key_2_new_key_node* key_2_new_key2;
		};

		static constexpr std::size_t MASK = NBUCKET - 1;

		static std::size_t hash(const LeftOperandType& leftOperand,
			const RightOperandType& rightOperand) {
			const auto h1 = std::hash<LeftOperandType>{}(leftOperand);
			const auto h2 = std::hash<RightOperandType>{}(rightOperand);
			const auto hash = dd::combineHash(h1, h2);
			return hash & MASK;
		}

		// access functions
		[[nodiscard]] const auto& getTable() const { return table; }

		void insert(const LeftOperandType& leftOperand,
			const RightOperandType& rightOperand, const ResultType& result,
			key_2_new_key_node* key_2_new_key1, key_2_new_key_node* key_2_new_key2, const int cont_num) {
			const auto key = hash(leftOperand, rightOperand);
			table[key] = { leftOperand, rightOperand, result, cont_num, key_2_new_key1, key_2_new_key2 };
			++count;
		}

		res lookup(const LeftOperandType& leftOperand,
			const RightOperandType& rightOperand, key_2_new_key_node* key_2_new_key1, key_2_new_key_node* key_2_new_key2) {
			ResultType result{};
			lookups++;
			const auto key = hash(leftOperand, rightOperand);
			auto& entry = table[key];
			if (entry.result.p == nullptr) {
				return { result,0 };
			}
			if (entry.leftOperand != leftOperand) {
				return { result,0 };
			}
			if (entry.rightOperand != rightOperand) {
				return { result,0 };
			}

			if (entry.key_2_new_key1 != key_2_new_key1) {
				return { result,0 };
			}

			if (entry.key_2_new_key2 != key_2_new_key2) {
				return { result,0 };
			}

			hits++;

			return { entry.result, entry.cont_num };
		}

		void clear() {
			if (count > 0) {
				for (auto& entry : table) {
					entry.result.p = nullptr;
				}
				count = 0;
			}
		}

		[[nodiscard]] fp hitRatio() const {
			return static_cast<fp>(hits) / static_cast<fp>(lookups);
		}

		std::ostream& printStatistics(std::ostream& os = std::cout) {
			os << "hits: " << hits << ", looks: " << lookups
				<< ", ratio: " << hitRatio() << std::endl;
			return os;
		}

	private:
		std::array<Entry, NBUCKET> table{};
		// compute table lookup statistics
		std::size_t hits = 0;
		std::size_t lookups = 0;
		std::size_t count = 0;
	};


} // namespace dd
