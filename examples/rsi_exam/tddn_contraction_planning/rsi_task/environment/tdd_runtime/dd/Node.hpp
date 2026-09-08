#pragma once

#include "Complex.hpp"
#include "ComplexValue.hpp"
#include "Definitions.hpp"
#include "Edge.hpp"

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

namespace dd {


	// NOLINTNEXTLINE(readability-identifier-naming)
	struct mNode {
		//std::array<Edge<mNode>, RADIX> e{}; // edges out of this node
		std::vector<Edge<mNode>> e = {}; // edges out of this node
		mNode* next{};                      // used to link nodes in unique table
		RefCount ref{};                     // reference count
		Qubit v{}; // variable index (nonterminal) value (-1 for terminal)
		std::uint8_t flags = 0;

		// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
		static mNode terminal;

		static constexpr bool isTerminal(const mNode* p) { return p == &terminal; }
		static constexpr mNode* getTerminal() { return &terminal; }

		mNode copy() {
			mNode copy;
			printf("About to copy TDD edge node value");
			//printf(" with %d edges\n", e.size());
			copy.e = {};
			if (&e != nullptr && !e.empty()) {
				printf("e is not empty");
				for (int i = 0; i < e.size(); i++) {
					copy.e.push_back(e[i].copy());
				}
			}
			printf("About to copy TDD edge node value p2\n");
			if (next != nullptr) {
				mNode temp = (*next);
				printf("Not an issue with the pointer: ");
				if (&temp != nullptr) {
					mNode tempCopy = temp.copy();
					copy.next = &tempCopy;
				}
			}
			printf("About to copy TDD edge node value p3\n");
			copy.ref = ref;
			copy.v = v;
			copy.flags = flags;

			return copy;
		}

	};
	using mEdge = Edge<mNode>;
	using mCachedEdge = CachedEdge<mNode>;

} // namespace dd


