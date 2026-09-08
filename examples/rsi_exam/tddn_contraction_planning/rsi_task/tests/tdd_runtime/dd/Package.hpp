#pragma once

#include "Complex.hpp"
#include "ComplexCache.hpp"
#include "ComplexNumbers.hpp"
#include "ComplexTable.hpp"
#include "ComplexValue.hpp"
#include "ComputeTable.hpp"
#include "Control.hpp"
#include "Definitions.hpp"
#include "Edge.cpp"
#include "GateMatrixDefinitions.hpp"
#include "Package_fwd.hpp"

#include "UniqueTable.hpp"

#include "Tdd.hpp"
#include "Tensor.hpp"


#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <queue>
#include <random>
#include <regex>
#include <set>
#include <stack>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <numeric>

#include <xtensor/xarray.hpp>
#include <xtensor/xshape.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xslice.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xview.hpp>

namespace dd {

	template <class Config> class Package {
		static_assert(std::is_base_of_v<DDPackageConfig, Config>, "Config must be derived from DDPackageConfig");

		///
		/// Complex number handling
		///
	public:
		ComplexNumbers cn{};

		///
		/// Construction, destruction, information and reset
		///

		static constexpr std::size_t MAX_POSSIBLE_QUBITS = static_cast<std::make_unsigned_t<Qubit>>(std::numeric_limits<Qubit>::max()) + 1U;
		static constexpr std::size_t DEFAULT_QUBITS = 300;


		bool to_test = true;

		std::map<std::string, int> varOrder;


		explicit Package(std::size_t nq = DEFAULT_QUBITS) : nqubits(nq) {
			resize(nq);
		};
		~Package() = default;
		Package(const Package& package) = delete;

		Package& operator=(const Package& package) = delete;

		// resize the package instance
		void resize(std::size_t nq) {
			if (nq > MAX_POSSIBLE_QUBITS) {
				throw std::invalid_argument("Requested too many qubits from package. "
					"Qubit datatype only allows up to " +
					std::to_string(MAX_POSSIBLE_QUBITS) +
					" qubits, while " + std::to_string(nq) +
					" were requested. Please recompile the "
					"package with a wider Qubit type!");
			}
			nqubits = nq;
			nodeUniqueTable.resize(nqubits);
		}

		// reset package state
		void reset() {
			clearUniqueTables();
			clearComputeTables();
			cn.clear();
		}

		// getter for qubits
		[[nodiscard]] auto qubits() const { return nqubits; }

	private:
		std::size_t nqubits;

		///
		/// Vector nodes, edges and quantum states
		///
	public:


		//template <class Node>

		//TDD Tensor_2_TDD(const Tensor tn)
		//{

		//	TDD res;

		//	if()




		//	return res;
		//}

		//bool check_edges_equal(std::vector<Edge<mNode>> edges) {
		//	auto verify_p = edges[0].p;
		//	auto verify_w = edges[0].w;

		//	for (auto& edge : edges) {
		//		if (edge.p != verify_p) {
		//			return false;
		//		}
		//		if (edge.w != verify_w) {
		//			return false;
		//		}
		//	}
		//	return true;
		//}

		Edge<mNode> xarray_2_edge(
			const xt::xarray<ComplexValue>& array,
			const std::vector<int>& order = {})
{
			std::size_t sum_of_dim = std::accumulate(array.shape().begin(),
				array.shape().end(),
				0);

			if (sum_of_dim == array.dimension()) {
				std::vector<Edge<mNode>> edges;
				for (auto num : array) {
					if (num == complex_zero) {
						edges.push_back(Edge<mNode>::zero);
					}
					else if (num == complex_one) {
						edges.push_back(Edge<mNode>::one);
					}
					else {
						edges.push_back(Edge<mNode>::terminal(cn.lookup(num)));
					}
				}
				auto dd_node = makeDDNode(-1, edges, false);
				return dd_node;

			}

			std::vector<int> true_order = order;
			if (order.empty()) {
				// list(range(dim))
				std::vector<int> ind_order(array.dimension());
				//std::vector<std::size_t> order = order(array.dimension());
				for (int i = 0; i < array.dimension(); i++) {
					ind_order[i] = i;
				}
				//std::iota(order.begin(), order.end(), 0);
				true_order = ind_order;
			}

			auto split_pos = std::max_element(true_order.begin(), true_order.end()) - true_order.begin();
			Qubit x = true_order[split_pos];
			true_order[split_pos] = -1;
			//std::vector<xt::xarray<ComplexValue>> split_U = xt::split(array, array.shape(split_pos), split_pos);
			auto split_U = xt::split(array, array.shape(split_pos), split_pos);


			std::vector<Edge<mNode>> edges;
			for (auto u : split_U) {
				auto new_node = xarray_2_edge(u, true_order);
				edges.push_back(new_node);
			}

			return makeDDNode((Qubit)x, edges, false);

		}

		std::vector<std::string> generate_key(std::vector<Index> var) {
			//std::vector<std::string> res(var.size());
			//int size = (*max_element(std::begin(var), std::end(var))).idx;
			//std::vector<std::string> res(size + 1);
			std::vector<std::string> res(0);
			for (auto& index : var) {
				res.push_back(index.key);
				//res.at(index.idx) = index.key;
			}
			return res;
		}

		TDD Tensor_2_TDD(const Tensor tn) {
			if (tn.data.dimension() != tn.index_set.size()) {
				throw "action non definies";
			}

			TDD res;
			res.e = xarray_2_edge(tn.data);
			res.index_set = tn.index_set;
			res.key_2_index = generate_key(tn.index_set);

			return res;
		}



		TDD Matrix2TDD(const GateMatrix mat, std::vector<Index> var_out)
		{

			TDD low, high, res;
			Edge<mNode> e_temp[4];

			std::vector<Edge<mNode>> e_low(2), e_high(2), e(2);

			int Radix = 2;

			for (int i = 0; i < Radix; i++) {
				for (int j = 0; j < Radix; j++) {
					if (mat[2 * i + j] == complex_zero) {
						e_temp[i * Radix + j] = Edge<mNode>::zero;
					}
					else if (mat[2 * i + j] == complex_one) {
						e_temp[i * Radix + j] = Edge<mNode>::one;
					}
					else {
						e_temp[i * Radix + j] = Edge<mNode>::terminal(cn.lookup(mat[2 * i + j]));
					}
				}
			}


			std::vector<std::string> key_2_index;
			std::vector<Index> ordered_index_set;
			if (varOrder[var_out[0].key] < varOrder[var_out[1].key]) {
				e_low[0] = e_temp[0];
				e_low[1] = e_temp[1];
				e_high[0] = e_temp[2];
				e_high[1] = e_temp[3];
				key_2_index = { var_out[1].key,var_out[0].key };
				ordered_index_set = { var_out[1], var_out[0] };
			}
			else {
				e_low[0] = e_temp[0];
				e_low[1] = e_temp[2];
				e_high[0] = e_temp[1];
				e_high[1] = e_temp[3];
				key_2_index = { var_out[0].key,var_out[1].key };
				ordered_index_set = { var_out[0], var_out[1] };
			}


			if (e_low[0].p == e_low[1].p and e_low[0].w == e_low[1].w) {
				low.e = e_low[0];
			}
			else {
				low.e = makeDDNode(0, e_low, false);
			}
			if (e_high[0].p == e_high[1].p and e_high[0].w == e_high[1].w) {
				high.e = e_high[0];
			}
			else {
				high.e = makeDDNode(0, e_high, false);
			}
			if (low.e.p == high.e.p and low.e.w == high.e.w) {
				res.e = low.e;
			}
			else {
				e[0] = low.e;
				e[1] = high.e;
				res.e = makeDDNode(1, e, false); ;
			}
			res.index_set = ordered_index_set;
			res.key_2_index = key_2_index;
			return res;
		}
		//template <class Node>
		TDD diag_matrix_2_TDD(const GateMatrix mat, std::vector<Index> var_out)
		{

			TDD res;
			int Radix = 2;

			std::vector<Edge<mNode>> e_temp(2);
			for (int i = 0; i < Radix; i++) {

				if (mat[2 * i + i] == complex_zero) {
					e_temp[i] = Edge<mNode>::zero;
				}
				else {
					if (mat[2 * i + i] == complex_one) {
						e_temp[i] = Edge<mNode>::one;
					}
					else {
						e_temp[i] = Edge<mNode>::terminal(cn.lookup(mat[2 * i + i]));
					}
				}
			}

			res.e = makeDDNode(0, e_temp, false);
			res.index_set = var_out;
			res.key_2_index = { var_out[0].key };
			return res;
		}

		TDD cgate_2_TDD(std::vector<Index> var, std::string gate_type, bool debug = true) {
			// New
			TDD low, high, res;
			std::vector<Edge<mNode>> e(2), e_low(2), e_high(2);

			bool flipped = varOrder[var[0].key] > varOrder[var[2].key];

			std::vector<int> offsetList = {0, 0, 0, 0};
			for (int i = 0; i < offsetList.size(); i++) {
				int offset = 0;

				for (int j = 0; j < offsetList.size(); j++) {
					if (varOrder[var[i].key] < varOrder[var[j].key]) {
						offset++;
					} 
				}

				offsetList[i] += offset;
			}

			std::vector<Index> orderedVars(4);
			for (int i = 0; i < orderedVars.size(); i++) {
				orderedVars[offsetList[i]] = var[i];
			}

			if (to_test) {
				for (int i = 0; i < orderedVars.size(); i++) {
					printf("OrderedVars: %s\n", orderedVars[i].key.c_str());
				}
			}

			dd::GateMatrix high_gate = Xmat;
			if (!gate_type.compare("z")) {
				high_gate = Zmat;
			} else if (!gate_type.compare("y")) {
				high_gate = FYmat;
			} else {
				if (gate_type.compare("x")) {
					printf("Gate type %s not supported, defaulting to cx gate", gate_type.c_str());
				}
			}

			if (!flipped || !gate_type.compare("z")) {
				if (to_test)
					printf("Not flipped or cz\n");
				// Create CX/CNOT from I and X gate
				// Var[0] and var[1] must be the two least values according to varOrder
				low = Matrix2TDD(Imat, { orderedVars[0] ,orderedVars[1] });
				high = Matrix2TDD(high_gate, { orderedVars[0] ,orderedVars[1] });
				
				// Prepare the additional layer (identity-ish layer) between root node and I/X TDDs
				e_low[0] = low.e;
				e_low[1] = Edge<mNode>::zero;
				e_high[0] = Edge<mNode>::zero;
				e_high[1] = high.e;
				e[0] = makeDDNode(2, e_low, false);
				e[1] = makeDDNode(2, e_high, false);
			} else if (!gate_type.compare("x")) {
				if (to_test)
					printf("Flipped cx\n");
				// Create CX/CNOT from I and X gate
				// Var[0] and var[1] must be the two least values according to varOrder
				low = Matrix2TDD(LLmat(complex_one), { orderedVars[0] ,orderedVars[1] });
				high = Matrix2TDD(HHmat(complex_one), { orderedVars[0] ,orderedVars[1] });
				
				// Prepare the additional layer (identity-ish layer) between root node and I/X TDDs
				e_low[0] = low.e;
				e_low[1] = high.e;
				e_high[0] = high.e;
				e_high[1] = low.e;
				e[0] = makeDDNode(2, e_low, false);
				e[1] = makeDDNode(2, e_high, false);
			} else if (!gate_type.compare("y")) {
				if (to_test)
					printf("Flipped cy\n");
				// Create CX/CNOT from I and X gate
				// Var[0] and var[1] must be the two least values according to varOrder
				low = Matrix2TDD(LLmat(complex_one), { orderedVars[0] ,orderedVars[1] });
				high = Matrix2TDD(HHmat(complex_i), { orderedVars[0] ,orderedVars[1] });
				
				// Prepare the additional layer (identity-ish layer) between root node and I/X TDDs
				e_low[0] = low.e;
				e_low[1] = high.e;
				high = Matrix2TDD(HHmat(complex_mi), { orderedVars[0] ,orderedVars[1] });
				e_high[0] = high.e;
				e_high[1] = low.e;
				e[0] = makeDDNode(2, e_low, false);
				e[1] = makeDDNode(2, e_high, false);
			} 

			// Create resulting CNOT node
			res.e = makeDDNode(3, e, false);
			res.index_set = { orderedVars[0],orderedVars[1],orderedVars[2],orderedVars[3] };

			// Attach the two missing indices (again in increasing order according to varOrder)
			low.key_2_index.push_back(orderedVars[2].key);
			low.key_2_index.push_back(orderedVars[3].key);
			res.key_2_index = low.key_2_index;

			// for (int i = 0; i < res.index_set.size(); i++) {
			// 	printf("Index set: %s\n", res.index_set[i].key.c_str());
			// }

			if (to_test) {
				for (int i = 0; i < low.key_2_index.size(); i++) {
					printf("K2Idx: %s\n", low.key_2_index[i].c_str());
				}
				printf("\n");
			}

			return res;
		}

		TDD split_cgate_2_TDD(std::vector<Index> var, std::string gate_type, bool is_control, bool debug = true) {
			// New
			TDD low, high, res;
			std::vector<Edge<mNode>> e(2);

			bool flipped = (varOrder[var[0].key] > varOrder[var[2].key] && is_control) || (varOrder[var[0].key] < varOrder[var[2].key] && !is_control);

			std::vector<int> offsetList = {0, 0, 0};
			for (int i = 0; i < offsetList.size(); i++) {
				int offset = 0;

				for (int j = 0; j < offsetList.size(); j++) {
					if (varOrder[var[i].key] < varOrder[var[j].key]) {
						offset++;
					} 
				}

				offsetList[i] += offset;
			}

			std::vector<Index> orderedVars(3);
			for (int i = 0; i < orderedVars.size(); i++) {
				orderedVars[offsetList[i]] = var[i];
			}

			if (to_test) {
				for (int i = 0; i < orderedVars.size(); i++) {
					printf("OrderedVars: %s\n", orderedVars[i].key.c_str());
				}
			}

			dd::GateMatrix high_gate = Xmat;
			if (!gate_type.compare("z")) {
				high_gate = Zmat;
			} else if (!gate_type.compare("y")) {
				if (!flipped)
					high_gate = FYmat;
				else
					high_gate = Ymat;
			} else {
				if (gate_type.compare("x")) {
					printf("Gate type %s not supported, defaulting to cx gate", gate_type.c_str());
				}
			}

			bool is_cz = !gate_type.compare("z");
			// (is_control && (!is_cz || !flipped)) || (!is_control && is_cz && flipped)
			if ((is_control && (!is_cz || !flipped)) || (!is_control && is_cz && flipped)) {
				if (to_test)
					printf("Not flipped\n");
				// Create CX/CNOT from I and X gate
				// Var[0] and var[1] must be the two least values according to varOrder
				low = Matrix2TDD(LLmat(complex_one), { orderedVars[0] ,orderedVars[1] });
				high = Matrix2TDD(HHmat(complex_one), { orderedVars[0] ,orderedVars[1] });
				
				// Prepare the additional layer (identity-ish layer) between root node and I/X TDDs
				e[0] = low.e;
				e[1] = high.e;
			} else {
				low = Matrix2TDD(Imat, { orderedVars[0] ,orderedVars[1] });
				high = Matrix2TDD(high_gate, { orderedVars[0] ,orderedVars[1] });
				
				// Prepare the additional layer (identity-ish layer) between root node and I/X TDDs
				e[0] = low.e;
				e[1] = high.e;
			}

			// Create resulting CNOT node
			res.e = makeDDNode(2, e, false);
			res.index_set = { orderedVars[0],orderedVars[1],orderedVars[2] };

			// Attach the two missing indices (again in increasing order according to varOrder)
			low.key_2_index.push_back(orderedVars[2].key);
			res.key_2_index = low.key_2_index;

			// for (int i = 0; i < res.index_set.size(); i++) {
			// 	printf("Index set: %s\n", res.index_set[i].key.c_str());
			// }

			if (to_test) {
				for (int i = 0; i < low.key_2_index.size(); i++) {
					printf("K2Idx: %s\n", low.key_2_index[i].c_str());
				}
				printf("\n");
			}

			return res;
		}

		//template <class Node>
		TDD cnot_2_TDD(std::vector<Index> var, int ca = 1, bool debug = false) {
			// New
			TDD low, high, res;
			std::vector<Edge<mNode>> e(2), e_low(2), e_high(2);

			std::vector<int> largerList = {0, 0, 0, 0};
			for (int i = 0; i < largerList.size(); i++) {
				int largerThan = 0;

				for (int j = 0; j < largerList.size(); j++) {
					if (varOrder[var[i].key] > varOrder[var[j].key]) {
						largerThan++;
					}
				}

				largerList[i] += largerThan;
			}

			std::vector<Index> orderedVars(4);
			for (int i = 0; i < orderedVars.size(); i++) {
				orderedVars[largerList[i]] = var[i];
			}

			if (to_test) {
				for (int i = 0; i < orderedVars.size(); i++) {
					printf("OrderedVars: %s\n", orderedVars[i].key.c_str());
				}
			}

			// Create CX/CNOT from I and X gate
			// Var[0] and var[1] must be the two least values according to varOrder
			low = Matrix2TDD(Imat, { orderedVars[0] ,orderedVars[1] });
			high = Matrix2TDD(Xmat, { orderedVars[0] ,orderedVars[1] });
			
			// Prepare the additional layer (identity-ish layer) between root node and I/X TDDs
			e_low[0] = low.e;
			e_low[1] = Edge<mNode>::zero;
			e_high[0] = Edge<mNode>::zero;
			e_high[1] = high.e;
			e[0] = makeDDNode(2, e_low, false);
			e[1] = makeDDNode(2, e_high, false);
			
			// Create resulting CNOT node
			res.e = makeDDNode(3, e, false);
			res.index_set = { orderedVars[0],orderedVars[1],orderedVars[2],orderedVars[3] };

			// Attach the two missing indices (again in increasing order according to varOrder)
			low.key_2_index.push_back(orderedVars[2].key);
			low.key_2_index.push_back(orderedVars[3].key);
			res.key_2_index = low.key_2_index;

			if (to_test) {
				for (int i = 0; i < low.key_2_index.size(); i++) {
					printf("K2Idx: %s\n", low.key_2_index[i].c_str());
				}
				printf("\n");
			}

			return res;
			// Old

			// TDD low, high, res;
			// std::vector<Edge<mNode>> e(2), e_low(2), e_high(2);
			// if (ca == 1) {
			// 	if (varOrder[var[0].key] < varOrder[var[2].key] && varOrder[var[0].key] < varOrder[var[3].key]) {
			// 		printf("CNOT case 1\n\n");
			// 		low = Matrix2TDD(Imat, { var[0] ,var[1] });
			// 		high = Matrix2TDD(Xmat, { var[0] ,var[1] });
			// 		e_low[0] = low.e;
			// 		e_low[1] = Edge<mNode>::zero;
			// 		e_high[0] = Edge<mNode>::zero;
			// 		e_high[1] = high.e;
			// 		e[0] = makeDDNode(2, e_low, false);
			// 		e[1] = makeDDNode(2, e_high, false);
			// 		res.e = makeDDNode(3, e, false);
			// 		res.index_set = { var[0],var[1],var[2],var[3] };
			// 		low.key_2_index.push_back(var[2].key);
			// 		low.key_2_index.push_back(var[3].key);
			// 		for (int i = 0; i < low.key_2_index.size(); i++) {
			// 			printf("K2Idx: %s\n", low.key_2_index[i].c_str());
			// 		}
			// 		printf("\n");
			// 		res.key_2_index = low.key_2_index;
			// 	}
			// 	else if (varOrder[var[2].key] > varOrder[var[0].key] && varOrder[var[2].key] > varOrder[var[3].key]) {
			// 		printf("CNOT case 2\n");
			// 		low = Matrix2TDD(Imat, { var[0] ,var[3] });
			// 		high = Matrix2TDD(Xmat, { var[0] ,var[3] });
			// 		e_low[0] = low.e;
			// 		e_low[1] = Edge<mNode>::zero;
			// 		e_high[0] = Edge<mNode>::zero;
			// 		e_high[1] = high.e;
			// 		e[0] = makeDDNode(2, e_low, false);
			// 		e[1] = makeDDNode(2, e_high, false);
			// 		res.e = makeDDNode(3, e, false);
			// 		res.index_set = { var[0],var[1],var[2],var[3] };
			// 		low.key_2_index.push_back(var[2].key);
			// 		res.key_2_index = low.key_2_index;
			// 	}
			// 	else {
			// 		printf("CNOT case 3\n");
			// 		low = Matrix2TDD(Imat, { var[0] ,var[2] });
			// 		high = Matrix2TDD(Xmat, { var[0] ,var[2] });
			// 		e_low[0] = low.e;
			// 		e_low[1] = Edge<mNode>::zero;
			// 		e_high[0] = Edge<mNode>::zero;
			// 		e_high[1] = high.e;
			// 		e[0] = makeDDNode(2, e_low, false);
			// 		e[1] = makeDDNode(2, e_high, false);
			// 		res.e = makeDDNode(3, e, false);
			// 		res.index_set = { var[0],var[1],var[2],var[3] };
			// 		low.key_2_index.push_back(var[3].key);
			// 		res.key_2_index = low.key_2_index;
			// 	}
			// 	return res;
			// }
			// return res;

		}

		// TODO: adapt
		TDD cy_2_TDD(std::vector<Index> var, int ca = 1) {


			TDD low, high, res;
			std::vector<Edge<mNode>> e(2);
			if (ca == 1) {
				if (varOrder[var[0].key] > varOrder[var[3].key] && varOrder[var[0].key] > varOrder[var[4].key]) {
					low = Matrix2TDD(Imat, { var[3] ,var[4] });
					high = Matrix2TDD(Ymat, { var[3] ,var[4] });
					e[0] = low.e;
					e[1] = high.e;
					res.e = makeDDNode(2, e, false);
					res.index_set = { var[0],var[2],var[3],var[4] };
					low.key_2_index.push_back(var[0].key);
					res.key_2_index = low.key_2_index;
				}
				else if (varOrder[var[3].key] > varOrder[var[0].key] && varOrder[var[3].key] > varOrder[var[4].key]) {
					low = Matrix2TDD(Imat, { var[0] ,var[4] });
					high = Matrix2TDD(Ymat, { var[0] ,var[4] });
					e[0] = low.e;
					e[1] = high.e;
					res.e = makeDDNode(2, e, false);
					res.index_set = { var[0],var[2],var[3],var[4] };
					low.key_2_index.push_back(var[3].key);
					res.key_2_index = low.key_2_index;
				}
				else {
					low = Matrix2TDD(Imat, { var[0] ,var[3] });
					high = Matrix2TDD(Ymat, { var[0] ,var[3] });
					e[0] = low.e;
					e[1] = high.e;
					res.e = makeDDNode(2, e, false);
					res.index_set = { var[0],var[2],var[3],var[4] };
					low.key_2_index.push_back(var[4].key);
					res.key_2_index = low.key_2_index;
				}
				return res;
			}
			return res;

		}

		// TODO: adapt
		TDD cz_2_TDD(std::vector<Index> var, int ca = 1) {


			TDD low, high, res;
			std::vector<Edge<mNode>> e(2);
			if (ca == 1) {
				if (varOrder[var[0].key] > varOrder[var[3].key] && varOrder[var[0].key] > varOrder[var[4].key]) {
					low = Matrix2TDD(Imat, { var[3] ,var[4] });
					high = Matrix2TDD(Zmat, { var[3] ,var[4] });
					e[0] = low.e;
					e[1] = high.e;
					res.e = makeDDNode(2, e, false);
					res.index_set = { var[0],var[2],var[3],var[4] };
					low.key_2_index.push_back(var[0].key);
					res.key_2_index = low.key_2_index;
				}
				else if (varOrder[var[3].key] > varOrder[var[0].key] && varOrder[var[3].key] > varOrder[var[4].key]) {
					low = Matrix2TDD(Imat, { var[0] ,var[4] });
					high = Matrix2TDD(Zmat, { var[0] ,var[4] });
					e[0] = low.e;
					e[1] = high.e;
					res.e = makeDDNode(2, e, false);
					res.index_set = { var[0],var[2],var[3],var[4] };
					low.key_2_index.push_back(var[3].key);
					res.key_2_index = low.key_2_index;
				}
				else {
					low = Matrix2TDD(Imat, { var[0] ,var[3] });
					high = Matrix2TDD(Zmat, { var[0] ,var[3] });
					e[0] = low.e;
					e[1] = high.e;
					res.e = makeDDNode(2, e, false);
					res.index_set = { var[0],var[2],var[3],var[4] };
					low.key_2_index.push_back(var[4].key);
					res.key_2_index = low.key_2_index;
				}
				return res;
			}
			return res;

		}


		///
		/// Matrix nodes, edges and quantum gates
		///
		template <class Node> Edge<Node> normalize(const Edge<Node>& e, bool cached) {

			auto argmax = -1;

			//auto zero = std::array{	e.p->e[0].w.approximatelyZero(), e.p->e[1].w.approximatelyZero()};
			int R = e.p->e.size();
			std::vector<bool> zero;
			for (int k = 0; k < R; k++) {
				zero.push_back(e.p->e[k].w.approximatelyZero());
			}

			// make sure to release cached numbers approximately zero, but not exactly
			// zero
			if (cached) {
				for (auto i = 0U; i < R; i++) {
					if (zero[i] && e.p->e[i].w != Complex::zero) {
						cn.returnToCache(e.p->e[i].w);
						e.p->e[i] = Edge<Node>::zero;
					}
				}
			}

			fp max = 0;
			auto maxc = Complex::one;
			// determine max amplitude
			for (auto i = 0U; i < R; ++i) {
				if (zero[i]) {
					continue;
				}
				if (argmax == -1) {
					argmax = static_cast<decltype(argmax)>(i);
					max = ComplexNumbers::mag2(e.p->e[i].w);
					maxc = e.p->e[i].w;
				}
				else {
					auto mag = ComplexNumbers::mag2(e.p->e[i].w);
					if (mag - max > ComplexTable<>::tolerance()) {
						argmax = static_cast<decltype(argmax)>(i);
						max = mag;
						maxc = e.p->e[i].w;
					}
				}
			}

			// all equal to zeroA
			if (argmax == -1) {
				if (!cached && !e.isTerminal()) {
					// If it is not a cached computation, the node has to be put back into
					// the chain
					getUniqueTable<Node>().returnNode(e.p);
				}
				return Edge<Node>::zero;
			}

			auto r = e;
			// divide each entry by max
			for (auto i = 0U; i < R; ++i) {
				if (static_cast<decltype(argmax)>(i) == argmax) {
					if (cached) {
						if (r.w.exactlyOne()) {
							r.w = maxc;
						}
						else {
							if (r.w.exactlyZero()) {
								printf("L528");
							}
							ComplexNumbers::mul(r.w, r.w, maxc);
						}
					}
					else {
						if (r.w.exactlyOne()) {
							r.w = maxc;
						}
						else {
							auto c = cn.getTemporary();
							ComplexNumbers::mul(c, r.w, maxc);
							r.w = cn.lookup(c);
						}
					}
					r.p->e[i].w = Complex::one;
				}
				else {
					if (zero[i]) {
						if (cached && r.p->e[i].w != Complex::zero) {
							cn.returnToCache(r.p->e[i].w);
						}
						r.p->e[i] = Edge<Node>::zero;
						continue;
					}
					if (cached && !zero[i] && !r.p->e[i].w.exactlyOne()) {
						cn.returnToCache(r.p->e[i].w);
					}
					if (r.p->e[i].w.approximatelyOne()) {
						r.p->e[i].w = Complex::one;
					}
					auto c = cn.getTemporary();
					ComplexNumbers::div(c, r.p->e[i].w, maxc);
					r.p->e[i].w = cn.lookup(c);
				}
			}

			if (r.p->e.size() == 1 && dd::mNode::isTerminal(r.p->e[0].p))
				r.p = Edge<Node>::one.p;


				//std::cout << typeid(r.p->e[0].p).name() << std::endl;

			return r;

		}


	private:

		///
		/// Unique tables, Reference counting and garbage collection
		///
	public:
		// unique tables
		template <class Node> [[nodiscard]] auto& getUniqueTable() {
			return nodeUniqueTable;
		}

		template <class Node> void incRef(const Edge<Node>& e) {
			getUniqueTable<Node>().incRef(e);
		}
		template <class Node> void decRef(const Edge<Node>& e) {
			getUniqueTable<Node>().decRef(e);
		}

		UniqueTable<mNode, Config::UT_MAT_NBUCKET, Config::UT_MAT_INITIAL_ALLOCATION_SIZE>	nodeUniqueTable{nqubits};

		bool garbageCollect(bool force = false) {
			// return immediately if no table needs collection
			if (!force &&
				!nodeUniqueTable.possiblyNeedsCollection() &&
				!cn.complexTable.possiblyNeedsCollection()) {
				return false;
			}

			auto cCollect = cn.garbageCollect(force);
			if (cCollect > 0) {
				// Collecting garbage in the complex numbers table requires collecting the
				// node tables as well
				force = true;
			}

			auto mCollect = nodeUniqueTable.garbageCollect(force);

			// invalidate all compute tables where any component of the entry contains
			// numbers from the complex table if any complex numbers were collected
			if (mCollect > 0) {

				addTable.clear();
				contTable.clear();
			}
			return  mCollect > 0;
		}

		void clearUniqueTables() {
			nodeUniqueTable.clear();

		}

		// create a normalized DD node and return an edge pointing to it. The node is
		// not recreated if it already exists.
		template <class Node>
		Edge<Node> makeDDNode(
			Qubit var,
			const std::vector<Edge<Node>>& edges,
			bool cached = false) {

			auto& uniqueTable = getUniqueTable<Node>();
			Edge<Node> e{uniqueTable.getNode(), Complex::one};
			e.p->v = var;
			e.p->e = edges;

			assert(e.p->ref == 0);

			bool all_equal = true;
			for (int k = 0; k < edges.size(); k++) {
				if (edges[k].p != edges[0].p || !e.p->e[k].w.approximatelyEquals(e.p->e[0].w)) {
					all_equal = false;
					break;
				}

			}

			if (all_equal) {
				if (cached) {
					for (int k = 1; k < edges.size(); k++) {
						if (e.p->e[k].w != Complex::zero && e.p->e[k].w != Complex::one) {
							cn.returnToCache(e.p->e[k].w);
							return edges[0];
						}
					}
					return edges[0];
				}
			}

			//if (edges[0].p == edges[1].p && e.p->e[0].w.approximatelyEquals(e.p->e[1].w)) {
			//	if (cached) {
			//		if (e.p->e[1].w != Complex::zero) {
			//			cn.returnToCache(e.p->e[1].w);
			//			return edges[0];
			//		}

			//		return edges[0];

			//	}
			//	return edges[0];
			//}

			e = normalize(e, cached);

			assert(e.p->v == var || e.isTerminal());

			// look it up in the unique tables
			auto l = uniqueTable.lookup(e, false);

			assert(l.p->v == var || l.isTerminal());

			return l;
		}


		///
		/// Compute table definitions
		///
	public:
		void clearComputeTables() {

		}



	public:


		ComputeTable<mCachedEdge, mCachedEdge, mCachedEdge, Config::CT_VEC_ADD_NBUCKET>	addTable{};
		ComputeTable2<mEdge, mEdge, mCachedEdge, Config::CT_MAT_MAT_MULT_NBUCKET>  contTable{};

		key_2_new_key_node key_2_new_key_tree_header_element = { -1,-1,{},nullptr };
		key_2_new_key_node* key_2_new_key_tree_header = &key_2_new_key_tree_header_element;

		key_2_new_key_node* append_new_key(key_2_new_key_node* self, float new_key) {

			auto it = self->next.find(new_key);
			
			if (it != self->next.end()) {
				return self->next[new_key];
			}
			else {
				self->next[new_key] = new key_2_new_key_node{ ((short)(self->level + 1)), new_key, {}, self };

				return self->next[new_key];
			}
		}

		void print_new_key_node_as_father(key_2_new_key_node* self) {
			std::string key_str = "Key node: ";
			key_2_new_key_node* curr_key = self;
			while (self->next.size() > 0) {
				assert(self->next.size() == 1);
				for (const auto& [key, value] : self->next) {
					key_str += std::to_string(key) + ", ";
					curr_key = value;
				}
			}
			printf("%s\n", key_str.c_str());
		}

		void print_new_key_node_as_son(key_2_new_key_node* self) {
			std::string key_str = "";
			key_2_new_key_node* curr_key = self;
			while (curr_key->father) {
				key_str = std::to_string(curr_key->new_key).c_str() + std::string(", ") + key_str;
				curr_key = curr_key->father;
			}
			printf("Key node: %s\n", key_str.c_str());
		}

		template <class Edge> Edge T_add(const Edge& x, const Edge& y) {

			return y;
		}


		//template <class LeftOperand, class RightOperand>
		TDD cont(TDD tdd1, TDD tdd2) {

			TDD res;

			std::vector<Index> var_out;
			std::vector<std::string> var_cont_temp;
			std::vector<std::string> var_cont;
			std::vector<std::string> var_out_key;

			int k;
			int k1;
			for (k = 0; k < tdd1.index_set.size(); ++k) {
				bool flag = true;

				for (k1 = 0; k1 < tdd2.index_set.size(); ++k1) {
					if (tdd2.index_set[k1].idx == tdd1.index_set[k].idx && tdd2.index_set[k1].key == tdd1.index_set[k].key) {
						var_cont_temp.push_back(tdd1.index_set[k].key);
						flag = false;
						break;
					}
				}
				if (flag) {
					var_out.push_back(tdd1.index_set[k]);
					var_out_key.push_back(tdd1.index_set[k].key);
				}
			}

			for (k = 0; k < tdd2.index_set.size(); ++k) {
				bool flag = true;
				for (k1 = 0; k1 < tdd1.index_set.size(); ++k1) {
					if (tdd1.index_set[k1].idx == tdd2.index_set[k].idx && tdd1.index_set[k1].key == tdd2.index_set[k].key) {
						flag = false;
						break;
					}
				}
				if (flag) {
					var_out.push_back(tdd2.index_set[k]);
					var_out_key.push_back(tdd2.index_set[k].key);
				}
			}
			for (k = 0; k < var_cont_temp.size(); ++k) {
				if (find(var_out_key.begin(), var_out_key.end(), var_cont_temp[k]) == var_out_key.end()) {
					if (find(var_cont.begin(), var_cont.end(), var_cont_temp[k]) == var_cont.end()) {
						if (to_test)
							printf("Var cont key %d: %s at %d\n", k, var_cont_temp[k].c_str(), varOrder[var_cont_temp[k]]);
						var_cont.push_back(var_cont_temp[k]);
					}
				}
			}

			if (to_test) {
				for (k = 0; k < var_out_key.size(); k++) {
					printf("Var out key %d: %s at %d\n", k, var_out_key[k].c_str(), varOrder[var_out_key[k]]);
				}
				printf("\n");
			}

			if (to_test) {
				std::cout << "TDD1: ";
				for (const auto& element : tdd1.key_2_index) {
					std::cout << element << " ";
				}
				std::cout << std::endl;
				std::cout << "TDD2: ";
				for (const auto& element : tdd2.key_2_index) {
					std::cout << element << " ";
				}
				std::cout << std::endl;

			}


			key_2_new_key_node* key_2_new_key1 = key_2_new_key_tree_header;
			key_2_new_key_node* key_2_new_key2 = key_2_new_key_tree_header;

			std::vector<std::string> new_key_2_index;
			k1 = 0;
			int k2 = 0;
			int new_key = 0;
			int m1 = tdd1.key_2_index.size();
			int m2 = tdd2.key_2_index.size();
			int repeat_time = 0;
			float last_cont_idx = -2;

			while (k1 < m1 || k2 < m2) {

				if (k1 == m1) {
					for (k2; k2 < m2; ++k2)
					{
						key_2_new_key2 = append_new_key(key_2_new_key2, new_key);
						new_key_2_index.push_back(tdd2.key_2_index[k2]);
						new_key++;
					}
					break;
				}
				if (k2 == m2) {
					for (k1; k1 < m1; ++k1)
					{
						key_2_new_key1 = append_new_key(key_2_new_key1, new_key);
						new_key_2_index.push_back(tdd1.key_2_index[k1]);
						new_key++;
					}
					break;
				}

				if (varOrder[tdd1.key_2_index[k1]] > varOrder[tdd2.key_2_index[k2]]) {
					key_2_new_key1 = append_new_key(key_2_new_key1, new_key);
					new_key_2_index.push_back(tdd1.key_2_index[k1]);
					new_key++;
					k1++;
					repeat_time++;
				}
				else if (varOrder[tdd1.key_2_index[k1]] < varOrder[tdd2.key_2_index[k2]]) {
					key_2_new_key2 = append_new_key(key_2_new_key2, new_key);
					new_key_2_index.push_back(tdd2.key_2_index[k2]);
					new_key++;
					k2++;
					repeat_time++;
				}
				else if (find(var_out_key.begin(), var_out_key.end(), tdd1.key_2_index[k1]) == var_out_key.end()) {
					// if (new_key - last_cont_idx <= 0.5) {
					// 	last_cont_idx = last_cont_idx + 1 / (3 * nqubits) * repeat_time;
					// 	repeat_time += 1;
					// 	key_2_new_key1 = append_new_key(key_2_new_key1, last_cont_idx);
					// 	key_2_new_key2 = append_new_key(key_2_new_key2, last_cont_idx);
					// 	k1++;
					// 	k2++;
					// }
					// else {
					// 	float intermediate_key = new_key - 0.5f + ((repeat_time + 1) * 0.0001f);
					// 	key_2_new_key1 = append_new_key(key_2_new_key1, intermediate_key);
					// 	key_2_new_key2 = append_new_key(key_2_new_key2, intermediate_key);
					// 	last_cont_idx = intermediate_key;
					// 	repeat_time += 1;
					// 	k1++;
					// 	k2++;
					// }					
					// else {
					// 	key_2_new_key1 = append_new_key(key_2_new_key1, new_key - 0.5);
					// 	key_2_new_key2 = append_new_key(key_2_new_key2, new_key - 0.5);
					// 	last_cont_idx = new_key - 0.5;
					// 	repeat_time += 1;
					// 	k1++;
					// 	k2++;
					// }
					float intermediate_key = new_key - 0.5f + ((repeat_time + 1) * 0.0001f);
					key_2_new_key1 = append_new_key(key_2_new_key1, intermediate_key);
					key_2_new_key2 = append_new_key(key_2_new_key2, intermediate_key);
					k1++;
					k2++;
					repeat_time++;
				}
				else {
					key_2_new_key1 = append_new_key(key_2_new_key1, new_key);
					key_2_new_key2 = append_new_key(key_2_new_key2, new_key);
					new_key_2_index.push_back(tdd1.key_2_index[k1]);
					new_key++;
					k1++;
					k2++;
					repeat_time++;
				}
			}

			if (to_test) {
				print_new_key_node_as_son(key_2_new_key1);
				print_new_key_node_as_son(key_2_new_key2);
			}

			res.index_set = var_out;
			res.key_2_index = new_key_2_index;
			if (to_test) {
				printf("Res index set:\n");
				for (k = 0; k < res.index_set.size(); k++) {
					printf("    Var out key %d: %s\n", k, res.index_set[k].key.c_str());
				}
				printf("\n");

				printf("Res key2index:\n");
				for (k = 0; k < res.key_2_index.size(); k++) {
					printf("    Key2Index %d: %s\n", k, res.key_2_index[k].c_str());
				}
				printf("\n");
				printf("Starting cont2 now\n");
			}
			[[maybe_unused]] const auto before = cn.cacheCount();
			
			res.e = cont2(tdd1.e, tdd2.e, key_2_new_key1, key_2_new_key2, var_cont.size());

			if (to_test) {
				std::cout << "TDD: ";
				for (const auto& element : res.key_2_index) {
					std::cout << element << " ";
				}
				std::cout << std::endl;

			}

			var_out.clear();
			var_cont_temp.clear();
			var_out_key.clear();
			var_cont.clear();

			if (!res.e.w.exactlyZero() && !res.e.w.exactlyOne()) {
				cn.returnToCache(res.e.w);
				res.e.w = cn.lookup(res.e.w);
			}

			[[maybe_unused]] const auto after = cn.cacheCount();
			//assert(before == after);

			return res;
		}

		/*
		def is_tdd_identitiy(node, expected_length = -1):
			if type(node) == TDD.TDD:
				return is_node_identitiy(node.node, expected_length == -1, expected_length)
			if node == TDD.terminal_node:
				return expected_length == 0 or expected_length is None
			return False

		def is_node_identitiy(node, length_indifferent, expected_length):
			
			dd::mNode::isTerminal(r.p->e[0].p)
		*/
		bool isTDDIdentity(TDD tdd, bool lengthIndifferent, int expectedLength) {
			if (to_test)
				printf("Initial expectedLength = %d\n", expectedLength);
			bool isTerminalNode = dd::mNode::isTerminal(tdd.e.p);
			if (isTerminalNode) {
				return lengthIndifferent || expectedLength == 0;
			}
			return isNodeIdentity(tdd.e, lengthIndifferent, expectedLength);
		}
		


	private:

		template <class Node>
		Edge<Node> T_add2(const Edge<Node>& x, const Edge<Node>& y) {

			if (x.p > y.p) {
				return T_add2(y, x);
			}

			if (x.p == nullptr) {
				return y;
			}
			if (y.p == nullptr) {
				return x;
			}

			if (x.w.exactlyZero()) {
				if (y.w.exactlyZero()) {
					return Edge<Node>::zero;
				}
				auto r = y;
				r.w = cn.getCached(CTEntry::val(y.w.r), CTEntry::val(y.w.i));
				return r;
			}
			if (y.w.exactlyZero()) {
				auto r = x;
				r.w = cn.getCached(CTEntry::val(x.w.r), CTEntry::val(x.w.i));
				return r;
			}
			if (x.p == y.p) {
				auto r = y;
				r.w = cn.addCached(x.w, y.w);
				if (r.w.approximatelyZero()) {
					cn.returnToCache(r.w);
					return Edge<Node>::zero;
				}
				return r;
			}

			auto xCopy = x;
			auto yCopy = y;
			if (x.w != Complex::one) {
				xCopy.w = Complex::one;
				yCopy.w = cn.divCached(y.w, x.w);
			}


			auto r = addTable.lookup({ xCopy.p, xCopy.w }, { yCopy.p, yCopy.w });

			if (r.p != nullptr) {
				if (r.w.approximatelyZero()) {
					return Edge<Node>::zero;
				}
				auto c = cn.getCached(r.w);
				if (x.w != Complex::one) {
					cn.mul(c, c, x.w);
					cn.returnToCache(yCopy.w);
				}
				return { r.p, c };
			}

			const Qubit w = (x.isTerminal() || (!y.isTerminal() && y.p->v > x.p->v))
				? y.p->v
				: x.p->v;

			int n = (x.p->v != w)? y.p->e.size() : x.p->e.size();

			std::vector<Edge<Node>> edge(n);
			for (std::size_t i = 0U; i < n; i++) {
				Edge<Node> e1{};
				if (!x.isTerminal() && x.p->v == w) {
					e1 = x.p->e[i];
					if (e1.w != Complex::zero) {
						e1.w = cn.mulCached(e1.w, xCopy.w);
					}
				}
				else {
					e1 = xCopy;
					if (y.p->e[i].p == nullptr) {
						e1 = { nullptr, Complex::zero };
					}
				}
				Edge<Node> e2{};
				if (!y.isTerminal() && y.p->v == w) {
					e2 = y.p->e[i];

					if (e2.w != Complex::zero) {
						e2.w = cn.mulCached(e2.w, yCopy.w);
					}
				}
				else {
					e2 = yCopy;
					if (x.p->e[i].p == nullptr) {
						e2 = { nullptr, Complex::zero };
					}
				}


				edge[i] = T_add2(e1, e2);


				if (!x.isTerminal() && x.p->v == w && e1.w != Complex::zero) {
					cn.returnToCache(e1.w);
				}

				if (!y.isTerminal() && y.p->v == w && e2.w != Complex::zero) {
					cn.returnToCache(e2.w);
				}
			}

			auto e = makeDDNode(w, edge, true);

			addTable.insert({ xCopy.p,xCopy.w }, { yCopy.p,yCopy.w }, { e.p, e.w });
			if (x.w != Complex::one) {
				if (!e.w.exactlyZero()) {
					cn.mul(e.w, e.w, x.w);
				} else {
					return Edge<Node>::zero;
					printf("L1058");
				}
				cn.returnToCache(yCopy.w);
			}
			return e;
		}

		//template <class LeftOperandNode, class RightOperandNode>
		Edge<mNode> cont2(const Edge<mNode>& x, const Edge<mNode>& y, key_2_new_key_node* key_2_new_key1, key_2_new_key_node* key_2_new_key2, const int var_num) {

			//std::cout <<"838 " << x.w << " " << y.w << " " << int(x.p->v) << " " << int(y.p->v) << std::endl;

			using ResultEdge = Edge<mNode>;

			if (x.p == nullptr) {
				return { nullptr, Complex::zero };
			}
			if (y.p == nullptr) {
				return y;
			}

			if (x.w.exactlyZero() || y.w.exactlyZero()) {
				return ResultEdge::zero;
			}


			if (x.p->v == -1 && y.p->v == -1)
			{
				auto c = cn.mulCached(x.w, y.w);

				if (var_num > 0) {
					ComplexNumbers::mul(c, c, cn.getTemporary(pow(2, var_num), 0));
				}

				return ResultEdge::terminal(c);
			}

			key_2_new_key_node* temp_key_2_new_key2 = key_2_new_key2;
			while (temp_key_2_new_key2->level > y.p->v) {
				temp_key_2_new_key2 = temp_key_2_new_key2->father;
			}


			if (x.p->v == -1 && var_num == 0 && std::abs(temp_key_2_new_key2->new_key - y.p->v) < 1e-10) {
				return 	ResultEdge{ y.p, cn.mulCached(x.w, y.w) };
			}

			key_2_new_key_node* temp_key_2_new_key1 = key_2_new_key1;
			while (temp_key_2_new_key1->level > x.p->v) {
				temp_key_2_new_key1 = temp_key_2_new_key1->father;
			}

			if (y.p->v == -1 && var_num == 0 && std::abs(temp_key_2_new_key1->new_key - x.p->v) < 1e-10) {
				return 	ResultEdge{ x.p, cn.mulCached(x.w, y.w) };
			}


			auto xCopy = x;
			xCopy.w = Complex::one;
			auto yCopy = y;
			yCopy.w = Complex::one;

			auto res = contTable.lookup(xCopy, yCopy, temp_key_2_new_key1, temp_key_2_new_key2);
			if (res.e.p != nullptr) {
				if (res.e.w.approximatelyZero()) {
					return ResultEdge::zero;
				}
				auto e = ResultEdge{ res.e.p, cn.getCached(res.e.w) };
				ComplexNumbers::mul(e.w, e.w, x.w);
				ComplexNumbers::mul(e.w, e.w, y.w);
				if (e.w.approximatelyZero()) {
					cn.returnToCache(e.w);
					return ResultEdge::zero;
				}
				if (res.cont_num != var_num) {
					ComplexNumbers::mul(e.w, e.w, cn.getTemporary(pow(2, var_num - res.cont_num), 0));// base 2 may be wrong for a tensor of arbitrary shape
				}
				return e;
			}


			float newk1 = temp_key_2_new_key1->new_key;

			float newk2 = temp_key_2_new_key2->new_key;

			ResultEdge e1{}, e2{}, r{};

			if (newk1 > newk2) {
				if (int(newk1 * 2) % 2 == 0) {
					std::vector<ResultEdge> e;
					for (int k = 0; k < x.p->e.size(); ++k) {
						e1 = x.p->e[k];
						e2 = yCopy;
						e.push_back(cont2(e1, e2, temp_key_2_new_key1, temp_key_2_new_key2, var_num));
					}
					r = makeDDNode(Qubit(newk1), e, true);
				}
				else {
					r = ResultEdge::zero;
					ResultEdge etemp{};
					for (int k = 0; k < x.p->e.size(); ++k) {
						e1 = x.p->e[k];
						e2 = yCopy;
						etemp = cont2(e1, e2, temp_key_2_new_key1, temp_key_2_new_key2, var_num - 1);
						if (!etemp.w.exactlyZero()) {
							if (r != ResultEdge::zero) {
								auto temp = r.w;
								r = T_add2(r, etemp);
								cn.returnToCache(temp);
								cn.returnToCache(etemp.w);
							}
							else {
								r = etemp;
							}
						}
					}
				}
			}
			else if (newk1 < newk2) {
				if (int(newk2 * 2) % 2 == 0) {
					std::vector<ResultEdge> e;
					for (int k = 0; k < y.p->e.size(); ++k) {
						e1 = xCopy;
						e2 = y.p->e[k];
						e.push_back(cont2(e1, e2, temp_key_2_new_key1, temp_key_2_new_key2, var_num));
					}
					r = makeDDNode(Qubit(newk2), e, true);
				}
				else {
					r = ResultEdge::zero;
					ResultEdge etemp{};
					for (int k = 0; k < y.p->e.size(); ++k) {
						e1 = xCopy;
						e2 = y.p->e[k];
						etemp = cont2(e1, e2, temp_key_2_new_key1, temp_key_2_new_key2, var_num - 1);
						if (!etemp.w.exactlyZero()) {
							if (r != ResultEdge::zero) {
								auto temp = r.w;
								r = T_add2(r, etemp);
								cn.returnToCache(temp);
								cn.returnToCache(etemp.w);
							}
							else {
								r = etemp;
							}
						}
					}
				}

			}
			else {
				if (int(newk2 * 2) % 2 == 0) {
					std::vector<ResultEdge> e;
					for (int k = 0; k < x.p->e.size(); ++k) {
						e1 = x.p->e[k];
						e2 = y.p->e[k];
						e.push_back(cont2(e1, e2, temp_key_2_new_key1, temp_key_2_new_key2, var_num));
					}
					r = makeDDNode(Qubit(newk1), e, true);
				}
				else {
					r = ResultEdge::zero;
					ResultEdge etemp{};
					for (int k = 0; k < x.p->e.size(); ++k) {
						e1 = x.p->e[k];
						e2 = y.p->e[k];
						etemp = cont2(e1, e2, temp_key_2_new_key1, temp_key_2_new_key2, var_num - 1);
						if (!etemp.w.exactlyZero()) {
							if (r != ResultEdge::zero) {
								auto temp = r.w;
								r = T_add2(r, etemp);
								cn.returnToCache(temp);
								cn.returnToCache(etemp.w);
							}
							else {
								r = etemp;
							}
						}
					}
				}
			}

			contTable.insert(xCopy, yCopy, { r.p, r.w }, temp_key_2_new_key1, temp_key_2_new_key2, var_num);

			if (!r.w.exactlyZero() && (x.w.exactlyOne() || !y.w.exactlyZero())) {
				if (r.w.exactlyOne()) {
					r.w = cn.mulCached(x.w, y.w);
				}
				else {
					ComplexNumbers::mul(r.w, r.w, x.w);
					ComplexNumbers::mul(r.w, r.w, y.w);
				}
				if (r.w.approximatelyZero()) {
					cn.returnToCache(r.w);
					return ResultEdge::zero;
				}
			}

			return r;

		}

		/*
		def is_node_identitiy(node, length_indifferent, expected_length):
			if node == TDD.terminal_node or (not length_indifferent and expected_length < 0):
				return length_indifferent or expected_length == 0
			left_node = node.succ[0]
			right_node = node.succ[1]

			return (node.out_weight[0] == ct.cn1 and
					node.out_weight[1] == ct.cn1 and
					left_node.out_weight is not None and right_node.out_weight is not None and
					left_node.out_weight[0] == ct.cn1 and
					left_node.out_weight[1] == ct.cn0 and
					right_node.out_weight[0] == ct.cn0 and
					right_node.out_weight[1] == ct.cn1 and
					left_node.succ[0] == right_node.succ[1] and
					is_node_identitiy(left_node.succ[0], length_indifferent, -1 if length_indifferent else expected_length - 1))

		*/
		bool isNodeIdentity(const Edge<mNode>& edge, bool lengthIndifferent, int expectedLenght) {
			auto node = edge.p;
			if (dd::mNode::isTerminal(node)) {
				if (!(lengthIndifferent || expectedLenght == 0)) {
					printf("I failed at node iden because expec len = %d\n", expectedLenght);
				}
				return lengthIndifferent || expectedLenght == 0;
			}
			
			
			if (node->e.size() == 2) {
				auto leftNode = node->e[0].p;
				auto rightNode = node->e[1].p;

				bool nodeWeights = node->e[0].w == Complex::one && node->e[1].w == Complex::one;
				bool leftWeights = leftNode->e.size() == 2 && leftNode->e[0].w == Complex::one && leftNode->e[1].w == Complex::zero;
				bool rightWeights = rightNode->e.size() == 2 && rightNode->e[0].w == Complex::zero && rightNode->e[1].w == Complex::one;
				bool sameSuccessors = leftNode->e[0] == rightNode->e[1];
				return nodeWeights && leftWeights && rightWeights && sameSuccessors && isNodeIdentity(leftNode->e[0], lengthIndifferent, expectedLenght - 1);
			}

			return false;
		}



	public:
		///
		/// Decision diagram size
		///
		template <class Edge> unsigned int size(const Edge& e) {
			static constexpr unsigned int NODECOUNT_BUCKETS = 200000;
			static std::unordered_set<decltype(e.p)> visited{NODECOUNT_BUCKETS}; // 2e6
			visited.max_load_factor(10);
			visited.clear();
			return nodeCount(e, visited);
		}

	private:
		template <class Edge>
		unsigned int nodeCount(const Edge& e,
			std::unordered_set<decltype(e.p)>& v) const {
			v.insert(e.p);
			unsigned int sum = 1;
			if (!e.isTerminal()) {
				for (const auto& edge : e.p->e) {
					if (edge.p != nullptr && !v.count(edge.p)) {
						sum += nodeCount(edge, v);
					}
				}
			}
			return sum;
		}


		///
		/// Printing and Statistics
		///
	public:
		// print information on package and its members
		static void printInformation() {
			std::cout << "\n  compiled: " << __DATE__ << " " << __TIME__
				<< "\n  Complex size: " << sizeof(Complex) << " bytes (aligned "
				<< alignof(Complex) << " bytes)"
				<< "\n  ComplexValue size: " << sizeof(ComplexValue)
				<< " bytes (aligned " << alignof(ComplexValue) << " bytes)"
				<< "\n  ComplexNumbers size: " << sizeof(ComplexNumbers)
				<< " bytes (aligned " << alignof(ComplexNumbers) << " bytes)"
				<< "\n  mEdge size: " << sizeof(mEdge) << " bytes (aligned "
				<< alignof(mEdge) << " bytes)"
				<< "\n  mNode size: " << sizeof(mNode) << " bytes (aligned "
				<< alignof(mNode) << " bytes)"
				<< "\n  Package size: " << sizeof(Package) << " bytes (aligned "
				<< alignof(Package) << " bytes)"
				<< "\n"
				<< std::flush;
		}

		// print unique and compute table statistics

		void statistics() {
			std::cout << "DD statistics:\n";
			std::cout << "[UniqueTable] ";
			nodeUniqueTable.printStatistics();
			std::cout << "[Add] ";
			addTable.printStatistics();
			std::cout << "[Cont] ";
			contTable.printStatistics();
			std::cout << "[ComplexTable] ";
			cn.complexTable.printStatistics();
		}

	};

} // namespace dd
