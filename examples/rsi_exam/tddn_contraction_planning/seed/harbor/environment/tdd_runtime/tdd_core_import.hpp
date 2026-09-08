#pragma once

//#include "DDpackage.h"
//#include "DDcomplex.h"


#include <nlohmann/json.hpp>
#include "dd/Package.hpp"
#include "dd/Export.hpp"

#include <unordered_set>
#include <vector>
#include <array>
#include <bitset>
#include <sstream>
#include <fstream>
#include <string>
#include <cstring>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <regex>
#include"time.h"
#include <math.h>
#include <chrono>
#include <ctime>
#include <sys/time.h>

#include <unistd.h>
#include <filesystem>
#include <thread>


using namespace std;
using json = nlohmann::json;
//auto dd = std::make_unique<dd::Package<>>(100);

struct gate {
	std::string name;
	short int qubits[2];
};

std::map<int, int> plan_offset;

constexpr long double PI = 3.14159265358979323846264338327950288419716939937510L;

bool release = true;
bool get_max_node = false;
bool to_test = false;
int split_gates_count = 0;




int qubits_num = 0;
int gates_num = 0;
clock_t   start, finish;
void print_index_set(std::vector<dd::Index> index_set) {
	for (int k = 0; k < index_set.size(); k++) {
		std::cout << "(" << index_set[k].key << ", " << index_set[k].idx << ") ,";

	}
	std::cout << std::endl;
}
// extract the gate name and qubits from one line of a qasm file
vector<std::string> split(const std::string& s, const std::string& seperator) {
	vector<std::string> result;
	typedef string::size_type string_size;
	string_size i = 0;

	while (i != s.size()) {
		// find the first character that is not a delimiter
		int flag = 0;
		while (i != s.size() && flag == 0) {
			flag = 1;
			for (string_size x = 0; x < seperator.size(); ++x)
				if (s[i] == seperator[x]) {
					++i;
					flag = 0;
					break;
				}
		}

		// find the next delimiter and take the substring between the two
		flag = 0;
		string_size j = i;
		while (j != s.size() && flag == 0) {
			for (string_size x = 0; x < seperator.size(); ++x)
				if (s[j] == seperator[x]) {
					flag = 1;
					break;
				}
			if (flag == 0)
				++j;
		}
		if (i != j) {
			result.push_back(s.substr(i, j - i));
			i = j;
		}
	}
	return result;
}


float match_a_string(string s) {
	smatch result;
	regex pattern("(-?\\d+.\\d+)");
	regex pattern2("(-?\\d+.\\d+)\\*?pi/(\\d+)");
	regex pattern3("(-?\\d+.\\d+)\\*?pi");
	regex pattern4("pi/(\\d+)");
	regex pattern5("(\\d+)");
	regex pattern6("-pi/(\\d+)");
	if (regex_match(s, result, pattern)) {
		//cout << result[1] << endl;
		return stof(result[1]);
	}
	else if (regex_match(s, result, pattern2)) {
		//cout << result[1] << ',' << result[2] << endl;
		return stof(result[1]) * PI / stof(result[2]);
	}
	else if (regex_match(s, result, pattern3)) {
		//cout << result[1] << endl;
		return stof(result[1]) * PI;
	}
	else if (regex_match(s, result, pattern4)) {
		//cout << result[1] << endl;
		return PI / stof(result[1]);
	}
	else if (regex_match(s, result, pattern5)) {
		//cout << result[1] << endl;
		return stof(result[1]);
	}
	else if (regex_match(s, result, pattern6)) {
		//cout << result[1] << endl;
		return -PI / stof(result[1]);
	}
	std::cout << s << endl;
	std::cout << "Not Macth" << endl;
	return 0.0;
}

// load a qasm file
std::map<int, gate> import_circuit(std::string  file_name) {

	qubits_num = 0;
	gates_num = 0;

	std::map<int, gate> gate_set;

	std::ifstream  infile;

	infile.open(file_name);

	std::string line;
	std::getline(infile, line);
	std::getline(infile, line);
	//std::getline(infile, line);
	std::getline(infile, line);
	while (std::getline(infile, line))
	{
		gate temp_gate;

		vector<std::string> g = split(line, " ");
		smatch result;

		temp_gate.name = g[0];

		if (g[0] == "cx") {
			regex pattern("q\\[(\\d+)\\], ?q\\[(\\d+)\\];");
			if (regex_match(g[1], result, pattern))
			{
				if (stoi(result[1]) > qubits_num) {
					qubits_num = stoi(result[1]);
				}
				if (stoi(result[2]) > qubits_num) {
					qubits_num = stoi(result[2]);
				}
				temp_gate.qubits[0] = stoi(result[1]);
				temp_gate.qubits[1] = stoi(result[2]);
			}

		}
		else {
			regex pattern("q\\[(\\d+)\\];");
			if (regex_match(g[1], result, pattern))
			{
				if (stoi(result[1]) > qubits_num) {
					qubits_num = stoi(result[1]);
				}
				temp_gate.qubits[0] = stoi(result[1]);
			}
		}

		gate_set[gates_num] = temp_gate;
		gates_num++;
	}
	infile.close();
	qubits_num += 1;
	return gate_set;
}

std::map<int, gate> import_circuit_from_string(std::string circuit) {

	qubits_num = 0;
	gates_num = 0;

	std::map<int, gate> gate_set;
	std::stringstream infile(circuit);

	std::string line;
	std::getline(infile, line);
	std::getline(infile, line);
	//std::getline(infile, line);
	std::getline(infile, line);
	while (std::getline(infile, line))
	{
		gate temp_gate;

		vector<std::string> sep = split(line, "#");
		vector<std::string> g = split(sep[1], " ");
		smatch result;

		temp_gate.name = g[0];

		if (g[0] == "cx" || g[0] == "cy" || g[0] == "cz" || g[0] == "cnot") {
			regex pattern("q\\[(\\d+)\\], ?q\\[(\\d+)\\];");
			if (regex_match(g[1], result, pattern))
			{
				if (stoi(result[1]) > qubits_num) {
					qubits_num = stoi(result[1]);
				}
				if (stoi(result[2]) > qubits_num) {
					qubits_num = stoi(result[2]);
				}
				temp_gate.qubits[0] = stoi(result[1]);
				temp_gate.qubits[1] = stoi(result[2]);
			}

		} else if (g[0] == "cx_c" || g[0] == "cy_c" || g[0] == "cz_c" || g[0] == "cx_t" || g[0] == "cy_t" || g[0] == "cz_t") {
			regex pattern("q\\[(\\d+)\\], ?q\\[(\\d+)\\];");
			if (regex_match(g[1], result, pattern))
			{
				if (stoi(result[1]) > qubits_num) {
					qubits_num = stoi(result[1]);
				}
				if (stoi(result[2]) > qubits_num) {
					qubits_num = stoi(result[2]);
				}
				temp_gate.qubits[0] = stoi(result[1]);
				temp_gate.qubits[1] = stoi(result[2]);
			}
			if (g[0] == "cx_c" || g[0] == "cy_c" || g[0] == "cz_c")
				split_gates_count++;
		} else {
			regex pattern("q\\[(\\d+)\\];");
			if (regex_match(g[1], result, pattern))
			{
				if (stoi(result[1]) > qubits_num) {
					qubits_num = stoi(result[1]);
				}
				temp_gate.qubits[0] = stoi(result[1]);
			}
		}

		gate_set[gates_num] = temp_gate;
		plan_offset[stoi(sep[0])] = gates_num;

		gates_num++;
	}
	
	qubits_num += 1;
	return gate_set;
}

std::map<int, std::vector<dd::Index>> get_index(std::map<int, gate> gate_set, std::map<std::string, int> var) {

	std::map<int, std::vector<dd::Index>> Index_set;

	std::map<std::string, short> hyper_idx;


	for (const auto& pair : var) {
		hyper_idx[pair.first] = 0;
	}



	int* qubit_idx = new   int[qubits_num]();
	int* between_idx = new int[qubits_num]();

	for (int k = 0; k < gates_num; k++)
	{
		//std::cout << k << std::endl;
		std::string nam = gate_set[k].name;
		//std::cout << nam << std::endl;
		//std::cout << gate_set[k].qubits[0]<<"    "<<gate_set[k].qubits[1] << endl;
		if (nam == "cx" || nam == "cy" || nam == "cz" || nam == "cnot") {
			int con_q = gate_set[k].qubits[0];
			int tar_q = gate_set[k].qubits[1];
			std::string cont_idx1 = "x";
			cont_idx1 += to_string(con_q);
			cont_idx1 += "_";
			cont_idx1 += to_string(qubit_idx[con_q]);
			qubit_idx[con_q] += 1;
			std::string cont_idx2 = "x";
			cont_idx2 += to_string(con_q);
			cont_idx2 += "_";
			cont_idx2 += to_string(qubit_idx[con_q]);
			std::string targ_idx1 = "x";
			targ_idx1 += to_string(tar_q);
			targ_idx1 += "_";
			targ_idx1 += to_string(qubit_idx[tar_q]);
			qubit_idx[tar_q] += 1;
			std::string targ_idx2 = "x";
			targ_idx2 += to_string(tar_q);
			targ_idx2 += "_";
			targ_idx2 += to_string(qubit_idx[tar_q]);
			Index_set[k] = { {cont_idx1,hyper_idx[cont_idx1]},{cont_idx2,hyper_idx[cont_idx2]},{targ_idx1,hyper_idx[targ_idx1]},{targ_idx2,hyper_idx[targ_idx2]} };
			//std::cout << cont_idx<<" " << hyper_idx[cont_idx] << " " << cont_idx << " " << hyper_idx[cont_idx] + 1 << " " << cont_idx << " " << hyper_idx[cont_idx] + 2 << " " << targ_idx1 << " " << hyper_idx[targ_idx1] << " " << targ_idx2 << " " <<hyper_idx[targ_idx2] << " " << std::endl;
			//hyper_idx[cont_idx] += 2;

		} else if (nam == "cx_c" || nam == "cy_c" || nam == "cz_c") {
			int con_q = gate_set[k].qubits[0];
			int tar_q = gate_set[k].qubits[1];
			std::string cont_idx1 = "x";
			cont_idx1 += to_string(con_q);
			cont_idx1 += "_";
			cont_idx1 += to_string(qubit_idx[con_q]);
			qubit_idx[con_q] += 1;
			std::string cont_idx2 = "x";
			cont_idx2 += to_string(con_q);
			cont_idx2 += "_";
			cont_idx2 += to_string(qubit_idx[con_q]);
			
			int bet_q = con_q > tar_q ? con_q : tar_q;
			std::string between_edge = "p";
			between_edge += to_string(bet_q);
			between_edge += "_";
			between_edge += to_string(between_idx[bet_q]);

			Index_set[k] = { {cont_idx1,hyper_idx[cont_idx1]}, {cont_idx2,hyper_idx[cont_idx2]}, {between_edge,0} };
		} else if (nam == "cx_t" || nam == "cy_t" || nam == "cz_t") {
			int con_q = gate_set[k].qubits[0];
			int tar_q = gate_set[k].qubits[1];
			std::string targ_idx1 = "x";
			targ_idx1 += to_string(tar_q);
			targ_idx1 += "_";
			targ_idx1 += to_string(qubit_idx[tar_q]);
			qubit_idx[tar_q] += 1;
			std::string targ_idx2 = "x";
			targ_idx2 += to_string(tar_q);
			targ_idx2 += "_";
			targ_idx2 += to_string(qubit_idx[tar_q]);
			
			int bet_q = con_q > tar_q ? con_q : tar_q;
			std::string between_edge = "p";
			between_edge += to_string(bet_q);
			between_edge += "_";
			between_edge += to_string(between_idx[bet_q]);
			between_idx[bet_q] += 1;

			Index_set[k] = { {targ_idx1,hyper_idx[targ_idx1]}, {targ_idx2,hyper_idx[targ_idx2]}, {between_edge,0} };
		} else {
			int tar_q = gate_set[k].qubits[0];
			std::string targ_idx1 = "x";
			std::string targ_idx2 = "x";

			targ_idx1 += to_string(tar_q);
			targ_idx1 += "_";
			targ_idx1 += to_string(qubit_idx[tar_q]);
			qubit_idx[tar_q] += 1;
			targ_idx2 += to_string(tar_q);
			targ_idx2 += "_";
			targ_idx2 += to_string(qubit_idx[tar_q]);
			Index_set[k] = { {targ_idx1,hyper_idx[targ_idx1]},{targ_idx2,hyper_idx[targ_idx2]} };
			if (false && (nam == "x" || nam == "h" || nam == "z" || nam == "s" || nam == "sdg" || nam == "t" || nam == "tdg" || (nam[0] == 'u' && nam[1] == '1') || (nam[0] == 'r' && nam[1] == 'z') || (nam[0] == 'r' && nam[1] == 'y'))) {
				Index_set[k] = { {targ_idx1,hyper_idx[targ_idx1]},{targ_idx1,((short)(hyper_idx[targ_idx1] + 1))} };
				qubit_idx[tar_q] -= 1;
				hyper_idx[targ_idx1] += 1;
			}
			else {
				Index_set[k] = { {targ_idx1,hyper_idx[targ_idx1]},{targ_idx2,hyper_idx[targ_idx2]} };
			}
		}
		//std::cout << k << " ";
		//print_index_set(Index_set[k]);
	}
	return Index_set;
}


std::map<int, map<int, std::vector<int>>>  cir_partition1(std::map<int, gate> gate_set, int cx_cut_max) {
	std::map<int, map<int, std::vector<int>>>   par;
	int cx_cut = 0;
	int block = 0;
	for (int k = 0; k < gates_num; k++)
	{
		std::string nam = gate_set[k].name;
		if (nam != "cx") {
			if (gate_set[k].qubits[0] <= qubits_num / 2) {
				par[block][0].push_back(k);
			}
			else {
				par[block][1].push_back(k);
			}
		}
		else {
			if (gate_set[k].qubits[0] <= qubits_num / 2 && gate_set[k].qubits[1] <= qubits_num / 2) {
				par[block][0].push_back(k);
			}
			else if (gate_set[k].qubits[0] > qubits_num / 2 && gate_set[k].qubits[1] > qubits_num / 2) {
				par[block][1].push_back(k);
			}
			else {
				if (cx_cut <= cx_cut_max) {
					if (gate_set[k].qubits[1] > qubits_num / 2) {
						par[block][1].push_back(k);
					}
					else {
						par[block][0].push_back(k);
					}
					cx_cut += 1;
				}
				else {
					block += 1;
					cx_cut = 1;
					if (gate_set[k].qubits[1] > qubits_num / 2) {
						par[block][1].push_back(k);
					}
					else {
						par[block][0].push_back(k);
					}
				}
			}
		}
	}

	//for (int k = 0; k < par.size(); k++) {
	//	for (int k1 = 0; k1 < par[k][0].size(); k1++) {
	//		cout << par[k][0][k1] << "  ";
	//	}
	//	cout << endl;
	//	for (int k1 = 0; k1 < par[k][1].size(); k1++) {
	//		cout << par[k][1][k1] << "  ";
	//	}
	//	cout << endl;
	//	cout << "--------" << endl;
	//}


	return par;
}

int min(int a, int b) {
	if (a <= b) {

		return a;
	}
	else {
		return b;
	}
}
int max(int a, int b) {
	if (a >= b) {

		return a;
	}
	else {
		return b;
	}
}

std::map<int, map<int, std::vector<int>>>  cir_partition2(std::map<int, gate> gate_set, int cx_cut_max, int c_part_width) {
	std::map<int, map<int, std::vector<int>>>   par;
	int cx_cut = 0;
	int block = 0;
	int c_part_min = qubits_num / 2;
	int c_part_max = qubits_num / 2;
	for (int k = 0; k < gates_num; k++)
	{
		std::string nam = gate_set[k].name;

		if (cx_cut <= cx_cut_max) {

			if (nam != "cx") {
				if (gate_set[k].qubits[0] <= qubits_num / 2) {
					par[block][0].push_back(k);
				}
				else {
					par[block][1].push_back(k);
				}
			}
			else {
				if (gate_set[k].qubits[0] <= qubits_num / 2 && gate_set[k].qubits[1] <= qubits_num / 2) {
					par[block][0].push_back(k);
				}
				else if (gate_set[k].qubits[0] > qubits_num / 2 && gate_set[k].qubits[1] > qubits_num / 2) {
					par[block][1].push_back(k);
				}
				else {
					if (gate_set[k].qubits[1] > qubits_num / 2) {
						par[block][1].push_back(k);
					}
					else {
						par[block][0].push_back(k);
					}
					cx_cut += 1;
				}
			}
		}
		else {
			if (nam != "cx") {
				if (gate_set[k].qubits[0] < c_part_min) {
					par[block][0].push_back(k);
				}
				else if (gate_set[k].qubits[0] > c_part_max) {
					par[block][1].push_back(k);
				}
				else {
					par[block][2].push_back(k);
				}
			}
			else if (gate_set[k].qubits[0] >= c_part_min && gate_set[k].qubits[0] <= c_part_max && gate_set[k].qubits[1] >= c_part_min && gate_set[k].qubits[1] <= c_part_max) {
				par[block][2].push_back(k);
			}
			else if (gate_set[k].qubits[0] < c_part_min && gate_set[k].qubits[1] < c_part_min)
			{
				par[block][0].push_back(k);
			}
			else if (gate_set[k].qubits[0] > c_part_max && gate_set[k].qubits[1] > c_part_max)
			{
				par[block][1].push_back(k);
			}
			else {
				int temp_c_min = min(c_part_min, min(gate_set[k].qubits[0], gate_set[k].qubits[1]));
				int temp_c_max = max(c_part_max, max(gate_set[k].qubits[0], gate_set[k].qubits[1]));
				if ((temp_c_max - temp_c_min) > c_part_width) {
					block += 1;
					cx_cut = 0;
					c_part_min = qubits_num / 2;
					c_part_max = qubits_num / 2;
					if (gate_set[k].qubits[0] <= qubits_num / 2 && gate_set[k].qubits[1] <= qubits_num / 2) {
						par[block][0].push_back(k);
					}
					else if (gate_set[k].qubits[0] > qubits_num / 2 && gate_set[k].qubits[1] > qubits_num / 2) {
						par[block][1].push_back(k);
					}
					else {
						if (gate_set[k].qubits[1] > qubits_num / 2) {
							par[block][1].push_back(k);
						}
						else {
							par[block][0].push_back(k);
						}
						cx_cut += 1;
					}
				}
				else {
					par[block][2].push_back(k);
					c_part_min = temp_c_min;
					c_part_max = temp_c_max;
				}
			}
		}
	}

	//for (int k = 0; k < 1; k++) {
	//for (int k1 = 0; k1 < par[k][0].size(); k1++) {
	//	cout << par[k][0][k1] << "  ";
	//}
	//cout << endl;
	//for (int k1 = 0; k1 < par[k][1].size(); k1++) {
	//	cout << par[k][1][k1] << "  ";
	//}
	//cout << endl;
	//for (int k1 = 0; k1 < par[k][2].size(); k1++) {
	//	cout << par[k][2][k1] << "  ";
	//}
	//cout << endl;
	//cout << "--------" << endl;
	//}
	return par;
}


dd::TDD apply(dd::TDD tdd, std::string nam, std::vector<dd::Index> index_set, std::unique_ptr<dd::Package<>>& dd) {

	//std::cout << nam << std::endl;

	std::map<std::string, int> gate_type;
	gate_type["x"] = 1;
	gate_type["y"] = 2;
	gate_type["z"] = 3;
	gate_type["h"] = 4;
	gate_type["s"] = 5;
	gate_type["sdg"] = 6;
	gate_type["t"] = 7;
	gate_type["tdg"] = 8;

	dd::TDD temp_tdd;

	if (nam == "cx") {
		temp_tdd = dd->cnot_2_TDD(index_set, 1);
	}
	else {
		switch (gate_type[nam]) {
		case 1:
			temp_tdd = dd->Matrix2TDD(dd::Xmat, index_set);
			break;
		case 2:
			temp_tdd = dd->Matrix2TDD(dd::Ymat, index_set);

			//std::cout << temp_tdd.e.w << " " << int(temp_tdd.e.p->v) << std::endl;
			//std::cout << temp_tdd.e.p->e[0].w << " " << int(temp_tdd.e.p->e[0].p->v) << std::endl;
			//std::cout << temp_tdd.e.p->e[1].w << " " << int(temp_tdd.e.p->e[1].p->v) << std::endl;

			break;
		case 3:
			temp_tdd = dd->diag_matrix_2_TDD(dd::Zmat, index_set);
			break;
		case 4:
			temp_tdd = dd->Matrix2TDD(dd::Hmat, index_set);
			break;
		case 5:
			temp_tdd = dd->diag_matrix_2_TDD(dd::Smat, index_set);
			break;
		case 6:
			temp_tdd = dd->diag_matrix_2_TDD(dd::Sdagmat, index_set);
			break;
		case 7:
			temp_tdd = dd->diag_matrix_2_TDD(dd::Tmat, index_set);
			break;
		case 8:
			temp_tdd = dd->diag_matrix_2_TDD(dd::Tdagmat, index_set);
			break;
		default:
			if (nam[0] == 'r' and nam[1] == 'z') {
				regex pattern("rz\\((-?\\d.\\d+)\\)");
				smatch result;
				regex_match(nam, result, pattern);
				float theta = stof(result[1]);
				//dd::GateMatrix Rzmat = { { 1, 0 }, { 0, 0 } , { 0, 0 }, { cos(theta), sin(theta) } };
				temp_tdd = dd->diag_matrix_2_TDD(dd::Phasemat(theta), index_set);
				break;
			}
			if (nam[0] == 'u' and nam[1] == '1') {
				//regex pattern("u1\\((-?\\d.\\d+)\\)");
				//smatch result;
				//regex_match(nam, result, pattern);
				//float theta = stof(result[1]);

				regex para(".*?\\((.*?)\\)");
				smatch result;
				regex_match(nam, result, para);
				float theta = match_a_string(result[1]);

				//dd::GateMatrix  U1mat = { { 1, 0 }, { 0, 0 } , { 0, 0 }, { cos(theta), sin(theta) }  };

				temp_tdd = dd->diag_matrix_2_TDD(dd::Phasemat(theta), index_set);
				break;
			}
			if (nam[0] == 'u' and nam[1] == '3') {
				//regex pattern("u3\\((-?\\d.\\d+), ?(-?\\d.\\d+), ?(-?\\d.\\d+)\\)");
				//smatch result;
				//regex_match(nam, result, pattern);
				//float theta = stof(result[1]);
				//float phi = stof(result[2]);
				//float lambda = stof(result[3]);

				regex para(".*?\\((.*?)\\)");
				smatch result;
				regex_match(nam, result, para);
				vector<string> para2 = split(result[1], ",");
				float theta = match_a_string(para2[0]);
				float phi = match_a_string(para2[1]);
				float lambda = match_a_string(para2[2]);
				//dd::GateMatrix  U3mat = { { cos(theta / 2), 0 }, { -cos(lambda) * sin(theta / 2),-sin(lambda) * sin(theta / 2)} , { cos(phi) * sin(theta / 2),sin(phi) * sin(theta / 2) }, { cos(lambda + phi) * cos(theta / 2),sin(lambda + phi) * cos(theta / 2) }  };
				temp_tdd = dd->Matrix2TDD(dd::U3mat(lambda, phi, theta), index_set);
				break;
			}
		}
	}


	if (release) {
		auto tmp = dd->cont(tdd, temp_tdd);
		dd->incRef(tmp.e);
		dd->decRef(tdd.e);
		tdd = tmp;
		dd->garbageCollect();
	}
	else {
		tdd = dd->cont(tdd, temp_tdd);
	}



	return tdd;
}


dd::TDD gateToTDD(std::string nam, std::vector<dd::Index> index_set, std::unique_ptr<dd::Package<>>& dd) {

	//std::cout << nam << std::endl;

	std::map<std::string, int> gate_type;
	gate_type["x"] = 1;
	gate_type["y"] = 2;
	gate_type["z"] = 3;
	gate_type["h"] = 4;
	gate_type["s"] = 5;
	gate_type["sdg"] = 6;
	gate_type["t"] = 7;
	gate_type["tdg"] = 8;
	gate_type["id"] = 9;

	dd::TDD temp_tdd;
	std::string gate_name = nam;
	std::vector<dd::fp> params = {};
	if (nam == "cx" || nam == "cnot") {
		//temp_tdd = dd->cnot_2_TDD(index_set, 1);
		temp_tdd = dd->cgate_2_TDD(index_set, "x");
	} else if (nam == "cy") {
		//temp_tdd = dd->cy_2_TDD(index_set, 1);
		temp_tdd = dd->cgate_2_TDD(index_set, "y");
	} else if (nam == "cz") {
		//temp_tdd = dd->cz_2_TDD(index_set, 1);
		temp_tdd = dd->cgate_2_TDD(index_set, "z");
	} else if (nam == "cx_c" || nam == "cx_t") {
		//temp_tdd = dd->cz_2_TDD(index_set, 1);
		temp_tdd = dd->split_cgate_2_TDD(index_set, "x", nam == "cx_c");
	} else if (nam == "cy_c" || nam == "cy_t") {
		//temp_tdd = dd->cz_2_TDD(index_set, 1);
		temp_tdd = dd->split_cgate_2_TDD(index_set, "y", nam == "cy_c");
	} else if (nam == "cz_c" || nam == "cz_t") {
		//temp_tdd = dd->cz_2_TDD(index_set, 1);
		temp_tdd = dd->split_cgate_2_TDD(index_set, "z", nam == "cz_c");
	} else {
		switch (gate_type[nam]) {
		case 1:
			temp_tdd = dd->Matrix2TDD(dd::Xmat, index_set);
			break;
		case 2:
			temp_tdd = dd->Matrix2TDD(dd::Ymat, index_set);
			break;
		case 3:
			temp_tdd = dd->Matrix2TDD(dd::Zmat, index_set);
			break;
		case 4:
			temp_tdd = dd->Matrix2TDD(dd::Hmat, index_set);
			break;
		case 5:
			temp_tdd = dd->Matrix2TDD(dd::Smat, index_set);
			break;
		case 6:
			temp_tdd = dd->Matrix2TDD(dd::Sdagmat, index_set);
			break;
		case 7:
			temp_tdd = dd->Matrix2TDD(dd::Tmat, index_set);
			break;
		case 8:
			temp_tdd = dd->Matrix2TDD(dd::Tdagmat, index_set);
			break;
		case 9:
			temp_tdd = dd->Matrix2TDD(dd::Imat, index_set);
			break;
		default:
			gate_name = "";
			gate_name += nam[0];
			gate_name += nam[1];
			if (nam[0] == 'r' and nam[1] == 'z') {
				regex pattern("rz\\((-?\\d.\\d+)\\)");
				smatch result;
				regex_match(nam, result, pattern);
				float theta = stof(result[1]);
				params.push_back(theta);
				// dd::fp act_theta;
				// if (fabs(fabs(theta) - dd::PI) < 0.0000001) {
				// 	act_theta = dd::PI;
				// } else if (fabs(fabs(theta) - dd::PI_2) < 0.0000001) {
				// 	act_theta = dd::PI_2;
				// } else if (fabs(fabs(theta) - dd::PI_4) < 0.0000001) {
				// 	act_theta = dd::PI_4;
				// } else {
				// 	act_theta = fabs(theta);
				// }
				// temp_tdd = dd->diag_matrix_2_TDD(dd::RZmat(theta < 0.0 ? -act_theta : act_theta), index_set);
				if (to_test) {
					for (int i = 0; i < index_set.size(); i++) {
						printf("RZ index set before %d: %s\n", i, index_set[i].key.c_str());
					}
				}
				temp_tdd = dd->Matrix2TDD(dd::RZmat(theta), index_set);
				if (to_test) {
					for (int i = 0; i < temp_tdd.index_set.size(); i++) {
						printf("RZ index set after %d: %s\n", i, temp_tdd.index_set[i].key.c_str());
					}
				}
				break;
			}
			if (nam[0] == 'r' and nam[1] == 'y') {
				regex pattern("ry\\((-?\\d.\\d+)\\)");
				smatch result;
				regex_match(nam, result, pattern);
				float theta = stof(result[1]);
				params.push_back(theta);
				// dd::fp act_theta;
				// if (fabs(fabs(theta) - dd::PI) < 0.0000001) {
				// 	act_theta = dd::PI;
				// } else if (fabs(fabs(theta) - dd::PI_2) < 0.0000001) {
				// 	act_theta = dd::PI_2;
				// } else if (fabs(fabs(theta) - dd::PI_4) < 0.0000001) {
				// 	act_theta = dd::PI_4;
				// } else {
				// 	act_theta = fabs(theta);
				// }
				// temp_tdd = dd->diag_matrix_2_TDD(dd::RYmat(theta < 0.0 ? -act_theta : act_theta), index_set);
				temp_tdd = dd->Matrix2TDD(dd::RYmat(theta), index_set);
				break;
			}
			if (nam[0] == 'r' and nam[1] == 'x') {
				regex pattern("rx\\((-?\\d.\\d+)\\)");
				smatch result;
				regex_match(nam, result, pattern);
				float theta = stof(result[1]);
				params.push_back(theta);
				// dd::fp act_theta;
				// if (fabs(fabs(theta) - dd::PI) < 0.0000000000001) {
				// 	act_theta = dd::PI;
				// } else if (fabs(fabs(theta) - dd::PI_2) < 0.0000000000001) {
				// 	act_theta = dd::PI_2;
				// } else if (fabs(fabs(theta) - dd::PI_4) < 0.0000000000001) {
				// 	act_theta = dd::PI_4;
				// } else {
				// 	act_theta = fabs(theta);
				// }
				// temp_tdd = dd->diag_matrix_2_TDD(dd::RXmat(theta < 0.0 ? -act_theta : act_theta), index_set);
				temp_tdd = dd->Matrix2TDD(dd::RXmat(theta), index_set);
				break;
			}
			if (nam[0] == 'u' and nam[1] == '1') {
				//regex pattern("u1\\((-?\\d.\\d+)\\)");
				//smatch result;
				//regex_match(nam, result, pattern);
				//float theta = stof(result[1]);

				regex para(".*?\\((.*?)\\)");
				smatch result;
				regex_match(nam, result, para);
				float theta = match_a_string(result[1]);
				params.push_back(theta);

				temp_tdd = dd->Matrix2TDD(dd::Phasemat(theta), index_set);
				break;
			}
			if (nam[0] == 'u' and nam[1] == '3') {
				//regex pattern("u3\\((-?\\d.\\d+), ?(-?\\d.\\d+), ?(-?\\d.\\d+)\\)");
				//smatch result;
				//regex_match(nam, result, pattern);
				//float theta = stof(result[1]);
				//float phi = stof(result[2]);
				//float lambda = stof(result[3]);

				regex para(".*?\\((.*?)\\)");
				smatch result;
				regex_match(nam, result, para);
				vector<string> para2 = split(result[1], ",");
				if (to_test)
					printf("U3 (name = %s) num of parameters = %d\n", nam.c_str(), para2.size());
				float theta = match_a_string(para2[0]);
				float phi = match_a_string(para2[1]);
				float lambda = match_a_string(para2[2]);
				params.push_back(theta);
				params.push_back(phi);
				params.push_back(lambda);
				//dd::GateMatrix  U3mat = { { cos(theta / 2), 0 }, { -cos(lambda) * sin(theta / 2),-sin(lambda) * sin(theta / 2)} , { cos(phi) * sin(theta / 2),sin(phi) * sin(theta / 2) }, { cos(lambda + phi) * cos(theta / 2),sin(lambda + phi) * cos(theta / 2) }  };
				temp_tdd = dd->Matrix2TDD(dd::U3mat(lambda, phi, theta), index_set);
				break;
			}
		}
	}
	dd->incRef(temp_tdd.e);
	dd::GateDef temp_gate = {gate_name, params};
	temp_tdd.gates = {temp_gate};
	return temp_tdd;
}

dd::TDD applyTDDs(dd::TDD tdd1, dd::TDD tdd2, std::unique_ptr<dd::Package<>>& dd) {
	std::vector<dd::GateDef> merged_gates;
	merged_gates.reserve(tdd1.gates.size() + tdd2.gates.size());
	merged_gates.insert(merged_gates.end(), tdd1.gates.begin(), tdd1.gates.end());
	merged_gates.insert(merged_gates.end(), tdd2.gates.begin(), tdd2.gates.end());
	if (release) {
		auto tmp = dd->cont(tdd1, tdd2);
		dd->incRef(tmp.e);
		dd->decRef(tdd1.e);
		tdd1 = tmp;
		dd->garbageCollect();
	}
	else {
		tdd1 = dd->cont(tdd1, tdd2);
	}
	tdd1.gates = std::move(merged_gates);

	return tdd1;
}

dd::TDD applyTDDsWithJSON(dd::TDD tdd1, dd::TDD tdd2, std::unique_ptr<dd::Package<>>& dd, json& res_json) {
	std::vector<dd::GateDef> new_gates;
	new_gates.reserve(tdd1.gates.size() + tdd2.gates.size());
	new_gates.insert(new_gates.end(), tdd1.gates.begin(), tdd1.gates.end());
	new_gates.insert(new_gates.end(), tdd2.gates.begin(), tdd2.gates.end());

	json left;
	left["nodes"] = (int)dd->size(tdd1.e);
	left["gates"] = tdd1.gates;
	left["indices"] = tdd1.key_2_index;

	json right = {
		{"nodes", (int)dd->size(tdd2.e)},
		{"gates", tdd2.gates},
		{"indices", tdd2.key_2_index}
	};

	struct timeval start, end;
	long mtime, seconds, useconds;  

	if (release) {
		gettimeofday(&start, NULL);
		auto tmp = dd->cont(tdd1, tdd2);
		gettimeofday(&end, NULL);
		dd->incRef(tmp.e);
		dd->decRef(tdd1.e);
		tdd1 = tmp;
		dd->garbageCollect();
	}
	else {
		gettimeofday(&start, NULL);
		tdd1 = dd->cont(tdd1, tdd2);
		gettimeofday(&end, NULL);
	}

	seconds  = end.tv_sec  - start.tv_sec;
	useconds = end.tv_usec - start.tv_usec;
	//mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
	mtime = useconds;

	tdd1.gates = new_gates;
	json result = {
		{"nodes", (int)dd->size(tdd1.e)},
		{"gates", tdd1.gates},
		{"indices", tdd1.key_2_index}
	};

	res_json["left"] = left;
	res_json["right"] = right;
	res_json["result"] = result;
	res_json["time"] = mtime;

	return tdd1;
}

std::map<std::string, int> get_var_order() {

	std::map<std::string, int> var;


	// set the variable order
	int order_num = 1000000;

	for (int k = qubits_num; k >= 0; k--) {
		string idx_nam;
		idx_nam = "y";
		idx_nam += to_string(k);
		var[idx_nam] = order_num;
		order_num -= 1;
		for (int k2 = gates_num; k2 >= 0; k2--) {
			idx_nam = "x";
			idx_nam += to_string(k);
			idx_nam += "_";
			idx_nam += to_string(k2);
			var[idx_nam] = order_num;
			order_num -= 1;
			//cout << idx_nam << endl;
		}
		if (k != 0) {
			for (int k3 = split_gates_count; k3 >= 0; k3--) {
				idx_nam = "p";
				idx_nam += to_string(k);
				idx_nam += "_";
				idx_nam += to_string(k3);
				var[idx_nam] = order_num;
				order_num -= 1;
			}
		}

		idx_nam = "x";
		idx_nam += to_string(k);
		var[idx_nam] = order_num;
		order_num -= 1;
	}

	return var;
}
