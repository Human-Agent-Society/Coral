#include "tdd_core_import.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct ActiveNode {
    int id{};
    dd::TDD tdd{};
};

struct ActiveEdge {
    int id{};
    int u{};
    int v{};
    int shared_indices{};
};

class Engine {
public:
    explicit Engine(const json& case_spec)
        : qubits_(case_spec.at("qubits").get<int>()),
          precision_bits_(case_spec.value("precision_bits", 18)),
          package_(std::make_unique<dd::Package<>>(std::max(3, 3 * qubits_))) {
        if (qubits_ <= 0) {
            throw std::invalid_argument("qubits must be positive");
        }
        if (precision_bits_ < 1 || precision_bits_ > 52) {
            throw std::invalid_argument("precision_bits must be between 1 and 52");
        }
        release = case_spec.value("garbage_collect", true);
        dd::ComplexNumbers::setTolerance(std::ldexp(1.0, -precision_bits_));
        plan_offset.clear();
        split_gates_count = 0;
        const auto circuit = case_spec.at("circuit").get<std::string>();
        const auto gates = import_circuit_from_string(circuit);
        package_->varOrder = get_var_order();
        const auto index_sets = get_index(gates, package_->varOrder);

        for (const auto& [tensor_id, gate_index] : plan_offset) {
            const auto gate_it = gates.find(gate_index);
            const auto index_it = index_sets.find(gate_index);
            if (gate_it == gates.end() || index_it == index_sets.end()) {
                throw std::runtime_error("inconsistent tensor-to-gate mapping");
            }
            auto tdd = gateToTDD(gate_it->second.name, index_it->second, package_);
            initial_peak_ = std::max<std::uint64_t>(
                initial_peak_, static_cast<std::uint64_t>(package_->size(tdd.e)));
            nodes_.emplace(tensor_id, ActiveNode{tensor_id, std::move(tdd)});
            max_vertex_id_ = std::max(max_vertex_id_, tensor_id + 1);
        }
        if (nodes_.empty()) {
            throw std::invalid_argument("circuit produced no TDD tensors");
        }

        for (const auto& pair : case_spec.at("edges")) {
            if (!pair.is_array() || pair.size() != 2) {
                throw std::invalid_argument("each edge must be a two-item array");
            }
            int u = pair.at(0).get<int>();
            int v = pair.at(1).get<int>();
            if (u == v || !nodes_.count(u) || !nodes_.count(v)) {
                throw std::invalid_argument("edge references invalid tensors");
            }
            const auto ordered_pair = ordered(u, v);
            u = ordered_pair.first;
            v = ordered_pair.second;
            const int edge_id = make_edge_id(u, v);
            auto [it, inserted] = edges_.emplace(
                edge_id, ActiveEdge{edge_id, u, v, 1});
            if (!inserted) {
                it->second.shared_indices += 1;
            }
        }
        retire_isolated_nodes();
        peak_ = initial_peak_;
        latest_size_ = initial_peak_;
    }

    [[nodiscard]] bool terminal() const {
        return edges_.empty();
    }

    [[nodiscard]] json observation() const {
        json node_rows = json::array();
        std::map<int, int> degree;
        for (const auto& [edge_id, edge] : edges_) {
            (void)edge_id;
            degree[edge.u] += 1;
            degree[edge.v] += 1;
        }
        for (const auto& [node_id, node] : nodes_) {
            const auto size = static_cast<std::uint64_t>(package_->size(node.tdd.e));
            node_rows.push_back({
                {"node_id", node_id},
                {"tdd_size", size},
                {"log2_tdd_size", std::log2(static_cast<double>(std::max<std::uint64_t>(1, size)))},
                {"rank", node.tdd.key_2_index.size()},
                {"gate_count", node.tdd.gates.size()},
                {"degree", degree[node_id]},
            });
        }

        int max_shared = 1;
        int max_local = 1;
        for (const auto& [edge_id, edge] : edges_) {
            (void)edge_id;
            max_shared = std::max(max_shared, edge.shared_indices);
            max_local = std::max(max_local, degree[edge.u] + degree[edge.v] - 2);
        }

        json edge_rows = json::array();
        json mask = json::object();
        for (const auto& [edge_id, edge] : edges_) {
            const auto left_size = static_cast<std::uint64_t>(package_->size(nodes_.at(edge.u).tdd.e));
            const auto right_size = static_cast<std::uint64_t>(package_->size(nodes_.at(edge.v).tdd.e));
            const int local = degree[edge.u] + degree[edge.v] - 2;
            edge_rows.push_back({
                {"edge_id", edge_id},
                {"u", edge.u},
                {"v", edge.v},
                {"shared_indices", edge.shared_indices},
                {"bond_dimension", std::ldexp(1.0, std::min(edge.shared_indices, 60))},
                {"normalized_bond_strength",
                 static_cast<double>(edge.shared_indices) / static_cast<double>(max_shared)},
                {"normalized_local_topology",
                 static_cast<double>(local) / static_cast<double>(max_local)},
                {"left_tdd_size", left_size},
                {"right_tdd_size", right_size},
                {"left_degree", degree[edge.u]},
                {"right_degree", degree[edge.v]},
            });
            mask[std::to_string(edge_id)] = true;
        }

        return {
            {"nodes", std::move(node_rows)},
            {"edges", std::move(edge_rows)},
            {"action_mask", std::move(mask)},
            {"global_features", {
                {"active_nodes", nodes_.size()},
                {"active_edges", edges_.size()},
                {"step", step_},
                {"qubits", qubits_},
                {"latest_tdd_size", latest_size_},
                {"peak_tdd_size", peak_},
            }},
        };
    }

    void contract(const int edge_id) {
        const auto edge_it = edges_.find(edge_id);
        if (edge_it == edges_.end()) {
            throw std::invalid_argument("selected edge is not enabled");
        }
        const ActiveEdge selected = edge_it->second;
        const int left_id = selected.u;
        const int right_id = selected.v;

        std::map<int, int> neighbor_shared;
        std::vector<int> incident;
        for (const auto& [candidate_id, candidate] : edges_) {
            if (candidate.u == left_id || candidate.v == left_id ||
                candidate.u == right_id || candidate.v == right_id) {
                incident.push_back(candidate_id);
                if (candidate_id == edge_id) {
                    continue;
                }
                int neighbor = candidate.u;
                if (neighbor == left_id || neighbor == right_id) {
                    neighbor = candidate.v;
                }
                if (neighbor == left_id || neighbor == right_id) {
                    continue;
                }
                neighbor_shared[neighbor] += candidate.shared_indices;
            }
        }

        const auto started = Clock::now();
        auto result = applyTDDs(
            nodes_.at(left_id).tdd, nodes_.at(right_id).tdd, package_);
        const auto stopped = Clock::now();
        contraction_seconds_ +=
            std::chrono::duration<double>(stopped - started).count();

        latest_size_ = static_cast<std::uint64_t>(package_->size(result.e));
        peak_ = std::max(peak_, latest_size_);
        nodes_.at(right_id).tdd = std::move(result);
        nodes_.erase(left_id);
        for (const int candidate_id : incident) {
            edges_.erase(candidate_id);
        }
        for (const auto& [neighbor, shared] : neighbor_shared) {
            auto [u, v] = ordered(neighbor, right_id);
            const int new_id = make_edge_id(u, v);
            edges_[new_id] = ActiveEdge{new_id, u, v, shared};
        }
        ++step_;
        retire_isolated_nodes();
    }

    [[nodiscard]] json result() {
        if (!terminal() || !nodes_.empty() || completed_.empty()) {
            throw std::runtime_error("result requested before complete contraction");
        }
        bool identity = true;
        std::uint64_t final_nodes = 0;
        for (const auto& final_tdd : completed_) {
            identity =
                package_->isTDDIdentity(final_tdd, true, qubits_) && identity;
            final_nodes += static_cast<std::uint64_t>(package_->size(final_tdd.e));
        }
        return {
            {"type", "result"},
            {"correct", identity},
            {"steps", step_},
            {"initial_peak_tdd_nodes", initial_peak_},
            {"peak_tdd_nodes", peak_},
            {"final_tdd_nodes", final_nodes},
            {"completed_components", completed_.size()},
            {"contraction_time_seconds", contraction_seconds_},
            {"precision_bits", precision_bits_},
        };
    }

private:
    void retire_isolated_nodes() {
        std::set<int> incident;
        for (const auto& [edge_id, edge] : edges_) {
            (void)edge_id;
            incident.insert(edge.u);
            incident.insert(edge.v);
        }
        for (auto it = nodes_.begin(); it != nodes_.end();) {
            if (incident.count(it->first)) {
                ++it;
                continue;
            }
            completed_.push_back(std::move(it->second.tdd));
            it = nodes_.erase(it);
        }
    }

    [[nodiscard]] static std::pair<int, int> ordered(const int a, const int b) {
        return a < b ? std::pair<int, int>{a, b} : std::pair<int, int>{b, a};
    }

    [[nodiscard]] int make_edge_id(const int u, const int v) const {
        const std::int64_t value =
            static_cast<std::int64_t>(u) * max_vertex_id_ + static_cast<std::int64_t>(v);
        if (value < 0 || value > std::numeric_limits<int>::max()) {
            throw std::overflow_error("stable edge identifier overflow");
        }
        return static_cast<int>(value);
    }

    int qubits_{};
    int precision_bits_{};
    int max_vertex_id_{1};
    int step_{};
    std::unique_ptr<dd::Package<>> package_;
    std::map<int, ActiveNode> nodes_;
    std::map<int, ActiveEdge> edges_;
    std::vector<dd::TDD> completed_;
    std::uint64_t initial_peak_{};
    std::uint64_t latest_size_{};
    std::uint64_t peak_{};
    double contraction_seconds_{};
};

void emit(const json& value) {
    std::cout << value.dump() << '\n' << std::flush;
}

}  // namespace

int main() {
    try {
        std::string line;
        if (!std::getline(std::cin, line)) {
            throw std::runtime_error("missing case specification");
        }
        Engine engine(json::parse(line));
        emit({
            {"type", "ready"},
            {"observation", engine.observation()},
        });
        if (engine.terminal()) {
            emit(engine.result());
            return 0;
        }
        while (!engine.terminal()) {
            if (!std::getline(std::cin, line)) {
                throw std::runtime_error("planner closed action stream");
            }
            const json request = json::parse(line);
            if (!request.is_object() || !request.contains("action") ||
                !request.at("action").is_number_integer()) {
                throw std::invalid_argument("action request must contain an integer action");
            }
            engine.contract(request.at("action").get<int>());
            if (engine.terminal()) {
                emit(engine.result());
            } else {
                emit({
                    {"type", "step"},
                    {"observation", engine.observation()},
                });
            }
        }
        if (engine.terminal()) {
            return 0;
        }
    } catch (const std::exception& exc) {
        emit({
            {"type", "error"},
            {"error", std::string(typeid(exc).name()) + ": " + exc.what()},
        });
        return 2;
    }
    return 0;
}
