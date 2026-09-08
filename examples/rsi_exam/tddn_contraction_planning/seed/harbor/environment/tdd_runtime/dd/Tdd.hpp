#pragma once


#include "Definitions.hpp"
#include "Edge.hpp"
#include "Node.hpp"
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <map>

namespace dd {

    struct Index {
        std::string key; //
        short idx;

        Index copy() {
            Index copy;
            copy.key = key;
            copy.idx = idx;
        }

        // std::strong_ordering operator<=>(const Index& other) const {
        //     if (auto cmp = idx <=> other.idx; cmp != 0) return cmp;
        //     return key <=> other.key;
        // }

        friend bool operator<(const Index& l, const Index& r) {
            if (l.idx < r.idx) return true;
            else if (l.idx == r.idx) return l.key < r.key;
            return false;
        }

        friend bool operator>(const Index& l, const Index& r) {
            return r < l;
        }

        friend bool operator<=(const Index& l, const Index& r) {
            return !(l > r);
        }

        friend bool operator>=(const Index& l, const Index& r) {
            return !(l < r);
        }

        friend bool operator==(const Index& l, const Index& r) {
            return l.idx == r.idx && l.key == r.key;
        }

        friend bool operator!=(const Index& l, const Index& r) {
            return !(l == r);
        }

    };

    struct GateDef {
        std::string name;
        std::vector<dd::fp> params;

        GateDef copy() {
            GateDef copy;
            copy.name = name;
            copy.params = params;
            return copy;
        }
    };


    struct TDD {

        Edge<mNode> e;
        std::vector<Index> index_set;
        std::vector<std::string> key_2_index;
        std::vector<GateDef> gates;
        float pred_size;

        TDD copy() {
            TDD copy;
            printf("About to copy TDD edge\n");
            copy.e = e.copy();
            printf("Done copying TDD edge\n");
            copy.index_set = {};
            for (int i = 0; i < index_set.size(); i++) {
                copy.index_set.push_back(index_set[i].copy());
            }
            copy.key_2_index = {};
            for (int i = 0; i < key_2_index.size(); i++) {
                copy.key_2_index.push_back(key_2_index[i]);
            }
            copy.gates = {};
            for (int i = 0; i < gates.size(); i++) {
                copy.gates.push_back(gates[i].copy());
            }
            copy.pred_size = pred_size;
            return copy;
        }
    };


    struct key_2_new_key_node {
        short level;
        float new_key;
        std::map<float, key_2_new_key_node*> next;
        key_2_new_key_node* father;
    };

    void to_json(json& j, const GateDef& g) {
        j = json{
            {"name", g.name},
            {"params", g.params}
        };
    }

}
