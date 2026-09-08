#pragma once

#include "ComplexNumbers.hpp"
#include "Definitions.hpp"



#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace dd {

/// Data structure for providing and uniquely storing DD nodes
/// \tparam Node class of nodes to provide/store
/// \tparam NBUCKET number of hash buckets to use (has to be a power of two)
/// \tparam INITIAL_ALLOCATION_SIZE number if nodes initially allocated
/// \tparam GROWTH_PERCENTAGE percentage that the allocations' size shall grow
/// over time \tparam INITIAL_GC_LIMIT number of nodes initially used as garbage
/// collection threshold \tparam GC_INCREMENT absolute number of nodes to
/// increase the garbage collection threshold after garbage collection has been
/// performed
template <class Node, std::size_t NBUCKET = 32768,
          std::size_t INITIAL_ALLOCATION_SIZE = 2048,
          std::size_t GROWTH_FACTOR = 2, std::size_t INITIAL_GC_LIMIT = 131072>
class UniqueTable {
public:
  explicit UniqueTable(const std::size_t nv) : nvars(nv) {}

  ~UniqueTable() = default;

  static constexpr std::size_t MASK = NBUCKET - 1;

  void resize(std::size_t nq) {
    nvars = nq;
    tables.resize(nq);
    // TODO: if the new size is smaller than the old one we might have to
    // release the unique table entries for the superfluous variables
    active.resize(nq);
    activeNodeCount = std::accumulate(active.begin(), active.end(),
                                      static_cast<std::size_t>(0U));
  }

  static std::size_t hash(const Node* p) {
    std::size_t key = 0;
    for (std::size_t i = 0; i < p->e.size(); ++i) {
      key = dd::combineHash(key, std::hash<Edge<Node>>{}(p->e[i]));
      // old hash function:
      //     key += ((reinterpret_cast<std::size_t>(p->e[i].p)   >>  i) +
      //             (reinterpret_cast<std::size_t>(p->e[i].w.r) >>  i) +
      //             (reinterpret_cast<std::size_t>(p->e[i].w.i) >> (i + 1))) &
      //             MASK;
    }
    key &= MASK;
    return key;
  }

  // access functions
  [[nodiscard]] std::size_t getNodeCount() const { return nodeCount; }

  [[nodiscard]] std::size_t getPeakNodeCount() const { return peakNodeCount; }

  [[nodiscard]] std::size_t getMaxActiveNodes() const { return maxActive; }

  [[nodiscard]] std::size_t getAllocations() const { return allocations; }

  [[nodiscard]] float getGrowthFactor() const { return GROWTH_FACTOR; }

  [[nodiscard]] const auto& getTables() const { return tables; }

  static bool nodesAreEqual(const Node* p, const Node* q) {

      return p->e == q->e;
    
  }

  // lookup a node in the unique table for the appropriate variable; insert it,
  // if it has not been found NOTE: reference counting is to be adjusted by
  // function invoking the table lookup and only normalized nodes shall be
  // stored.
  Edge<Node> lookup(const Edge<Node>& e, bool keepNode = false) {
    // there are unique terminal nodes
    if (e.isTerminal()) {
      return e;
    }

    ++lookups;
    const auto key = hash(e.p);
    const auto v = e.p->v;

    // search bucket in table corresponding to hashed value for the given node
    // and return it if found.
    if (const auto hashedNode = searchTable(e, key, keepNode);
        hashedNode != Edge<Node>::zero) {

      return hashedNode;
    }

    // if node not found -> add it to front of unique table bucket
    e.p->next = tables[static_cast<std::size_t>(v)][key];
    tables[static_cast<std::size_t>(v)][key] = e.p;
    ++nodeCount;
    peakNodeCount = std::max(peakNodeCount, nodeCount);

    return e;
  }

  [[nodiscard]] Node* getNode() {
    // a node is available on the stack
    if (available != nullptr) {
      Node* p = available;
      available = p->next;
      // returned nodes could have a ref count != 0
      p->ref = 0;
      return p;
    }

    // new chunk has to be allocated
    if (chunkIt == chunkEndIt) {
      chunks.emplace_back(allocationSize);
      allocations += allocationSize;
      allocationSize *= GROWTH_FACTOR;
      chunkID++;
      chunkIt = chunks[chunkID].begin();
      chunkEndIt = chunks[chunkID].end();
    }

    auto p = &(*chunkIt);
    ++chunkIt;
    return p;
  }

  void returnNode(Node* p) {
    p->next = available;
    available = p;
  }

  // increment reference counter for node e points to
  // and recursively increment reference counter for
  // each child if this is the first reference
  void incRef(const Edge<Node>& e) {
    dd::ComplexNumbers::incRef(e.w);
    if (e.p == nullptr || e.isTerminal()) {
      return;
    }

    if (e.p->ref == std::numeric_limits<decltype(e.p->ref)>::max()) {
      std::clog << "[WARN] MAXREFCNT reached for p="
                << reinterpret_cast<std::uintptr_t>(e.p)
                << ". Node will never be collected." << std::endl;
      return;
    }

    e.p->ref++;

    if (e.p->ref == 1) {
      for (const auto& edge : e.p->e) {
        if (edge.p != nullptr) {
          incRef(edge);
        }
      }
      active[static_cast<std::size_t>(e.p->v)]++;
      activeNodeCount++;
      maxActive = std::max(maxActive, activeNodeCount);
    }
  }

  // decrement reference counter for node e points to
  // and recursively decrement reference counter for
  // each child if this is the last reference
  void decRef(const Edge<Node>& e) {
    dd::ComplexNumbers::decRef(e.w);
    if (e.p == nullptr || e.isTerminal()) {
      return;
    }
    if (e.p->ref == std::numeric_limits<decltype(e.p->ref)>::max()) {
      return;
    }

    if (e.p->ref == 0) {
      throw std::runtime_error("In decref: ref==0 before decref\n");
    }

    e.p->ref--;

    if (e.p->ref == 0) {
      for (const auto& edge : e.p->e) {
        if (edge.p != nullptr) {
          decRef(edge);
        }
      }
      active[static_cast<std::size_t>(e.p->v)]--;
      activeNodeCount--;
    }
  }

  [[nodiscard]] bool possiblyNeedsCollection() const {
    return nodeCount >= gcLimit;
  }

  std::size_t garbageCollect(bool force = false) {
    gcCalls++;
    if ((!force && nodeCount < gcLimit) || nodeCount == 0) {
      return 0;
    }

    gcRuns++;
    std::size_t collected = 0;
    std::size_t remaining = 0;
    for (auto& table : tables) {
      for (auto& bucket : table) {
        Node* p = bucket;
        Node* lastp = nullptr;
        while (p != nullptr) {
          if (p->ref == 0) {
            assert(!Node::isTerminal(p));
            Node* next = p->next;
            if (lastp == nullptr) {
              bucket = next;
            } else {
              lastp->next = next;
            }
            returnNode(p);
            p = next;
            collected++;
          } else {
            lastp = p;
            p = p->next;
            remaining++;
          }
        }
      }
    }
    // The garbage collection limit changes dynamically depending on the number
    // of remaining (active) nodes. If it were not changed, garbage collection
    // would run through the complete table on each successive call once the
    // number of remaining entries reaches the garbage collection limit. It is
    // increased whenever the number of remaining entries is rather close to the
    // garbage collection threshold and decreased if the number of remaining
    // entries is much lower than the current limit.
    if (remaining > gcLimit / 10 * 9) {
      gcLimit = remaining + INITIAL_GC_LIMIT;
    }
    nodeCount = remaining;
    return collected;
  }

  void clear() {
    // clear unique table buckets
    for (auto& table : tables) {
      for (auto& bucket : table) {
        bucket = nullptr;
      }
    }
    // clear available stack
    available = nullptr;

    // release memory of all but the first chunk TODO: it could be desirable to
    // keep the memory
    while (chunkID > 0) {
      chunks.pop_back();
      chunkID--;
    }
    // restore initial chunk setting
    chunkIt = chunks[0].begin();
    chunkEndIt = chunks[0].end();
    allocationSize = INITIAL_ALLOCATION_SIZE * GROWTH_FACTOR;
    allocations = INITIAL_ALLOCATION_SIZE;

    for (auto& node : chunks[0]) {
      node.ref = 0;
    }

    nodeCount = 0;
    peakNodeCount = 0;

    collisions = 0;
    hits = 0;
    lookups = 0;

    std::fill(active.begin(), active.end(), 0);
    activeNodeCount = 0;
    maxActive = 0;

    gcCalls = 0;
    gcRuns = 0;
    gcLimit = INITIAL_GC_LIMIT;
  };

  void print() {
    auto q = static_cast<dd::Qubit>(nvars - 1);
    for (auto it = tables.rbegin(); it != tables.rend(); ++it) {
      auto& table = *it;
      std::cout << "\tq" << static_cast<std::size_t>(q) << ":"
                << "\n";
      for (std::size_t key = 0; key < table.size(); ++key) {
        auto p = table[key];
        if (p != nullptr) {
          std::cout << "\tkey=" << key << ": ";
        }

        while (p != nullptr) {
          std::cout << "\t\t" << std::hex << reinterpret_cast<std::uintptr_t>(p)
                    << std::dec << " " << p->ref << std::hex;
          for (const auto& e : p->e) {
            std::cout << " p" << reinterpret_cast<std::uintptr_t>(e.p) << "(r"
                      << reinterpret_cast<std::uintptr_t>(e.w.r) << " i"
                      << reinterpret_cast<std::uintptr_t>(e.w.i) << ")";
          }
          std::cout << std::dec << "\n";
          p = p->next;
        }
      }
      --q;
    }
  }

  void printActive() {
    std::cout << "#printActive: " << activeNodeCount << ", ";
    for (const auto& a : active) {
      std::cout << a << " ";
    }
    std::cout << "\n";
  }

  [[nodiscard]] fp hitRatio() const {
    return static_cast<fp>(hits) / static_cast<fp>(lookups);
  }

  [[nodiscard]] fp colRatio() const {
    return static_cast<fp>(collisions) / static_cast<fp>(lookups);
  }

  [[nodiscard]] std::size_t getActiveNodeCount() const {
    return activeNodeCount;
  }

  [[nodiscard]] std::size_t getActiveNodeCount(Qubit var) {
    return active.at(var);
  }

  std::ostream& printStatistics(std::ostream& os = std::cout) {
    os << "hits: " << hits << ", collisions: " << collisions
       << ", looks: " << lookups << ", hitRatio: " << hitRatio()
       << ", colRatio: " << colRatio() << ", gc calls: " << gcCalls
       << ", gc runs: " << gcRuns << "\n";
    return os;
  }

private:
  using NodeBucket = Node*;
  using Table = std::array<NodeBucket, NBUCKET>;

  // unique tables (one per input variable)
  std::size_t nvars = 0;
  std::vector<Table> tables{nvars};

  Node* available{};
  std::vector<std::vector<Node>> chunks{
      1, std::vector<Node>{INITIAL_ALLOCATION_SIZE}};
  std::size_t chunkID{0};
  typename std::vector<Node>::iterator chunkIt{chunks[0].begin()};
  typename std::vector<Node>::iterator chunkEndIt{chunks[0].end()};
  std::size_t allocationSize{INITIAL_ALLOCATION_SIZE * GROWTH_FACTOR};

  std::size_t allocations = INITIAL_ALLOCATION_SIZE;
  std::size_t nodeCount = 0;
  std::size_t peakNodeCount = 0;

  // unique table lookup statistics
  std::size_t collisions = 0;
  std::size_t hits = 0;
  std::size_t lookups = 0;

  // (max) active nodes
  // number of active vector nodes for each variable
  std::vector<std::size_t> active{std::vector<std::size_t>(nvars, 0)};
  std::size_t activeNodeCount = 0;
  std::size_t maxActive = 0;

  // garbage collection
  std::size_t gcCalls = 0;
  std::size_t gcRuns = 0;
  std::size_t gcLimit = INITIAL_GC_LIMIT;

  /**
  Searches for a node in the hash table with the given key.
  @param e The node to search for.
  @param key The hashed value used to search the table.
  @param keepNode If true, the node pointed to by e.p will not be put on the
  available chain.
  @return The Edge<Node> found in the hash table or Edge<Node>::zero if not
  found.
  **/
  Edge<Node> searchTable(const Edge<Node>& e, const std::size_t& key,
                         const bool keepNode = false) {
    const auto v = e.p->v;

    Node* p = tables[static_cast<std::size_t>(v)][key];

    while (p != nullptr) {
      if (nodesAreEqual(e.p, p)) {
        // Match found
        if (e.p != p && !keepNode) {
          // put node pointed to by e.p on available chain
          returnNode(e.p);
        }
        ++hits;
        // variables should stay the same
        assert(p->v == e.p->v);
        return {p, e.w};
      }
      ++collisions;
      p = p->next;
    }

    // Node not found in bucket
    return Edge<Node>::zero;
  }
};

} // namespace dd
