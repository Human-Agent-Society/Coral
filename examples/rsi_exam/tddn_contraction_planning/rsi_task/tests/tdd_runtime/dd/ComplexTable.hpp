#pragma once

#include "Definitions.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace dd {
template <std::size_t NBUCKET = 65537,
          std::size_t INITIAL_ALLOCATION_SIZE = 2048,
          std::size_t GROWTH_FACTOR = 2, std::size_t INITIAL_GC_LIMIT = 65536>
class ComplexTable {
public:
  struct Entry {
    fp value{};
    Entry* next{};
    RefCount refCount{};

    Entry copy() {
      Entry copy;
      copy.value = value;
      Entry nextCopy = (*next).copy();
      copy.next = &nextCopy;
      copy.refCount = refCount;
    }

    ///
    /// The sign of number is encoded in the least significant bit of its entry
    /// pointer If not handled properly, this causes misaligned access These
    /// routines allow to obtain safe pointers
    ///
    [[nodiscard]] static inline Entry* getAlignedPointer(const Entry* e) {
      return reinterpret_cast<Entry*>(reinterpret_cast<std::uintptr_t>(e) &
                                      ~static_cast<std::uintptr_t>(1U));
    }

    [[nodiscard]] static inline Entry* getNegativePointer(const Entry* e) {
      return reinterpret_cast<Entry*>(reinterpret_cast<std::uintptr_t>(e) |
                                      static_cast<std::uintptr_t>(1U));
    }

    [[nodiscard]] static inline bool exactlyZero(const Entry* e) {
      return (e == &ComplexTable<NBUCKET, INITIAL_ALLOCATION_SIZE,
                                 GROWTH_FACTOR, INITIAL_GC_LIMIT>::zero);
    }

    [[nodiscard]] static inline bool exactlyOne(const Entry* e) {
      return (e == &ComplexTable<NBUCKET, INITIAL_ALLOCATION_SIZE,
                                 GROWTH_FACTOR, INITIAL_GC_LIMIT>::one);
    }

    [[nodiscard]] static inline Entry* flipPointerSign(const Entry* e) {
      if (e == &ComplexTable<NBUCKET, INITIAL_ALLOCATION_SIZE, GROWTH_FACTOR,
                             INITIAL_GC_LIMIT>::zero) {
        // No point in flipping the sign of 0
        return reinterpret_cast<Entry*>(reinterpret_cast<std::uintptr_t>(e));
      }
      return reinterpret_cast<Entry*>(reinterpret_cast<std::uintptr_t>(e) ^
                                      static_cast<std::uintptr_t>(1U));
    }

    [[nodiscard]] static inline bool isNegativePointer(const Entry* e) {
      return (reinterpret_cast<std::uintptr_t>(e) &
              static_cast<std::uintptr_t>(1U)) != 0U;
    }

    [[nodiscard]] static inline fp val(const Entry* e) {
      if (isNegativePointer(e)) {
        return -getAlignedPointer(e)->value;
      }
      return e->value;
    }

    [[nodiscard]] static inline RefCount ref(const Entry* e) {
      if (isNegativePointer(e)) {
        return -getAlignedPointer(e)->refCount;
      }
      return e->refCount;
    }

    [[nodiscard]] static constexpr bool
    approximatelyEquals(const Entry* left, const Entry* right) {
      return left == right || approximatelyEquals(val(left), val(right));
    }
    [[nodiscard]] static constexpr bool approximatelyEquals(const fp left,
                                                            const fp right) {
      // equivalence check is a shortcut before check with tolerance
      // NOLINTNEXTLINE(clang-diagnostic-float-equal)
      return left == right || std::abs(left - right) <= TOLERANCE;
    }

    [[nodiscard]] static constexpr bool approximatelyZero(const Entry* e) {
      return e == &zero || approximatelyZero(val(e));
    }
    [[nodiscard]] static constexpr bool approximatelyZero(const fp e) {
      return std::abs(e) <= TOLERANCE;
    }

    [[nodiscard]] static constexpr bool approximatelyOne(const Entry* e) {
      return e == &one || approximatelyOne(val(e));
    }
    [[nodiscard]] static constexpr bool approximatelyOne(fp e) {
      return approximatelyEquals(e, 1.0);
    }

    static void writeBinary(const Entry* e, std::ostream& os) {
      auto temp = val(e);
      os.write(reinterpret_cast<const char*>(&temp), sizeof(decltype(temp)));
    }
  };

  // NOLINTNEXTLINE(readability-identifier-naming,cppcoreguidelines-avoid-non-const-global-variables)
  static inline Entry zero{0., nullptr, 1};
  // NOLINTNEXTLINE(readability-identifier-naming,cppcoreguidelines-avoid-non-const-global-variables)
  static inline Entry sqrt2_2{SQRT2_2, nullptr, 1};
  // NOLINTNEXTLINE(readability-identifier-naming,cppcoreguidelines-avoid-non-const-global-variables)
  static inline Entry one{1., nullptr, 1};

  ComplexTable() {
    // add 1/2 to the complex table and increase its ref count (so that it is
    // not collected)
    lookup(0.5L)->refCount++;
  }

  ~ComplexTable() = default;

  static fp tolerance() { return TOLERANCE; }

  static void setTolerance(fp tol) { TOLERANCE = tol; }

  static constexpr std::int64_t MASK = NBUCKET - 1;

  // linear (clipped) hash function
  static constexpr std::int64_t hash(const fp val) {
    assert(val >= 0);
    auto key = static_cast<std::int64_t>(std::nearbyint(val * MASK));
    return std::min<std::int64_t>(key, MASK);
  }

  // access functions
  [[nodiscard]] std::size_t getCount() const { return count; }

  [[nodiscard]] std::size_t getPeakCount() const { return peakCount; }

  [[nodiscard]] std::size_t getAllocations() const { return allocations; }

  [[nodiscard]] std::size_t getGrowthFactor() const { return GROWTH_FACTOR; }

  [[nodiscard]] const auto& getTable() const { return table; }

  [[nodiscard]] bool availableEmpty() const { return available == nullptr; };

  Entry* lookup(const fp& val) {
    assert(!std::isnan(val));
    assert(val >= 0); // required anyway for the hash function
    ++lookups;
    if (Entry::approximatelyZero(val)) {
      ++hits;
      return &zero;
    }

    if (Entry::approximatelyOne(val)) {
      ++hits;
      return &one;
    }

    if (Entry::approximatelyEquals(val, SQRT2_2)) {
      ++hits;
      return &sqrt2_2;
    }

    assert(val - TOLERANCE >= 0); // should be handle above as special case

    const auto lowerKey = hash(val - TOLERANCE);
    const auto upperKey = hash(val + TOLERANCE);

    if (upperKey == lowerKey) {
      ++findOrInserts;
      return findOrInsert(lowerKey, val);
    }

    // code below is to properly handle border cases |----(-|-)----|
    // in case a value close to a border is looked up,
    // only the last entry in the lower bucket and the first entry in the upper
    // bucket need to be checked

    const auto key = hash(val);

    Entry* pLower; // NOLINT(cppcoreguidelines-init-variables)
    Entry* pUpper; // NOLINT(cppcoreguidelines-init-variables)
    if (lowerKey != key) {
      pLower = tailTable[static_cast<std::size_t>(lowerKey)];
      pUpper = table[static_cast<std::size_t>(key)];
      ++lowerNeighbors;
    } else {
      pLower = tailTable[static_cast<std::size_t>(key)];
      pUpper = table[static_cast<std::size_t>(upperKey)];
      ++upperNeighbors;
    }

    bool lowerMatchFound =
        (pLower != nullptr && Entry::approximatelyEquals(val, pLower->value));
    bool upperMatchFound =
        (pUpper != nullptr && Entry::approximatelyEquals(val, pUpper->value));

    if (lowerMatchFound && upperMatchFound) {
      ++hits;
      const auto diffToLower = std::abs(pLower->value - val);
      const auto diffToUpper = std::abs(pUpper->value - val);
      // val is actually closer to p_lower than to p_upper
      if (diffToLower < diffToUpper) {
        return pLower;
      }
      return pUpper;
    }

    if (lowerMatchFound) {
      ++hits;
      return pLower;
    }

    if (upperMatchFound) {
      ++hits;
      return pUpper;
    }

    // value was not found in the table -> get a new entry and add it to the
    // central bucket
    Entry* entry = insert(key, val);
    return entry;
  }

  [[nodiscard]] Entry* getEntry() {
    // an entry is available on the stack
    if (!availableEmpty()) {
      Entry* entry = available;
      available = entry->next;
      // returned entries could have a ref count != 0
      entry->refCount = 0;
      return entry;
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

    auto entry = &(*chunkIt);
    ++chunkIt;
    return entry;
  }

  void returnEntry(Entry* entry) {
    entry->next = available;
    available = entry;
  }

  // increment reference count for corresponding entry
  static void incRef(Entry* entry) {
    // get valid pointer
    auto entryPtr = Entry::getAlignedPointer(entry);

    if (entryPtr == nullptr) {
      return;
    }

    // important (static) numbers are never altered
    if (entryPtr != &one && entryPtr != &zero && entryPtr != &sqrt2_2) {
      if (entryPtr->refCount == std::numeric_limits<RefCount>::max()) {
        std::clog << "[WARN] MAXREFCNT reached for " << entryPtr->value
                  << ". Number will never be collected." << std::endl;
        return;
      }

      // increase reference count
      entryPtr->refCount++;
    }
  }

  // decrement reference count for corresponding entry
  static void decRef(Entry* entry) {
    // get valid pointer
    auto entryPtr = Entry::getAlignedPointer(entry);

    if (entryPtr == nullptr) {
      return;
    }

    // important (static) numbers are never altered
    if (entryPtr != &one && entryPtr != &zero && entryPtr != &sqrt2_2) {
      if (entryPtr->refCount == std::numeric_limits<RefCount>::max()) {
        return;
      }
      if (entryPtr->refCount == 0) {
        throw std::runtime_error("In ComplexTable: RefCount of entry " +
                                 std::to_string(entryPtr->value) +
                                 " is zero before decrement");
      }

      // decrease reference count
      entryPtr->refCount--;
    }
  }

  [[nodiscard]] bool possiblyNeedsCollection() const {
    return count >= gcLimit;
  }

  std::size_t garbageCollect(bool force = false) {
    gcCalls++;
    // nothing to be done if garbage collection is not forced, and the limit has
    // not been reached, or the current count is minimal (the complex table
    // always contains at least 0.5)
    if ((!force && count < gcLimit) || count <= 1) {
      return 0;
    }

    gcRuns++;
    std::size_t collected = 0;
    std::size_t remaining = 0;
    for (std::size_t key = 0; key < table.size(); ++key) {
      Entry* p = table[key];
      Entry* lastp = nullptr;
      while (p != nullptr) {
        if (p->refCount == 0) {
          Entry* next = p->next;
          if (lastp == nullptr) {
            table[key] = next;
          } else {
            lastp->next = next;
          }
          returnEntry(p);
          p = next;
          collected++;
        } else {
          lastp = p;
          p = p->next;
          remaining++;
        }
        tailTable[key] = lastp;
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
    } else if (remaining < gcLimit / 128) {
      gcLimit /= 2;
    }
    count = remaining;
    return collected;
  }

  void clear() {
    // clear table buckets
    for (auto& bucket : table) {
      bucket = nullptr;
    }
    for (auto& entry : tailTable) {
      entry = nullptr;
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

    for (auto& entry : chunks[0]) {
      entry.refCount = 0;
    }

    count = 0;
    peakCount = 0;

    collisions = 0;
    insertCollisions = 0;
    hits = 0;
    findOrInserts = 0;
    lookups = 0;
    inserts = 0;
    lowerNeighbors = 0;
    upperNeighbors = 0;

    gcCalls = 0;
    gcRuns = 0;
    gcLimit = INITIAL_GC_LIMIT;
  };

  void print() {
    const auto precision = std::cout.precision();
    std::cout.precision(std::numeric_limits<dd::fp>::max_digits10);
    for (std::size_t key = 0; key < table.size(); ++key) {
      auto p = table[key];
      if (p != nullptr) {
        std::cout << key << ": "
                  << "\n";
      }

      while (p != nullptr) {
        std::cout << "\t\t" << p->value << " "
                  << reinterpret_cast<std::uintptr_t>(p) << " " << p->refCount
                  << "\n";
        p = p->next;
      }

      if (table[key] != nullptr) {
        std::cout << "\n";
      }
    }
    std::cout.precision(precision);
  }

  [[nodiscard]] fp hitRatio() const {
    return static_cast<fp>(hits) / static_cast<fp>(lookups);
  }

  [[nodiscard]] fp colRatio() const {
    return static_cast<fp>(collisions) / static_cast<fp>(lookups);
  }

  std::map<std::string, std::size_t, std::less<>> getStatistics() {
    return {
        {"hits", hits},
        {"collisions", collisions},
        {"lookups", lookups},
        {"inserts", inserts},
        {"insertCollisions", insertCollisions},
        {"findOrInserts", findOrInserts},
        {"upperNeighbors", upperNeighbors},
        {"lowerNeighbors", lowerNeighbors},
        {"gcCalls", gcCalls},
        {"gcRuns", gcRuns},
    };
  }

  std::ostream& printStatistics(std::ostream& os = std::cout) {
    // clang-format off
            os << "hits: " << hits
               << ", collisions: " << collisions
               << ", looks: " << lookups
               << ", inserts: " << inserts
               << ", insertCollisions: " << insertCollisions
               << ", findOrInserts: " << findOrInserts
               << ", upperNeighbors: " << upperNeighbors
               << ", lowerNeighbors: " << lowerNeighbors
               << ", hitRatio: " << hitRatio()
               << ", colRatio: " << colRatio()
               << ", gc calls: " << gcCalls
               << ", gc runs: " << gcRuns
               << "\n";
    // clang-format on
    return os;
  }

  std::ostream& printBucketDistribution(std::ostream& os = std::cout) {
    for (auto bucket : table) {
      if (bucket == nullptr) {
        os << "0\n";
        continue;
      }
      std::size_t bucketCount = 0;
      while (bucket != nullptr) {
        ++bucketCount;
        bucket = bucket->next;
      }
      os << bucketCount << "\n";
    }
    os << std::endl;
    return os;
  }

private:
  using Bucket = Entry*;
  using Table = std::array<Bucket, NBUCKET>;

  Table table{};

  std::array<Entry*, NBUCKET> tailTable{};

  // table lookup statistics
  std::size_t collisions = 0;
  std::size_t insertCollisions = 0;
  std::size_t hits = 0;
  std::size_t findOrInserts = 0;
  std::size_t lookups = 0;
  std::size_t inserts = 0;
  std::size_t lowerNeighbors = 0;
  std::size_t upperNeighbors = 0;

  // numerical tolerance to be used for floating point values
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,readability-identifier-naming)
  static inline fp TOLERANCE = pow(2.0, -18.0);//std::numeric_limits<dd::fp>::epsilon() * 8;//0.00000381469726563f;//std::numeric_limits<dd::fp>::epsilon() * 8;

  Entry* available{};
  std::vector<std::vector<Entry>> chunks{
      1, std::vector<Entry>{INITIAL_ALLOCATION_SIZE}};
  std::size_t chunkID{};
  typename std::vector<Entry>::iterator chunkIt{chunks.at(0).begin()};
  typename std::vector<Entry>::iterator chunkEndIt{chunks.at(0).end()};
  std::size_t allocationSize{INITIAL_ALLOCATION_SIZE * GROWTH_FACTOR};

  std::size_t allocations = INITIAL_ALLOCATION_SIZE;
  std::size_t count = 0;
  std::size_t peakCount = 0;

  // garbage collection
  std::size_t gcCalls = 0;
  std::size_t gcRuns = 0;
  std::size_t gcLimit = 100000;

  inline Entry* findOrInsert(const std::int64_t key, const fp val) {
    [[maybe_unused]] const fp valTol = val + TOLERANCE;

    Entry* curr = table[static_cast<std::size_t>(key)];
    Entry* prev = nullptr;

    while (curr != nullptr && curr->value <= valTol) {
      if (Entry::approximatelyEquals(curr->value, val)) {
        // check if val is actually closer to the next element in the list (if
        // there is one)
        if (curr->next != nullptr) {
          const auto& next = curr->next;
          // potential candidate in range
          if (valTol >= next->value) {
            const auto diffToCurr = std::abs(curr->value - val);
            const auto diffToNext = std::abs(next->value - val);
            // val is actually closer to next than to curr
            if (diffToNext < diffToCurr) {
              ++hits;
              return next;
            }
          }
        }
        ++hits;
        return curr;
      }
      ++collisions;
      prev = curr;
      curr = curr->next;
    }

    ++inserts;
    Entry* entry = getEntry();
    entry->value = val;

    if (prev == nullptr) {
      // table bucket is empty
      table[static_cast<std::size_t>(key)] = entry;
    } else {
      prev->next = entry;
    }
    entry->next = curr;
    if (curr == nullptr) {
      tailTable[static_cast<std::size_t>(key)] = entry;
    }
    count++;
    peakCount = std::max(peakCount, count);
    return entry;
  }

  /**
   * Inserts a value into the bucket indexed by key. This function assumes no
   * element within TOLERANCE is present in the bucket.
   * @param key index to the bucket
   * @param val value to be inserted
   * @return pointer to the inserted entry
   */
  inline Entry* insert(const std::int64_t key, const fp val) {
    ++inserts;
    Entry* entry = getEntry();
    entry->value = val;

    Entry* curr = table[static_cast<std::size_t>(key)];
    Entry* prev = nullptr;

    while (curr != nullptr && curr->value <= val) {
      ++insertCollisions;
      prev = curr;
      curr = curr->next;
    }

    if (prev == nullptr) {
      // table bucket is empty
      table[static_cast<std::size_t>(key)] = entry;
    } else {
      prev->next = entry;
    }
    entry->next = curr;
    if (curr == nullptr) {
      tailTable[static_cast<std::size_t>(key)] = entry;
    }
    count++;
    peakCount = std::max(peakCount, count);
    return entry;
  }
};
} // namespace dd
