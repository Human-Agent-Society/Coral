#pragma once

#include "ComplexTable.hpp"
#include "Definitions.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

namespace dd {
struct ComplexValue {
  fp r, i;

  [[nodiscard]] constexpr bool
  approximatelyEquals(const ComplexValue& c) const {
    return ComplexTable<>::Entry::approximatelyEquals(r, c.r) &&
           ComplexTable<>::Entry::approximatelyEquals(i, c.i);
  }

  [[nodiscard]] constexpr bool approximatelyZero() const {
    return ComplexTable<>::Entry::approximatelyZero(r) &&
           ComplexTable<>::Entry::approximatelyZero(i);
  }

  [[nodiscard]] constexpr bool approximatelyOne() const {
    return ComplexTable<>::Entry::approximatelyOne(r) &&
           ComplexTable<>::Entry::approximatelyZero(i);
  }

  constexpr bool operator==(const ComplexValue& other) const {
    // NOLINTNEXTLINE(clang-diagnostic-float-equal)
    return r == other.r && i == other.i;
  }

  constexpr bool operator!=(const ComplexValue& other) const {
    return !operator==(other);
  }

  void readBinary(std::istream& is) {
    is.read(reinterpret_cast<char*>(&r), sizeof(decltype(r)));
    is.read(reinterpret_cast<char*>(&i), sizeof(decltype(i)));
  }

  void writeBinary(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&r), sizeof(decltype(r)));
    os.write(reinterpret_cast<const char*>(&i), sizeof(decltype(i)));
  }

  void fromString(const std::string& realStr, std::string imagStr) {
    const fp real = realStr.empty() ? 0. : std::stod(realStr);

    imagStr.erase(remove(imagStr.begin(), imagStr.end(), ' '), imagStr.end());
    imagStr.erase(remove(imagStr.begin(), imagStr.end(), 'i'), imagStr.end());
    if (imagStr == "+" || imagStr == "-") {
      imagStr = imagStr + "1";
    }
    const fp imag = imagStr.empty() ? 0. : std::stod(imagStr);
    r = {real};
    i = {imag};
  }

  static auto
  getLowestFraction(const double x,
                    const std::uint64_t maxDenominator = 1U << 10,
                    const fp tol = dd::ComplexTable<>::tolerance()) {
    assert(x >= 0.);

    std::pair<std::uint64_t, std::uint64_t> lowerBound{0U, 1U};
    std::pair<std::uint64_t, std::uint64_t> upperBound{1U, 0U};

    while ((lowerBound.second <= maxDenominator) &&
           (upperBound.second <= maxDenominator)) {
      auto num = lowerBound.first + upperBound.first;
      auto den = lowerBound.second + upperBound.second;
      auto median = static_cast<fp>(num) / static_cast<fp>(den);
      if (std::abs(x - median) <= tol) {
        if (den <= maxDenominator) {
          return std::pair{num, den};
        }
        if (upperBound.second > lowerBound.second) {
          return upperBound;
        }
        return lowerBound;
      }
      if (x > median) {
        lowerBound = {num, den};
      } else {
        upperBound = {num, den};
      }
    }
    if (lowerBound.second > maxDenominator) {
      return upperBound;
    }
    return lowerBound;
  }

  static void printFormatted(std::ostream& os, fp num, bool imaginary = false) {
    if (std::abs(num) <= ComplexTable<>::tolerance()) {
      os << (std::signbit(num) ? "-" : "+") << "0" << (imaginary ? "i" : "");
      return;
    }

    const auto absnum = std::abs(num);
    auto fraction = getLowestFraction(absnum);
    auto approx =
        static_cast<fp>(fraction.first) / static_cast<fp>(fraction.second);
    auto error = std::abs(absnum - approx);

    if (error <= ComplexTable<>::tolerance()) { // suitable fraction a/b found
      const std::string sign = std::signbit(num) ? "-" : (imaginary ? "+" : "");

      if (fraction.first == 1U && fraction.second == 1U) {
        os << sign << (imaginary ? "i" : "1");
      } else if (fraction.second == 1U) {
        os << sign << fraction.first << (imaginary ? "i" : "");
      } else if (fraction.first == 1U) {
        os << sign << (imaginary ? "i" : "1") << "/" << fraction.second;
      } else {
        os << sign << fraction.first << (imaginary ? "i" : "") << "/"
           << fraction.second;
      }

      return;
    }

    const auto abssqrt = absnum / SQRT2_2;
    fraction = getLowestFraction(abssqrt);
    approx = static_cast<fp>(fraction.first) / static_cast<fp>(fraction.second);
    error = std::abs(abssqrt - approx);

    if (error <= ComplexTable<>::tolerance()) { // suitable fraction a/(b *
                                                // sqrt(2)) found
      const std::string sign = std::signbit(num) ? "-" : (imaginary ? "+" : "");

      if (fraction.first == 1U && fraction.second == 1U) {
        os << sign << (imaginary ? "i" : "1") << "/√2";
      } else if (fraction.second == 1U) {
        os << sign << fraction.first << (imaginary ? "i" : "") << "/√2";
      } else if (fraction.first == 1U) {
        os << sign << (imaginary ? "i" : "1") << "/(" << fraction.second
           << "√2)";
      } else {
        os << sign << fraction.first << (imaginary ? "i" : "") << "/("
           << fraction.second << "√2)";
      }
      return;
    }

    const auto abspi = absnum / PI;
    fraction = getLowestFraction(abspi);
    approx = static_cast<fp>(fraction.first) / static_cast<fp>(fraction.second);
    error = std::abs(abspi - approx);

    if (error <= ComplexTable<>::tolerance()) { // suitable fraction a/b π found
      const std::string sign = std::signbit(num) ? "-" : (imaginary ? "+" : "");
      const std::string imagUnit = imaginary ? "i" : "";

      if (fraction.first == 1U && fraction.second == 1U) {
        os << sign << "π" << imagUnit;
      } else if (fraction.second == 1U) {
        os << sign << fraction.first << "π" << imagUnit;
      } else if (fraction.first == 1U) {
        os << sign << "π" << imagUnit << "/" << fraction.second;
      } else {
        os << sign << fraction.first << "π" << imagUnit << "/"
           << fraction.second;
      }
      return;
    }

    if (imaginary) { // default
      os << (std::signbit(num) ? "" : "+") << num << "i";
    } else {
      os << num;
    }
  }

  static std::string toString(const fp& real, const fp& imag,
                              bool formatted = true, int precision = -1) {
    std::ostringstream ss{};

    if (precision >= 0) {
      ss << std::setprecision(precision);
    }
    const auto tol = ComplexTable<>::tolerance();

    if (std::abs(real) <= tol && std::abs(imag) <= tol) {
      return "0";
    }

    if (std::abs(real) > tol) {
      if (formatted) {
        printFormatted(ss, real);
      } else {
        ss << real;
      }
    }
    if (std::abs(imag) > tol) {
      if (formatted) {
        if (std::abs(real - imag) <= tol) {
          ss << "(1+i)";
          return ss.str();
        }
        if (std::abs(real + imag) <= tol) {
          ss << "(1-i)";
          return ss.str();
        }
        printFormatted(ss, imag, true);
      } else {
        if (std::abs(real) <= tol) {
          ss << imag;
        } else {
          if (imag > 0.) {
            ss << "+";
          }
          ss << imag;
        }
        ss << "i";
      }
    }

    return ss.str();
  }

  explicit operator auto() const { return std::complex<dd::fp>{r, i}; }

  ComplexValue& operator+=(const ComplexValue& rhs) {
    r += rhs.r;
    i += rhs.i;
    return *this;
  }

  friend ComplexValue operator+(ComplexValue lhs, const ComplexValue& rhs) {
    lhs += rhs;
    return lhs;
  }
};

inline std::ostream& operator<<(std::ostream& os, const ComplexValue& c) {
  return os << ComplexValue::toString(c.r, c.i);
}
} // namespace dd

namespace std {
template <> struct hash<dd::ComplexValue> {
  std::size_t operator()(dd::ComplexValue const& c) const noexcept {
    auto h1 = dd::murmur64(static_cast<std::size_t>(
        std::round(c.r / dd::ComplexTable<>::tolerance())));
    auto h2 = dd::murmur64(static_cast<std::size_t>(
        std::round(c.i / dd::ComplexTable<>::tolerance())));
    return dd::combineHash(h1, h2);
  }
};
} // namespace std
