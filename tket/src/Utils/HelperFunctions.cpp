// Copyright Quantinuum
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tket/Utils/HelperFunctions.hpp"

#include <boost/dynamic_bitset.hpp>

namespace tket {

GrayCode gen_graycode(unsigned m_controls) {
  const unsigned n = m_controls;
  if (m_controls == 0) return {};
  GrayCode gc{{0}, {1}};

  for (unsigned i = 2; i < (1u << n); i = i << 1) {
    for (unsigned j = 0; j < i; ++j) {
      gc.push_back(gc[i - 1 - j]);
    }
    for (unsigned j = 0; j < i; ++j) {
      gc[j].push_back(0);
    }
    for (unsigned j = i; j < 2 * i; ++j) {
      gc[j].push_back(1);
    }
  }
  return gc;
}

uint64_t reverse_bits(uint64_t v, unsigned w) {
  uint64_t r = 0;
  for (unsigned i = w; i--;) {
    r |= (v & 1) << i;
    v >>= 1;
  }
  return r;
}

std::vector<bool> dec_to_bin(unsigned long long dec, unsigned width) {
  auto bs = boost::dynamic_bitset<>(width, dec);
  std::vector<bool> bits(width);
  for (unsigned long long i = 0; i < width; i++) {
    bits[width - i - 1] = bs[i];
  }
  return bits;
}

// big endian
unsigned long long bin_to_dec(const std::vector<bool> &bin) {
  unsigned long long res = 0;
  for (unsigned long long i = 0; i < bin.size(); i++) {
    if (bin[i]) {
      res = res + (1ULL << (bin.size() - 1 - i));
    }
  }
  return res;
}

}  // namespace tket
