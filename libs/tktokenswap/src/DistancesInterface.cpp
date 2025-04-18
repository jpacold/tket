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

#include "DistancesInterface.hpp"

using std::vector;

namespace tket {

void DistancesInterface::register_shortest_path(
    const vector<std::size_t>& /*path*/) {}

void DistancesInterface::register_neighbours(
    std::size_t vertex, const vector<std::size_t>& neighbours) {
  for (std::size_t nv : neighbours) {
    register_edge(vertex, nv);
  }
}

void DistancesInterface::register_edge(
    std::size_t /*vertex1*/, std::size_t /*vertex2*/) {}

DistancesInterface::~DistancesInterface() {}

}  // namespace tket
