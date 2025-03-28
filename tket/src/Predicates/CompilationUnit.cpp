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

#include "tket/Predicates/CompilationUnit.hpp"

#include <memory>

#include "tket/Utils/UnitID.hpp"
namespace tket {

CompilationUnit::CompilationUnit(const Circuit& circ) : circ_(circ) {
  initialize_maps();
}

CompilationUnit::CompilationUnit(
    const Circuit& circ, const PredicatePtrMap& preds)
    : circ_(circ), target_preds(preds) {
  initialize_maps();
  initialize_cache();
}

CompilationUnit::CompilationUnit(
    const Circuit& circ, const std::vector<PredicatePtr>& preds)
    : circ_(circ) {
  for (const PredicatePtr& pp : preds) target_preds.insert(make_type_pair(pp));
  initialize_maps();
  initialize_cache();
}

TypePredicatePair CompilationUnit::make_type_pair(const PredicatePtr& ptr) {
  const Predicate& p = *ptr;
  std::type_index ti = typeid(p);
  return {ti, ptr};
}

bool CompilationUnit::calc_predicate(const Predicate& pred) const {
  return pred.verify(circ_);
}

bool CompilationUnit::check_all_predicates() const {
  for (const TypePredicatePair& ref_pred : target_preds) {
    if (!calc_predicate(*ref_pred.second)) return false;
  }
  return true;
}

std::string CompilationUnit::to_string() const {
  std::stringstream s;
  s << "~~~CompilationUnit~~~" << std::endl
    << "<tket::Circuit qubits=" << circ_.n_qubits()
    << ", gates=" << circ_.n_gates() << ">" << std::endl;

  if (!target_preds.empty()) {
    s << "Target Predicates:" << std::endl;
    for (const TypePredicatePair& pp : target_preds) {
      s << "  " << pp.second->to_string() << std::endl;
    }
  } else
    s << "Target Predicates empty" << std::endl;
  if (!cache_.empty()) {
    s << "Cache:" << std::endl;
    for (const std::pair<const std::type_index, std::pair<PredicatePtr, bool>>&
             tp : cache_) {
      s << " " << tp.second.first->to_string()
        << " :: " << ((tp.second.second) ? "True" : "False") << std::endl;
    }
  } else
    s << "Cache empty" << std::endl;
  return s.str();
}

void CompilationUnit::empty_cache() const { cache_ = {}; }

void CompilationUnit::initialize_cache() const {
  if (!cache_.empty())
    throw std::logic_error("PredicateCache must be empty to be initialized");
  for (const TypePredicatePair& pr : target_preds) {
    const Predicate& p = *(pr.second);
    std::type_index ti = typeid(p);
    if (cache_.find(ti) != cache_.end())
      throw std::logic_error("Duplicate verify type in Predicate list");
    bool to_cache = calc_predicate(p);
    cache_.insert({ti, {pr.second, to_cache}});
  }
}

void CompilationUnit::initialize_maps() {
  if (maps) throw std::logic_error("Maps already initialized");
  maps = std::make_shared<unit_bimaps_t>();
  for (const UnitID& u : circ_.all_units()) {
    maps->initial.insert({u, u});
    maps->final.insert({u, u});
  }
}

}  // namespace tket
