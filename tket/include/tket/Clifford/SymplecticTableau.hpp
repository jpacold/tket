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

#pragma once

#include "tket/OpType/OpType.hpp"
#include "tket/Utils/MatrixAnalysis.hpp"
#include "tket/Utils/PauliTensor.hpp"

namespace tket {

/**
 * Boolean encoding of Pauli
 * <x, z> = <false, false> ==> I
 * <x, z> = <false, true>  ==> Z
 * <x, z> = <true,  false> ==> X
 * <x, z> = <true,  true>  ==> Y
 */
struct BoolPauli {
  bool x;
  bool z;

  /**
   * Lexicographic ordering by <x, z>
   */
  bool operator<(const BoolPauli &other) const;

  Pauli to_pauli() const;

  /**
   * Look-up table for Pauli multiplication with boolean encoding
   */
  static const std::map<
      std::pair<BoolPauli, BoolPauli>, std::pair<BoolPauli, Complex>>
      mult_lut;
};

class SymplecticTableau {
  /**
   * Class to represent a tableau of Paulis via the symplectic (binary)
   * decomposition. Specifically, each element in the tableau represents a Pauli
   * by a pair of binary values:
   * - (x, z) <==> X^x Z^z ignoring phase
   * - (0, 0) <==> I
   * - (0, 1) <==> Z
   * - (1, 0) <==> X
   * - (1, 1) <==> Y
   * Each row also maintains a phase value as a single boolean (-1)^p making the
   * assumption that rows are intended to capture stabilizers of some system and
   * thus have real coefficients. Pulling these together, each row represents a
   * (phaseful) Pauli string. Qubits are indexed by unsigneds in a linear array.
   *
   * This class provides the data structure, mechanisms for row multiplication,
   * gate application and validity checks.
   *
   * It is expected that all Pauli strings in a tableau should be linearly
   * independent, so a validity check is provided. A utility for checking the
   * mutual commutativity of terms is also provided.
   *
   * This serves as a base class for:
   * - UnitaryTableau where there is both a Z row and an X row for each input
   * describing the Pauli string formed when pushing a Z or X on the input
   * through to the outputs.
   * Future possible subclasses include:
   * - StabTableau where each row represents a stabilizer of a given state.
   * - IsometryTableau is a generalisation of StabTableau and UnitaryTableau
   * that can represent Clifford isometries by a Z and X row per input and a
   * spare row per free stabilizer.
   */
 public:
  /**
   * Constructor to initialise the tableau into a given state.
   * Includes checks for compatibility of sizes but will not force commutativity
   * or linear independence.
   */
  explicit SymplecticTableau(
      const MatrixXb &xmat, const MatrixXb &zmat, const VectorXb &phase);
  explicit SymplecticTableau(const PauliStabiliserVec &rows);

  /**
   * Other required constructors
   */
  SymplecticTableau(const SymplecticTableau &other) = default;
  SymplecticTableau(SymplecticTableau &&other) = default;
  SymplecticTableau &operator=(const SymplecticTableau &other) = default;
  SymplecticTableau &operator=(SymplecticTableau &&other) = default;

  /**
   * Get the number of rows in the tableau
   */
  unsigned get_n_rows() const;
  /**
   * Get the number of qubits in the tableau (number of binary columns is
   * 2*n_qubits + 1)
   */
  unsigned get_n_qubits() const;

  /**
   * Read off a row as a Pauli string
   */
  PauliStabiliser get_pauli(unsigned i) const;

  /**
   * Format in output stream as a binary tableau
   * Prints as "xmat zmat phase"
   */
  friend std::ostream &operator<<(
      std::ostream &os, const SymplecticTableau &tab);

  /**
   * Equality operator checks all contents
   */
  bool operator==(const SymplecticTableau &other) const;

  /**
   * Row multiplication
   * Multiplies rows ra and rw, stores the result in row rw
   */
  void row_mult(unsigned ra, unsigned rw, Complex coeff = 1.);

  /**
   * Applies a chosen gate to the given qubit(s)
   */
  void apply_S(unsigned qb);
  void apply_Z(unsigned qb);
  void apply_V(unsigned qb);
  void apply_X(unsigned qb);
  void apply_H(unsigned qb);
  void apply_CX(unsigned qc, unsigned qt);
  void apply_gate(OpType type, const std::vector<unsigned> &qbs);

  /**
   * Applies a pauli gadget (defined as e^{pauli * half_pis * -i * pi/2}).
   * If row commutes with pauli, no change is required.
   * If row anticommutes with pauli and half_pis % 4 == 1:
   * e^{pauli * -i * pi/2} row = (i * row * pauli) e^{pauli * -i * pi/2}
   */
  void apply_pauli_gadget(const PauliStabiliser &pauli, unsigned half_pis);

  /**
   * Generates relation of anti-commutativity between rows
   * Matrix element (i, j) is 0 if row i and row j commute, 1 if they
   * anti-commute
   */
  MatrixXb anticommuting_rows() const;

  /**
   * Obtains rank of tableau matrix for determining linear independence of rows
   */
  unsigned rank() const;

  /**
   * Applies row-wise gaussian elimination over the entire tableau, in qubit
   * order, subdivided by X first and then Z, reducing the combined boolean
   * matrix to reduced row echelon form.
   * Users of this class should only apply this in settings where row
   * multiplications are free actions, e.g. ok in ChoiMixTableau, not ok in
   * UnitaryTableau
   */
  void gaussian_form();

  /**
   * Tableau contents
   */
  MatrixXb xmat;
  MatrixXb zmat;
  VectorXb phase;

  /**
   * Complex conjugate of the state by conjugating rows
   */
  SymplecticTableau conjugate() const;

 private:
  /**
   * Helper methods for manipulating the tableau when applying gates
   */
  void row_mult(
      const MatrixXb::RowXpr &xa, const MatrixXb::RowXpr &za, const bool &pa,
      const MatrixXb::RowXpr &xb, const MatrixXb::RowXpr &zb, const bool &pb,
      Complex phase, MatrixXb::RowXpr &xw, MatrixXb::RowXpr &zw, bool &pw);
  void col_mult(
      const MatrixXb::ColXpr &a, const MatrixXb::ColXpr &b, bool flip,
      MatrixXb::ColXpr &w, VectorXb &pw);
};

JSON_DECL(SymplecticTableau)

std::ostream &operator<<(std::ostream &os, const SymplecticTableau &tab);

}  // namespace tket
