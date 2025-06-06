# Copyright Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The circuit module provides an API to interact with the
tket :py:class:`Circuit` data structure.
 This module is provided in binary form during the PyPI installation."""
from typing import (
    Any,
    Callable,
    Optional,
    Sequence,
    Union,
)

from pytket import wasm
from pytket._tket.circuit import *
from pytket._tket.circuit import Circuit
from pytket._tket.pauli import Pauli
from pytket._tket.unit_id import *

# prefixes for assertion bits
from pytket._tket.unit_id import (
    _DEBUG_ONE_REG_PREFIX,
    _DEBUG_ZERO_REG_PREFIX,
    Bit,
    BitRegister,
)

from .logic_exp import (
    BinaryOp,
    Ops,
    if_bit,
    if_not_bit,
    reg_eq,
    reg_geq,
    reg_gt,
    reg_leq,
    reg_lt,
    reg_neq,
)


def add_wasm(
    self: Circuit,
    funcname: str,
    filehandler: wasm.WasmModuleHandler,
    list_i: Sequence[int],
    list_o: Sequence[int],
    args: Union[Sequence[int], Sequence[Bit]],
    args_wasm: Optional[Sequence[int]] = None,
    **kwargs: Any,
) -> Circuit:
    """Add a classical function call from a wasm file to the circuit.
    \n\n:param funcname: name of the function that is called
    \n:param filehandler: wasm file or module handler to identify the wasm module
    \n:param list_i: list of the number of bits in the input variables
    \n:param list_o: list of the number of bits in the output variables
    \n:param args: vector of circuit bits the wasm op should be added to
    \n:param args_wasm: vector of wasmstates the wasm op should be added to
    \n:param kwargs: additional arguments passed to `add_gate_method` .
     Allowed parameters are `opgroup`,  `condition` , `condition_bits`,
     `condition_value`
    \n:return: the new :py:class:`Circuit`"""

    if args_wasm is None:
        args_wasm = [0]

    for x in list_i:
        if x > filehandler._int_size:
            raise ValueError(
                f"only functions with i{filehandler._int_size} type are allowed"
            )

    for x in list_o:
        if x > filehandler._int_size:
            raise ValueError(
                f"only functions with i{filehandler._int_size} type are allowed"
            )

    if filehandler.check_function(funcname, len(list_i), len(list_o)):
        if (len(args_wasm)) > 0:
            self._add_w_register(max(args_wasm) + 1)
        return self._add_wasm(
            funcname, str(filehandler), list_i, list_o, args, args_wasm, **kwargs
        )

    raise ValueError(f"{funcname} not found, check {repr(filehandler)}")


setattr(Circuit, "add_wasm", add_wasm)


def add_wasm_to_reg(
    self: Circuit,
    funcname: str,
    filehandler: wasm.WasmModuleHandler,
    list_i: Sequence[BitRegister],
    list_o: Sequence[BitRegister],
    args_wasm: Optional[Sequence[int]] = None,
    **kwargs: Any,
) -> Circuit:
    """Add a classical function call from a wasm file to the circuit.
    \n\n:param funcname: name of the function that is called
    \n:param filehandler: wasm file or module handler to identify the wasm module
    \n:param list_i: list of the classical registers assigned to
     the input variables of the function call
    \n:param list_o: list of the classical registers assigned to
     the output variables of the function call
    \n:param args_wasm: vector of wasmstates the wasm op should be added to
    \n:param kwargs: additional arguments passed to `add_gate_method` .
     Allowed parameters are `opgroup`,  `condition` , `condition_bits`,
     `condition_value`
    \n:return: the new :py:class:`Circuit`"""

    if args_wasm is None:
        args_wasm = [0]

    if filehandler.checked:
        for reg in list_i:
            if reg.size > 32:
                raise ValueError(
                    """wasm is only supporting 32 bit size registers,
please use only registers of at most 32 bits"""
                )

        for reg in list_o:
            if reg.size > 32:
                raise ValueError(
                    """wasm is only supporting 32 bit size registers,
please use only registers of at most 32 bits"""
                )

    # If the filehandler has not been checked we allow it to
    # be added without checking the function arity.
    if not filehandler.checked or filehandler.check_function(
        funcname, len(list_i), len(list_o)
    ):
        if (len(args_wasm)) > 0:
            self._add_w_register(max(args_wasm) + 1)
        return self._add_wasm(
            funcname, str(filehandler), list_i, list_o, args_wasm, **kwargs
        )

    raise ValueError(f"{funcname} not found, check {repr(filehandler)}")


setattr(Circuit, "add_wasm_to_reg", add_wasm_to_reg)
