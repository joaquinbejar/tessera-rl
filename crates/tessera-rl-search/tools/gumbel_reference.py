#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Reference table for the Gumbel sampler tests (noise.rs).

For each 52-bit k, u = (2k + 1) / 2^53 exactly (a Fraction), and
g = -ln(-ln u) evaluated with the standard-library decimal module at 80
significant digits, printed to 60. The nearest f64 of g is printed in
hexadecimal (float.hex). Only the standard library is used.

Usage: python3 crates/tessera-rl-search/tools/gumbel_reference.py
"""

from decimal import Decimal, getcontext
from fractions import Fraction

getcontext().prec = 80

KS = [0, 1, 2**51, 2**52 - 1]


def gumbel(k: int) -> Decimal:
    u = Fraction(2 * k + 1, 2**53)
    u_dec = Decimal(u.numerator) / Decimal(u.denominator)
    return -((-(u_dec.ln())).ln())


def main() -> None:
    for k in KS:
        g = gumbel(k)
        nearest = float(Fraction(g))
        print(f"k={k} g={g:.60e} f64={nearest.hex()}")


if __name__ == "__main__":
    main()
