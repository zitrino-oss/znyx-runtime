"""
Checksum algorithms used by regional identifier validators.

Keeping these in one place so PII patterns stay declarative and the validation
math is testable on its own.
"""
from __future__ import annotations


# ---- Verhoeff (used by Indian Aadhaar) -------------------------------------

_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)

_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def verhoeff_check(number: str) -> bool:
    """
    Validate a digit string with the Verhoeff algorithm.

    Used by: Indian Aadhaar.
    """
    if not number or not number.isdigit():
        return False
    c = 0
    for i, digit in enumerate(reversed(number)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(digit)]]
    return c == 0


# ---- Mod-10 (Luhn-style with a single weighting) ---------------------------

def mod10_sin_check(number: str) -> bool:
    """
    Validate a 9-digit Canadian SIN using mod-10 / Luhn.
    """
    digits = number.replace(' ', '').replace('-', '')
    if len(digits) != 9 or not digits.isdigit():
        return False
    total = 0
    for i, ch in enumerate(digits):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


# ---- Mod-11 (NHS) ----------------------------------------------------------

def mod11_nhs_check(number: str) -> bool:
    """
    Validate a UK NHS number (10 digits, mod-11 checksum).

    The final digit is the check digit. The first 9 digits are multiplied by
    weights 10..2. The sum mod 11 is compared against 11 - (digit); check=10 is
    invalid, check=11 maps to 0.
    """
    digits = number.replace(' ', '').replace('-', '')
    if len(digits) != 10 or not digits.isdigit():
        return False
    total = sum(int(d) * w for d, w in zip(digits[:9], range(10, 1, -1)))
    check = 11 - (total % 11)
    if check == 11:
        check = 0
    if check == 10:
        return False
    return check == int(digits[9])


# ---- BR CPF / CNPJ ---------------------------------------------------------

def cpf_check(number: str) -> bool:
    """Validate a Brazilian CPF (11 digits, 2 check digits)."""
    digits = ''.join(c for c in number if c.isdigit())
    if len(digits) != 11 or len(set(digits)) == 1:
        return False

    def _dv(slice_: str, start_weight: int) -> int:
        total = sum(int(d) * (start_weight - i) for i, d in enumerate(slice_))
        mod = total % 11
        return 0 if mod < 2 else 11 - mod

    d1 = _dv(digits[:9], 10)
    d2 = _dv(digits[:10], 11)
    return d1 == int(digits[9]) and d2 == int(digits[10])


def cnpj_check(number: str) -> bool:
    """Validate a Brazilian CNPJ (14 digits, 2 check digits)."""
    digits = ''.join(c for c in number if c.isdigit())
    if len(digits) != 14 or len(set(digits)) == 1:
        return False

    weights1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    weights2 = [6] + weights1

    def _dv(slice_: str, weights) -> int:
        total = sum(int(d) * w for d, w in zip(slice_, weights))
        mod = total % 11
        return 0 if mod < 2 else 11 - mod

    d1 = _dv(digits[:12], weights1)
    d2 = _dv(digits[:13], weights2)
    return d1 == int(digits[12]) and d2 == int(digits[13])


# ---- ES NIF/NIE ------------------------------------------------------------

_NIF_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"


def nif_check(number: str) -> bool:
    """Validate a Spanish NIF/NIE (8 digits + letter, or prefix X/Y/Z + 7 digits + letter)."""
    s = number.upper().replace('-', '').replace(' ', '')
    if len(s) != 9:
        return False
    prefix_map = {'X': '0', 'Y': '1', 'Z': '2'}
    body = s[:-1]
    letter = s[-1]
    if body[0] in prefix_map:  # NIE
        body = prefix_map[body[0]] + body[1:]
    if not body.isdigit() or not letter.isalpha():
        return False
    return _NIF_LETTERS[int(body) % 23] == letter
