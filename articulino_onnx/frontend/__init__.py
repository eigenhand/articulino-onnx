"""German text frontend (numbers, dates, times, money, units, phone numbers, formulas, addresses, abbreviations).

>>> from articulino.frontend import normalize
>>> normalize("Der Termin ist am 15.11. um 15:30 Uhr.")
'Der Termin ist am fünfzehnten November um fünfzehn Uhr dreißig.'
"""
from .german import (StreamNormalizer, clean_markup, expand_abbreviations, expand_units, latex, math_symbols,
                     normalize, numbers, phone_numbers, spell_out_letters)

__all__ = ["normalize", "StreamNormalizer", "phone_numbers", "clean_markup", "expand_units", "math_symbols",
           "numbers", "expand_abbreviations", "spell_out_letters", "latex"]
