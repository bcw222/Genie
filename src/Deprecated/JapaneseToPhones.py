from Source.JapaneseG2P import JapaneseG2P
from Source.SymbolsV2 import symbols_v2, symbol_to_id_v2


def japanese_to_phones(text: str) -> list[int]:
    phones = JapaneseG2P.g2p(text)
    phones = ["SP" if x == "#" else x for x in phones]
    phones = ["UNK" if ph not in symbols_v2 else ph for ph in phones]
    phones = [symbol_to_id_v2[ph] for ph in phones]
    return phones
