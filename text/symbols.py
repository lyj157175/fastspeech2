""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

from text import cmudict, pinyin

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil"]

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ["@" + s for s in cmudict.valid_symbols]
_pinyin = ["@" + s for s in pinyin.valid_symbols]

_diff_pinyin = ["@" + s for s in pinyin.diff_py]



# Export all symbols:
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _arpabet
    + _pinyin
    + _silences
)

my_symbols2 = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _arpabet
    + _pinyin
    +_diff_pinyin
    + _silences
)

my_symbols = ['_', '-', '!', "'", '(', ')', ',', '.', ':', ';', '?', ' ','k', 'a2', 'er2', 'p', 'u3', 'ei2', 'uai4', 's', 'uen1', 'uan2', 'h', 'ua2', 't', 'i1', 'j', 'ia2', 'v3', 'c', 'ian2', 'b', 'ie2', 'z', 'ai4', 'iong1', 'ao4', 'uo3', 'ao2', 'm', 'a3', 'ei4', 'g', 'ua4', 'o3', 'l', 'uo2', 'an1', 'd', 'iao1', 'ch', 'an2', 'van4', 'zh', 'en3', 'ong3', 'ueng1', 'a4', 'eng4', 'x', 'iao3', 'ing2', 'q', 'ie4', 'er3', 'uei4', 'u4', 'iou4', 'ai3', 'v2', 'van3', 'sh', 'ua3', 'en1', 'ang2', 've1', 'u2', 'iii3', 'er4', 'uen4', 'f', 'uo4', 'i3', 'ang4', 'i2', 'vn2', 'eng2', 'v4', 'uei3', 'u1', 'in3', 'uei1', 'iou3', 'an3', 'van2', 've4', 'uei2', 'v1', 'ong2', 'e2', 'uo1', 'e1', 'e4', 'n', 'uan1', 'uen2', 'ai2', 'iao4', 'ai1', 'iong4', 'ie3', 'ie1', 'r', 'ang3', 'ian4', 'van1', 'eng1', 'eng5', 'ai5', 'ii3', 'uang2', 'iang1', 'iou2', 'uang4', 'uan3', 'iao2', 'iong2', 'ian3', 'en4', 'ii1', 'ou1', 'ing4', 'ei3', 'en2', 'iou5', 'ing1', 'ang1', 'i4', 'ia4', 'ii5', 'iou1', 'ua1', 'uan4', 'iii4', 'in4', 'a1', 'uanr1', 'ei1', 'uang3', 'ia1', 'ou4', 'iii2', 'vn1', 'u5', 'iang2', 'e5', 'in1', 'in2', 'iii1', 'ao3', 'e3', 'ir4', 'uang1', 'vn4', 'ao1', 'an4', 'ian1', 'o2', 'a5', 'vn3', 'enr2', 'ong1', 'iang3', 'ii2', 'uen3', 'eng3', 'ou3', 'ing3', 'iong3', 'ao5', 'uo5', 'ia3', 'ou2', 've2', 'ueng4', 'ii4', 'uai2', 'ong4', 've3', 'an5', 'aor4', 'o4', 'vanr4', 'ua5', 'iang4', 'ang5', 'van5', 'iii5', 'o1', 'ong5', 'ia5', 'uen5', 'uai3', 'ou5', 'i5', 'ing5', 'uan5', 'ir1', 'o5', 'iang5', 'en5', 'uang5', 'iao5', 'ie5', 'ueir4', 'air2', 'uanr2', 'uai1', 'enr4', 'ir2', 'ar4', 'inr4', 'ei5', 'uenr4', 'er5', 'ingr3', 'uor3', 'iong5', 'enr5', 'anr4', 've5', 'ir3', 'v5', 'uei5', 'our2', 'in5', 'iiir4', 'ian5', 'ianr3', 'ur4', 'ar3', 'ar2', 'uenr3', 'iar3', 'uair4', 'uai5', 'enr1', 'ingr2', 'ueir1', 'vn5', 'anr3', 'aor3', 'pl', 'iyl4', 'ianr2', 'ueng2', 'ueng3', 'ir5', 'anr1', 'ianr1', 'ueir3', 'io5', 'ng1', 'ur3', 'ongr4', 'air4', 'uor2', 'iangr4', 'iar1', 'iour1', 'sp', 'sil' ]

# print(symbols)
# print(symbols.index('@AA'), symbols.index('@ZH'))
# print(len(_arpabet))

if __name__=='__main__':

    print(my_symbols)

    diff_pin = []

    print(len(symbols))
    print(len(my_symbols))






