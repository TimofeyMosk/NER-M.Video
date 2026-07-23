"""Canonical 30-color palette and deterministic aliases.

The palette reflects the most frequent product-facing color classes in the
catalog.  It deliberately keeps metallic finishes that matter for electronics
and appliances.  No regular expressions are used.
"""

from __future__ import annotations

import unicodedata


COLOR_NORMALIZATION_VERSION = "catalog_palette_30_v1"

# Stable machine id, Russian model token, English display label.
PALETTE_30: tuple[tuple[str, str, str], ...] = (
    ("black", "черный", "black"),
    ("white", "белый", "white"),
    ("multicolor", "разноцветный", "multicolor"),
    ("transparent", "прозрачный", "transparent"),
    ("gray", "серый", "gray"),
    ("silver", "серебристый", "silver"),
    ("chrome", "хром", "chrome"),
    ("beige", "бежевый", "beige"),
    ("blue", "синий", "blue"),
    ("green", "зеленый", "green"),
    ("red", "красный", "red"),
    ("brown", "коричневый", "brown"),
    ("stainless_steel", "нержавеющая сталь", "stainless steel"),
    ("light_blue", "голубой", "light blue"),
    ("pink", "розовый", "pink"),
    ("orange", "оранжевый", "orange"),
    ("purple", "фиолетовый", "purple"),
    ("graphite", "графитовый", "graphite"),
    ("gold", "золотистый", "gold"),
    ("yellow", "желтый", "yellow"),
    ("amber", "янтарный", "amber"),
    ("bronze", "бронзовый", "bronze"),
    ("turquoise", "бирюзовый", "turquoise"),
    ("cream", "кремовый", "cream"),
    ("mint", "мятный", "mint"),
    ("sand", "песочный", "sand"),
    ("chocolate", "шоколадный", "chocolate"),
    ("lilac", "сиреневый", "lilac"),
    ("burgundy", "бордовый", "burgundy"),
    ("copper", "медный", "copper"),
)

PALETTE_BY_ID = {item[0]: item for item in PALETTE_30}
PALETTE_ID_BY_LABEL = {item[1]: item[0] for item in PALETTE_30}

# Seeds are expanded to Russian inflections by the dictionary builder.
COLOR_SEEDS: dict[str, tuple[str, ...]] = {
    "black": ("черный", "угольный", "оникс", "black", "noir"),
    "white": ("белый", "белоснежный", "white"),
    "multicolor": ("разноцветный", "многоцветный", "радужный", "мультиколор", "multicolor", "multi"),
    "transparent": ("прозрачный", "бесцветный", "transparent", "clear"),
    "gray": ("серый", "антрацит", "антрацитовый", "дымчатый", "пепельный", "маренго", "gray", "grey", "anthracite"),
    "silver": ("серебро", "серебристый", "серебряный", "стальной", "никель", "никелевый", "титан", "титановый", "алюминиевый", "silver", "platinum"),
    "chrome": ("хром", "хромированный", "chrome"),
    "beige": ("бежевый", "кашемировый", "капучино", "латте", "карамельный", "beige", "cashmere"),
    "blue": ("синий", "индиго", "ультрамарин", "сапфировый", "кобальтовый", "navy", "blue", "indigo"),
    "green": ("зеленый", "салатовый", "оливковый", "изумрудный", "хаки", "лайм", "шалфей", "green", "olive", "khaki", "lime", "sage"),
    "red": ("красный", "алый", "рубиновый", "малиновый", "red", "scarlet", "ruby"),
    "brown": ("коричневый", "кофейный", "ореховый", "каштановый", "венге", "махагон", "brown", "walnut", "mocha", "mokka", "coffee"),
    "stainless_steel": ("нержавеющий", "inox", "stainless"),
    "light_blue": ("голубой", "небесный", "лазурный", "аквамариновый", "cyan", "skyblue", "aqua"),
    "pink": ("розовый", "пудровый", "фуксия", "pink", "rose"),
    "orange": ("оранжевый", "абрикосовый", "персиковый", "коралловый", "терракотовый", "orange", "apricot", "peach", "coral", "terracotta"),
    "purple": ("фиолетовый", "лиловый", "лавандовый", "аметистовый", "пурпурный", "purple", "violet", "lavender"),
    "graphite": ("графит", "графитовый", "graphite"),
    "gold": ("золото", "золотой", "золотистый", "латунь", "шампань", "gold", "golden", "champagne"),
    "yellow": ("желтый", "лимонный", "горчичный", "охра", "yellow", "lemon", "mustard"),
    "amber": ("янтарный", "медовый", "amber"),
    "bronze": ("бронза", "бронзовый", "bronze"),
    "turquoise": ("бирюзовый", "teal", "turquoise"),
    "cream": ("кремовый", "молочный", "ванильный", "жемчужный", "ivory", "cream", "pearl"),
    "mint": ("мятный", "mint"),
    "sand": ("песочный", "песчаный", "sand", "sandy"),
    "chocolate": ("шоколадный", "какао", "chocolate", "cocoa"),
    "lilac": ("сиреневый", "lilac"),
    "burgundy": ("бордовый", "винный", "вишневый", "бургунди", "burgundy", "maroon"),
    "copper": ("медный", "медь", "copper"),
}

# Phrases whose meaning cannot safely be inferred token-by-token.
PHRASE_OVERRIDES: dict[str, str] = {
    "нержавеющая сталь": "stainless_steel",
    "нержавеющей стали": "stainless_steel",
    "нерж сталь": "stainless_steel",
    "нержсталь": "stainless_steel",
    "stainless steel": "stainless_steel",
    "stainless steel color": "stainless_steel",
    "розовое золото": "gold",
    "rose gold": "gold",
    "слоновая кость": "cream",
    "ivory white": "cream",
    "дуб сонома": "brown",
    "мокрый асфальт": "gray",
    "оружейная сталь": "gray",
    "gunmetal": "gray",
    "space gray": "gray",
    "space grey": "gray",
    "серый космос": "gray",
    "космический серый": "gray",
    "midnight black": "black",
    "space black": "black",
    "черный космос": "black",
    "темная ночь": "black",
    "темно синяя полночь": "blue",
}


def normalize_color_text(text: str) -> str:
    """Normalize an alias using the project contract, without regex."""
    value = unicodedata.normalize("NFKC", str(text)).replace("ё", "е").replace("Ё", "Е").casefold()
    output: list[str] = []
    pending_space = False
    for char in value:
        if char.isalnum():
            if pending_space and output:
                output.append(" ")
            output.append(char)
            pending_space = False
        else:
            pending_space = bool(output)
    return "".join(output).strip()


def split_color_tokens(text: str) -> tuple[str, ...]:
    normalized = normalize_color_text(text)
    return tuple(normalized.split()) if normalized else ()
