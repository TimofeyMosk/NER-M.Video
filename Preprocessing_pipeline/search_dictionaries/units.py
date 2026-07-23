"""Canonical measurement units and aliases used in product search."""

from __future__ import annotations

import unicodedata


def unit(
    canonical: str,
    dimension: str,
    preferred: str,
    aliases: list[str],
    base_unit: str | None = None,
    factor_to_base: float | None = None,
) -> dict[str, object]:
    return {
        "canonical": canonical,
        "dimension": dimension,
        "preferred": preferred,
        "base_unit": base_unit or canonical,
        "factor_to_base": factor_to_base,
        "aliases": aliases,
    }


UNIT_DEFINITIONS = [
    # Length.
    unit("mm", "length", "мм", ["мм", "mm", "миллиметр", "миллиметра", "миллиметры", "миллиметров"], "m", 0.001),
    unit("cm", "length", "см", ["см", "cm", "сантиметр", "сантиметра", "сантиметры", "сантиметров"], "m", 0.01),
    unit("m", "length", "м", ["м", "m", "метр", "метра", "метры", "метров"], "m", 1.0),
    unit("km", "length", "км", ["км", "km", "километр", "километра", "километры", "километров"], "m", 1000.0),
    unit("um", "length", "мкм", ["мкм", "µm", "um", "микрометр", "микрона", "микрон"], "m", 0.000001),
    unit("nm", "length", "нм", ["нм", "nm", "нанометр", "нанометра", "нанометров"], "m", 0.000000001),
    unit("inch", "length", "дюйм", ["\"", "″", "дюйм", "дюйма", "дюймы", "дюймов", "inch", "inches", "in"], "m", 0.0254),
    unit("foot", "length", "фут", ["'", "′", "фут", "фута", "футов", "ft"], "m", 0.3048),

    # Mass.
    unit("mg", "mass", "мг", ["мг", "mg", "миллиграмм", "миллиграмма", "миллиграммов"], "kg", 0.000001),
    unit("g", "mass", "г", ["г", "гр", "g", "грамм", "грамма", "граммы", "граммов"], "kg", 0.001),
    unit("kg", "mass", "кг", ["кг", "kg", "килограмм", "килограмма", "килограммы", "килограммов", "кило"], "kg", 1.0),
    unit("t", "mass", "т", ["т", "t", "тонна", "тонны", "тонн"], "kg", 1000.0),

    # Area and volume.
    unit("mm2", "area", "мм²", ["мм2", "мм²", "mm2", "mm²", "кв мм", "квадратный миллиметр"], "m2", 0.000001),
    unit("cm2", "area", "см²", ["см2", "см²", "cm2", "cm²", "кв см", "квадратный сантиметр"], "m2", 0.0001),
    unit("m2", "area", "м²", ["м2", "м²", "m2", "m²", "кв м", "квадратный метр", "квадратных метров"], "m2", 1.0),
    unit("mm3", "volume", "мм³", ["мм3", "мм³", "mm3", "mm³"], "l", 0.000001),
    unit("cm3", "volume", "см³", ["см3", "см³", "cm3", "cm³", "куб см"], "l", 0.001),
    unit("m3", "volume", "м³", ["м3", "м³", "m3", "m³", "куб м"], "l", 1000.0),
    unit("ml", "volume", "мл", ["мл", "ml", "миллилитр", "миллилитра", "миллилитров"], "l", 0.001),
    unit("l", "volume", "л", ["л", "l", "литр", "литра", "литры", "литров"], "l", 1.0),

    # Power and energy.
    unit("mw", "power", "мВт", ["мвт", "mw", "милливатт", "милливатта", "милливаттов"], "w", 0.001),
    unit("w", "power", "Вт", ["вт", "w", "ватт", "ватта", "ваттов"], "w", 1.0),
    unit("kw", "power", "кВт", ["квт", "kw", "киловатт", "киловатта", "киловаттов"], "w", 1000.0),
    unit("hp", "power", "л.с.", ["лс", "л.с", "л.с.", "hp", "лошадиная сила", "лошадиных сил"], "w", 735.49875),
    unit("wh", "energy", "Вт·ч", ["втч", "вт ч", "wh", "w h", "ватт час", "ватт часов"], "wh", 1.0),
    unit("kwh", "energy", "кВт·ч", ["квтч", "квт ч", "kwh", "киловатт час", "киловатт часов"], "wh", 1000.0),
    unit("j", "energy", "Дж", ["дж", "j", "джоуль", "джоуля", "джоулей"], "j", 1.0),
    unit("kj", "energy", "кДж", ["кдж", "kj", "килоджоуль", "килоджоулей"], "j", 1000.0),
    unit("kcal", "energy", "ккал", ["ккал", "kcal", "килокалория", "килокалорий"], "j", 4184.0),
    unit("btu", "energy", "BTU", ["btu", "бте"], "j", 1055.05585262),

    # Electricity.
    unit("mv", "voltage", "мВ", ["мв", "mv", "милливольт", "милливольта"], "v", 0.001),
    unit("v", "voltage", "В", ["в", "v", "вольт", "вольта", "вольтов"], "v", 1.0),
    unit("kv", "voltage", "кВ", ["кв", "kv", "киловольт", "киловольта"], "v", 1000.0),
    unit("ma", "current", "мА", ["ма", "ma", "миллиампер", "миллиампера"], "a", 0.001),
    unit("a", "current", "А", ["а", "a", "ампер", "ампера", "амперов"], "a", 1.0),
    unit("mah", "electric_charge", "мА·ч", ["мач", "ма ч", "mah", "миллиампер час", "миллиампер часов"], "ah", 0.001),
    unit("ah", "electric_charge", "А·ч", ["ач", "а ч", "ah", "ампер час", "ампер часов"], "ah", 1.0),
    unit("ohm", "resistance", "Ом", ["ом", "ohm", "ω"], "ohm", 1.0),
    unit("kohm", "resistance", "кОм", ["ком", "kohm", "kω"], "ohm", 1000.0),
    unit("mohm", "resistance", "МОм", ["мом", "mohm", "mω"], "ohm", 1000000.0),
    unit("va", "apparent_power", "ВА", ["ва", "va", "вольт ампер"], "va", 1.0),

    # Frequency and time.
    unit("hz", "frequency", "Гц", ["гц", "hz", "герц"], "hz", 1.0),
    unit("khz", "frequency", "кГц", ["кгц", "khz", "килогерц"], "hz", 1000.0),
    unit("mhz", "frequency", "МГц", ["мгц", "mhz", "мегагерц"], "hz", 1000000.0),
    unit("ghz", "frequency", "ГГц", ["ггц", "ghz", "гигагерц"], "hz", 1000000000.0),
    unit("ms", "time", "мс", ["мс", "ms", "миллисекунда", "миллисекунды"], "s", 0.001),
    unit("s", "time", "с", ["с", "сек", "сек.", "s", "sec", "секунда", "секунды", "секунд"], "s", 1.0),
    unit("min", "time", "мин", ["мин", "мин.", "min", "минута", "минуты", "минут"], "s", 60.0),
    unit("h", "time", "ч", ["ч", "час", "часа", "часов", "h", "hr"], "s", 3600.0),
    unit("day", "time", "сутки", ["сутки", "суток", "день", "дня", "дней"], "s", 86400.0),
    unit("month", "time", "мес.", ["мес", "мес.", "месяц", "месяца", "месяцев"], "month", 1.0),
    unit("year", "time", "год", ["год", "года", "лет"], "year", 1.0),

    # Rate, speed, pressure and flow.
    unit("rpm", "rotation_rate", "об/мин", ["об/мин", "об мин", "rpm", "оборот в минуту", "оборотов в минуту"], "rpm", 1.0),
    unit("mps", "speed", "м/с", ["м/с", "м/сек", "m/s", "метр в секунду", "метров в секунду"], "mps", 1.0),
    unit("kmh", "speed", "км/ч", ["км/ч", "km/h", "километр в час", "километров в час"], "mps", 0.2777777778),
    unit("pa", "pressure", "Па", ["па", "pa", "паскаль", "паскалей"], "pa", 1.0),
    unit("kpa", "pressure", "кПа", ["кпа", "kpa", "килопаскаль", "килопаскалей"], "pa", 1000.0),
    unit("mpa", "pressure", "МПа", ["мпа", "mpa", "мегапаскаль", "мегапаскалей"], "pa", 1000000.0),
    unit("bar", "pressure", "бар", ["бар", "bar", "бара", "баров"], "pa", 100000.0),
    unit("atm", "pressure", "атм", ["атм", "atm", "атмосфера", "атмосферы"], "pa", 101325.0),
    unit("mmhg", "pressure", "мм рт. ст.", ["мм рт ст", "мм рт. ст.", "mmhg"], "pa", 133.322387415),
    unit("lpm", "flow", "л/мин", ["л/мин", "l/min", "литр в минуту", "литров в минуту"], "lpm", 1.0),
    unit("lph", "flow", "л/ч", ["л/ч", "l/h", "литр в час", "литров в час"], "lpm", 1 / 60),
    unit("m3h", "flow", "м³/ч", ["м3/ч", "м³/ч", "м3/час", "m3/h", "m³/h"], "lpm", 1000 / 60),
    unit("kgday", "production_rate", "кг/сутки", ["кг/сутки", "кг в сутки", "kg/day"], "kgday", 1.0),
    unit("tday", "production_rate", "т/сутки", ["т/д", "т/сутки", "t/day"], "kgday", 1000.0),
    unit("gmin", "mass_flow", "г/мин", ["г/мин", "g/min"], "kg_s", 1 / 60000),
    unit("kgmin", "mass_flow", "кг/мин", ["кг/мин", "kg/min"], "kg_s", 1 / 60),
    unit("gh", "mass_flow", "г/ч", ["г/ч", "g/h"], "kg_s", 1 / 3600000),
    unit("cmps", "speed", "см/с", ["см/сек", "см/с", "cm/s"], "mps", 0.01),
    unit("mpm", "speed", "м/мин", ["м/мин", "m/min"], "mps", 1 / 60),
    unit("lps", "flow", "л/с", ["л/сек", "л/с", "l/s"], "lpm", 60.0),
    unit("mlday", "flow", "мл/сутки", ["мл/сутки", "ml/day"], "lpm", 0.001 / 1440),

    # Data and media.
    unit("bit", "data_size", "бит", ["бит", "bit", "бита", "битов"], "byte", 0.125),
    unit("byte", "data_size", "Б", ["байт", "byte", "байта", "байтов"], "byte", 1.0),
    unit("kb", "data_size", "КБ", ["кб", "kb", "килобайт", "килобайта", "килобайтов"], "byte", 1000.0),
    unit("mb", "data_size", "МБ", ["мб", "mb", "мегабайт", "мегабайта", "мегабайтов"], "byte", 1000000.0),
    unit("gb", "data_size", "ГБ", ["гб", "gb", "гигабайт", "гигабайта", "гигабайтов"], "byte", 1000000000.0),
    unit("tb", "data_size", "ТБ", ["тб", "tb", "терабайт", "терабайта", "терабайтов"], "byte", 1000000000000.0),
    unit("mbps", "data_rate", "Мбит/с", ["мбит/с", "мбит/сек", "mbps", "mb/s", "мб/сек"], "bps", 1000000.0),
    unit("gbps", "data_rate", "Гбит/с", ["гбит/с", "гбит/сек", "gbps"], "bps", 1000000000.0),
    unit("pixel", "resolution", "пикс.", ["пикс", "пикс.", "pixel", "px", "пиксель", "пикселей"], "pixel", 1.0),
    unit("mpixel", "resolution", "МПикс", ["мпикс", "мп", "mp", "mpix", "мегапиксель", "мегапикселей"], "pixel", 1000000.0),
    unit("ppi", "pixel_density", "ppi", ["ppi", "пикс/дюйм"], "ppi", 1.0),
    unit("dpi", "pixel_density", "dpi", ["dpi", "точек на дюйм"], "dpi", 1.0),
    unit("fps", "frame_rate", "кадр/с", ["кадр/сек", "кадров/сек", "fps"], "fps", 1.0),

    # Other common product characteristics.
    unit("percent", "ratio", "%", ["%", "процент", "процента", "процентов"], "percent", 1.0),
    unit("db", "sound_level", "дБ", ["дб", "db", "децибел", "децибела"], "db", 1.0),
    unit("celsius", "temperature", "°C", ["°c", "°с", "*c", "*с", "градус c", "градус с", "градуса", "градусов", "цельсий"], "celsius", None),
    unit("kelvin", "temperature", "K", ["k", "к", "кельвин", "кельвина"], "kelvin", 1.0),
    unit("lumen", "luminous_flux", "лм", ["лм", "lm", "lumen", "люмен", "люмена", "люменов"], "lumen", 1.0),
    unit("candela_m2", "luminance", "кд/м²", ["кд/м2", "кд/м²", "cd/m2", "cd/m²"], "candela_m2", 1.0),
    unit("candela", "luminous_intensity", "кд", ["кд", "cd", "кандела"], "candela", 1.0),
    unit("lumen_w", "luminous_efficacy", "лм/Вт", ["лм/вт", "lm/w"], "lumen_w", 1.0),
    unit("gram_m2", "surface_density", "г/м²", ["г/м2", "г/м²", "g/m2", "g/m²"], "gram_m2", 1.0),
    unit("kg_m3", "density", "кг/м³", ["кг/м3", "кг/м³", "kg/m3", "kg/m³"], "kg_m3", 1.0),
    unit("nm_torque", "torque", "Н·м", ["нм момент", "н.м", "n.m", "nm torque"], "nm_torque", 1.0),
    unit("kn", "force", "кН", ["кн", "kn", "килоньютон"], "n", 1000.0),
    unit("mmh2o", "pressure", "мм вод. ст.", ["мм в.ст", "мм в.ст.", "mmh2o"], "pa", 9.80665),
    unit("w_channel", "power_per_channel", "Вт/канал", ["вт/канал", "w/channel"], "w_channel", 1.0),
    unit("w_m", "linear_power", "Вт/м", ["вт/м", "w/m"], "w_m", 1.0),
    unit("ppm", "ratio", "ppm", ["ppm", "частей на миллион"], "ppm", 1.0),
    unit("db_oct", "frequency_slope", "дБ/окт", ["дб/окт", "db/oct"], "db_oct", 1.0),
    unit("page", "count", "стр.", ["стр", "стр.", "страница", "страницы", "страниц"], "page", 1.0),
    unit("page_min", "rate", "стр/мин", ["стр/мин", "страниц в минуту", "ppm pages"], "page_min", 1.0),
    unit("sheet", "count", "лист", ["лист", "лист.", "листов", "листы"], "sheet", 1.0),
    unit("stitch_min", "rate", "стеж./мин", ["стеж/мин", "стеж./мин", "стежков в минуту"], "stitch_min", 1.0),
    unit("piece", "count", "шт.", ["шт", "шт.", "штука", "штуки", "штук", "pcs"], "piece", 1.0),
]


def normalize_unit_alias(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    value = value.replace("²", "2").replace("³", "3").replace("″", '"').replace("′", "'")
    value = value.replace("·", ".").replace(" ", "")
    while value.endswith("."):
        value = value[:-1]
    return value


def build_unit_lookup() -> tuple[dict[str, str], dict[str, list[str]]]:
    lookup: dict[str, str] = {}
    collisions: dict[str, list[str]] = {}
    for definition in UNIT_DEFINITIONS:
        canonical = str(definition["canonical"])
        for alias in [canonical, str(definition["preferred"]), *definition["aliases"]]:
            normalized = normalize_unit_alias(alias)
            previous = lookup.get(normalized)
            if previous and previous != canonical:
                collisions.setdefault(normalized, sorted({previous, canonical}))
                continue
            lookup[normalized] = canonical
    return lookup, collisions
