from build_dictionaries import extract_numbers


def main() -> None:
    assert extract_numbers("55") == ["55"]
    assert extract_numbers("1,5") == ["1.5"]
    assert extract_numbers("10x20x30 см") == ["10", "20", "30"]
    assert extract_numbers("8/256 ГБ") == ["8", "256"]
    assert extract_numbers("от -5 до +10 °C") == ["-5", "10"]
    assert extract_numbers("без чисел") == []
    print("PASSED")


if __name__ == "__main__":
    main()
