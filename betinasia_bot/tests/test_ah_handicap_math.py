import pytest


def _mult_back_from_scores(*, line: str, side: str, hs: int, aws: int) -> float:
    """
    Implementação mínima (copiada da lógica usada nos relatórios) para testar a convenção:
    - Se `line` vier sem sinal (ex.: "2"), tratamos como magnitude do handicap do `side`.
    - Se `line` vier com sinal (ex.: "-0.5"), tratamos como handicap do HOME já assinado.
    Retorna o multiplicador do BACK: +1 win, 0 push, -1 loss, ±0.5 meia vitória/derrota.
    """
    goal_diff = int(hs) - int(aws)

    sel = (side or "").strip().lower()
    raw = str(line).strip().replace(",", ".").replace("−", "-")
    ah = float(raw)
    home_handicap = ah if (raw.startswith("+") or raw.startswith("-")) else (ah if sel == "home" else -ah)

    if sel == "home":
        adjusted = goal_diff + home_handicap
    elif sel == "away":
        adjusted = -goal_diff - home_handicap
    else:
        raise ValueError("side must be home/away")

    if adjusted > 0.25:
        return 1.0
    if adjusted == 0.25:
        return 0.5
    if adjusted == 0:
        return 0.0
    if adjusted == -0.25:
        return -0.5
    return -1.0


def test_unsigned_line_is_magnitude_of_side() -> None:
    # Away +2 (line="2", side="away") deve virar home_handicap=-2, portanto:
    # Se o away perde por 1 (hs=1, aws=0), com +2 o away "ganha" no handicap.
    assert _mult_back_from_scores(line="2", side="away", hs=1, aws=0) == 1.0

    # Home +2 (line="2", side="home"): se o home ganha por 1, continua win.
    assert _mult_back_from_scores(line="2", side="home", hs=1, aws=0) == 1.0


@pytest.mark.parametrize(
    "line,side,hs,aws,exp",
    [
        ("-0.5", "home", 1, 0, 1.0),  # home -0.5, vence por 1 => win
        ("-0.5", "home", 0, 0, -1.0),  # home -0.5, empata => loss
        # IMPORTANTE: quando `line` vem com sinal, ele é handicap do HOME.
        # Portanto, "away +0.5" equivale a "home -0.5" => line="-0.5" com side="away".
        ("-0.5", "away", 0, 0, 1.0),  # away +0.5, empata => win
        ("-0.5", "away", 1, 0, -1.0),  # away +0.5, perde por 1 => loss
        ("+0.5", "away", 0, 0, -1.0),  # away -0.5 (pois home +0.5), empata => loss
        ("0", "home", 0, 0, 0.0),  # pk, empate => push
    ],
)
def test_signed_line_is_home_handicap(line: str, side: str, hs: int, aws: int, exp: float) -> None:
    assert _mult_back_from_scores(line=line, side=side, hs=hs, aws=aws) == exp


def test_regression_example_old_bug_would_flip_away_unsigned_lines() -> None:
    # Cenário que o bug antigo errava: line="2" (sem sinal) + side="away".
    # Com a convenção correta, o away +2 deve ser WIN se perder por 1.
    mult = _mult_back_from_scores(line="2", side="away", hs=1, aws=0)
    assert mult == 1.0

