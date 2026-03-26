import re


def categorize_dgs_gloss(
    gloss: str,
    variation_pattern: str = r"^(.*?)((\d+[a-z]*)?[\*\^]*)$",
    pointing_pattern: str = r"^\$index.*",
    depictive_pattern: str = r"^\$prod.*",
    buoys_pattern: str = r"^\$list.*",
    palm_up_pattern: str = r"^\$gest-off.*",
    gesture_pattern: str = r"^\$gest.*",
    fingerspelling_pattern: str = r"^\$alpha.*",
    number_pattern: str = r"^\$num.*",
    entity_pattern: str = r"^\$(?:name|org)(?:-(.+))?$",
    morpheme_pattern: str = r"^\$morph-(.+)$",
    foreign_pattern: str = r"^(.*)-(?:asl|bsl|ints|lis|lsm|nzsl|pjm)$",
    init_pattern: str = r"^\$init.*",
    cued_speech_pattern: str = r"^\$cued-speech.*",
    unclear_pattern: str = r"^(\$unclear|\$\$extra-ling-act).*",
) -> tuple[str | None, str, str | None, str | None]:
    """
    Parses a DGS Corpus sign language gloss into its linguistic components.
    """

    # 1. Extract the specifier
    # Colons exclusively introduce specifiers like fingerspelled letters or digits
    if ":" in gloss:
        left, specifier = gloss.split(":", maxsplit=1)
    else:
        left = gloss
        specifier = None

    # 2. Extract variation from the left side
    # Variations in DGS are appended to the base (e.g., 1, 2A, *, ^)
    if variation_pattern and (var_match := re.fullmatch(variation_pattern, left)):
        parsed_base = var_match.group(1)
        # Avoid assigning an empty string; use None if no variation is found
        variation = var_match.group(2) if var_match.group(2) else None
    else:
        parsed_base = left
        variation = None

    # 3. Categorize and determine lemma
    sign_type = "lexical"
    lemma = parsed_base

    # Check against standard structural patterns
    if pointing_pattern and re.fullmatch(pointing_pattern, parsed_base):
        sign_type = "pointing"
        lemma = None
    elif depictive_pattern and re.fullmatch(depictive_pattern, parsed_base):
        sign_type = "depictive"
        lemma = None
    elif buoys_pattern and re.fullmatch(buoys_pattern, parsed_base):
        sign_type = "buoy"
        lemma = None
    elif palm_up_pattern and re.fullmatch(palm_up_pattern, parsed_base):
        sign_type = "palm_up"
        lemma = None
    elif gesture_pattern and re.fullmatch(gesture_pattern, parsed_base):
        sign_type = "gesture"
        lemma = None
    elif fingerspelling_pattern and re.fullmatch(fingerspelling_pattern, parsed_base):
        sign_type = "fingerspelling"
        lemma = None
    elif number_pattern and re.fullmatch(number_pattern, parsed_base):
        sign_type = "number"
        lemma = None
    elif init_pattern and re.fullmatch(init_pattern, parsed_base):
        sign_type = "initialization"
        lemma = None
    elif cued_speech_pattern and re.fullmatch(cued_speech_pattern, parsed_base):
        sign_type = "cued_speech"
        lemma = None
    elif unclear_pattern and re.fullmatch(unclear_pattern, parsed_base):
        sign_type = "unclear"
        lemma = None

    # Check against patterns that extract specific lemmas from capture groups
    elif entity_pattern and (ent_match := re.fullmatch(entity_pattern, parsed_base)):
        sign_type = "entity"
        lemma = ent_match.group(1)  # Evaluates to None for anonymous $NAME / $ORG
    elif morpheme_pattern and (
        morph_match := re.fullmatch(morpheme_pattern, parsed_base)
    ):
        sign_type = "morpheme"
        lemma = morph_match.group(1)
    elif foreign_pattern and (
        foreign_match := re.fullmatch(foreign_pattern, parsed_base)
    ):
        sign_type = "foreign"
        lemma = foreign_match.group(1)

    return lemma, sign_type, specifier, variation
