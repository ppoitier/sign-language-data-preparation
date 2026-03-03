import re


def categorize_gloss(
    gloss: str,
    variation_pattern: str | None = r"^(.*?)((\([^)]*\))?\*?(?:-(?:1|2)h)?\*?\+*)$",
    pt_pattern: str | None = "^pt$",
    depictive_pattern: str | None = "^ds$",
    buoys_pattern: str | None = "^lbuoy$",
    palm_up_pattern: str | None = "^palm-up$",
) -> tuple[str | None, str, str | None, str | None]:
    """
    Parses a sign language gloss into its linguistic components.

    Args:
        gloss (str): The raw gloss string (e.g., 'pt:pro1++++', 'ns:belgique(b-joue)', 'lbuoy(6):un').
        variation_pattern (str | None): Regex to extract variations.
        pt_pattern (str | None): Regex for pointing signs.
        depictive_pattern (str | None): Regex for depictive signs.
        buoys_pattern (str | None): Regex for buoys.
        palm_up_pattern (str | None): Regex for palm-ups.

    Returns:
        tuple: (lemma, sign_type, specifier, variation)
    """

    # 1. Extract variation from the end of the string
    if variation_pattern and (var_match := re.fullmatch(variation_pattern, gloss)):
        core_gloss = var_match.group(1)
        variation = var_match.group(2) or None
    else:
        core_gloss = gloss
        variation = None

    # 2. Extract the specifier and handle left-side variations
    if ":" in core_gloss:
        left, right = core_gloss.split(":", maxsplit=1)

        # Strip potential variation from the left side (e.g., 'lbuoy(6)')
        left_base = left
        if variation_pattern and (left_match := re.fullmatch(variation_pattern, left)):
            left_base = left_match.group(1)
            if left_var := left_match.group(2):
                variation = left_var  # Overwrite with the main left-side variation

        # Check if the clean left side is a structural base
        is_structural = False
        for pattern in (pt_pattern, depictive_pattern, buoys_pattern, palm_up_pattern):
            if pattern and re.fullmatch(pattern, left_base):
                is_structural = True
                break

        # Assign base and specifier based on structural check
        if is_structural:
            parsed_base = left_base
            specifier = right
        else:
            parsed_base = right
            specifier = left
    else:
        parsed_base = core_gloss
        specifier = None

    # 3. Categorize and nullify non-lemmas
    sign_type = "lexical"
    lemma = parsed_base

    if pt_pattern and re.fullmatch(pt_pattern, parsed_base):
        sign_type = "pointing"
        lemma = None
    elif depictive_pattern and re.fullmatch(depictive_pattern, parsed_base):
        sign_type = "depictive"
        lemma = None
    elif buoys_pattern and re.fullmatch(buoys_pattern, parsed_base):
        sign_type = "buoy"
        lemma = None
    elif palm_up_pattern and re.fullmatch(palm_up_pattern, parsed_base):
        sign_type = "palm-up"
        lemma = None

    return lemma, sign_type, specifier, variation
