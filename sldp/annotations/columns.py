GLOSS_COLUMNS = [
    "start_ms",
    "end_ms",
    "gloss",
    "start_frame",
    "end_frame",
    "lemma",
    "sign_type",
    "specifier",
    "variation",
]

SUBTITLE_COLUMNS = [
    "start_ms",
    "end_ms",
    "text",
    "start_frame",
    "end_frame",
]

DEFAULT_COLUMNS = {
    "left_hand": GLOSS_COLUMNS,
    "right_hand": GLOSS_COLUMNS,
    "both_hands": GLOSS_COLUMNS,
    "translation": SUBTITLE_COLUMNS,
}