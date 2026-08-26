"""
Similarity calibration.

intfloat/e5-large-v2 is a retrieval model: it packs all English text into a
narrow cone, so two completely unrelated requirements still score around 0.75
cosine and outright nonsense scores about 0.69. Measured over 1,770 random
requirement pairs from this database:

    p5 0.712   p25 0.739   median 0.757   p75 0.777   p95 0.810   p99 0.846

Genuine matches score 0.92 - 0.99, so the entire useful signal lives in the top
quarter of the range. That is why a "70%" threshold accepted everything: 0.70
sits far below the floor of the data.

all-mpnet-base-v2, which this project used before e5, is trained for semantic
similarity rather than retrieval and spreads the same pairs across the full
range - unrelated text lands near 0.1 and matches near 0.8. The thresholds and
the (1 + cosine) / 2 display in this codebase were written for that behaviour.

calibrate() rescales e5's cosine so it behaves like mpnet's, without changing
the model or re-embedding anything:

    calibrated = (cosine - FLOOR) / (1 - FLOOR),  clamped to [0, 1]

FLOOR is the median of unrelated pairs, so a typical unrelated pair scores 0
and only similarity above the noise floor counts. Checked against mpnet on the
same pairs:

    pair                                    e5 raw   calibrated   mpnet
    "antistatic finger bags" / "spare parts"  0.787       0.148    0.159
    "PQ support" / "archive backup database"  0.773       0.092    0.183
    "banana bread" / "exhaust blower VFD"     0.687       0.000   -0.152
    archive backup / archive backup           0.946       0.784    0.871
    lock users out / lock the account         0.927       0.708    0.718

Thresholds are therefore back on a scale where 0.70 means what it looks like,
and every layer uses the same scale.
"""

import os
import re

# Median cosine of unrelated requirement pairs under e5-large-v2.
SIMILARITY_FLOOR = float(os.getenv("SIMILARITY_FLOOR", "0.75"))


def calibrate(cosine: float) -> float:
    """
    Rescale a raw cosine similarity so the noise floor becomes 0.0.

    Values at or below the floor return 0.0; an exact match returns 1.0.
    """
    if cosine is None:
        return 0.0
    span = 1.0 - SIMILARITY_FLOOR
    if span <= 0:
        return max(0.0, min(1.0, float(cosine)))
    scaled = (float(cosine) - SIMILARITY_FLOOR) / span
    return max(0.0, min(1.0, scaled))


def calibrate_from_distance(distance: float) -> float:
    """
    Same, for pgvector's cosine distance (<=>), where cosine = 1 - distance.

    Replaces the old `1 - distance / 2`, which mapped the theoretical [0, 2]
    distance range onto [0, 1]. That is correct in general but wrong for this
    data: e5 never produces anti-parallel vectors, so it compressed every real
    score into 0.85 - 0.99.
    """
    return calibrate(1.0 - float(distance))


# Leading document section numbers - "5.4.4.13 - ", "8.2.9 ", "4.2 - ".
#
# These identify where a requirement sits in one document, not what it asks
# for, so the same requirement carries a different prefix in every document and
# the difference dilutes the match. Measured on a real pair that differs only
# by section number and two typos:
#
#     with prefixes     cosine 0.9729 -> calibrated 0.89
#     prefixes removed  cosine 0.9960 -> calibrated 0.98
#
# 100 of 331 requirements in one URS carry such a prefix.
#
# The pattern deliberately requires either a multi-part number, or a separator
# followed by a letter, so measurement ranges survive untouched: "2 - 4 bar",
# "250L - 300L bowl", "0.4 mm surface roughness", "1 -3 bar".
SECTION_PREFIX_RE = re.compile(
    r'^\s*(?:\d+(?:\.\d+)+\s*[-–—:]\s*'      # 5.4.4.13 -
    r'|\d+(?:\.\d+){2,}\s+'                    # 8.2.9 (three or more parts)
    r'|\d+\s*[-–—]\s+(?=[A-Za-z]))'             # 4 - Guidelines
)


def normalise_for_embedding(text: str) -> str:
    """
    Strip a leading section number before embedding.

    Only affects the vector; the stored and displayed requirement keeps its
    original text.
    """
    if not text:
        return ""
    return SECTION_PREFIX_RE.sub("", text).strip() or text.strip()
