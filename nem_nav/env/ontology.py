"""Semantic ontology for the SemanticHomeWorld testbed.

The ontology defines a small but realistic set of room categories and object
classes, together with a hand-curated co-occurrence matrix used to derive
fixed semantic embeddings via spectral decomposition. This serves as a
deterministic stand-in for a frozen vision-language encoder (e.g. CLIP) in a
controlled benchmarking setting.

A real OpenCLIP backbone can be substituted by re-implementing
``encode_image`` and ``encode_text`` in
``nem_nav.models.perception.encoder``; the rest of the system is unchanged.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

ROOM_TYPES: List[str] = [
    "kitchen",
    "bedroom",
    "bathroom",
    "living_room",
    "office",
    "dining_room",
    "hallway",
]

OBJECT_CLASSES: List[str] = [
    # kitchen
    "fridge", "stove", "sink", "microwave", "kettle",
    # bedroom
    "bed", "pillow", "wardrobe", "nightstand",
    # bathroom
    "toilet", "shower", "bathtub", "towel",
    # living_room
    "sofa", "tv", "coffee_table", "bookshelf",
    # office
    "desk", "computer", "chair", "printer",
    # dining_room
    "dining_table", "wine_glass",
    # hallway
    "umbrella", "shoe_rack",
]

# Hand-curated room/object affinity (rows: object, cols: room). The values are
# typical-co-occurrence priors in [0,1]; they are *not* used as planning oracles
# directly, only to derive fixed semantic embeddings via SVD.
_OBJECT_TO_PRIMARY_ROOM: Dict[str, str] = {
    "fridge": "kitchen", "stove": "kitchen", "sink": "kitchen",
    "microwave": "kitchen", "kettle": "kitchen",
    "bed": "bedroom", "pillow": "bedroom", "wardrobe": "bedroom",
    "nightstand": "bedroom",
    "toilet": "bathroom", "shower": "bathroom", "bathtub": "bathroom",
    "towel": "bathroom",
    "sofa": "living_room", "tv": "living_room", "coffee_table": "living_room",
    "bookshelf": "living_room",
    "desk": "office", "computer": "office", "chair": "office",
    "printer": "office",
    "dining_table": "dining_room", "wine_glass": "dining_room",
    "umbrella": "hallway", "shoe_rack": "hallway",
}

# A few cross-room affinities (e.g. a chair can also live in a dining room).
_SECONDARY_AFFINITIES: Dict[str, Dict[str, float]] = {
    "chair": {"dining_room": 0.6, "kitchen": 0.3},
    "sink": {"bathroom": 0.6},
    "bookshelf": {"office": 0.5, "bedroom": 0.3},
    "computer": {"living_room": 0.2, "bedroom": 0.2},
    "tv": {"bedroom": 0.4},
    "towel": {"kitchen": 0.3},
    "coffee_table": {"office": 0.2},
    "wardrobe": {"hallway": 0.3},
    "shoe_rack": {"bedroom": 0.2},
    "umbrella": {"living_room": 0.1},
}


def affinity_matrix() -> np.ndarray:
    """Return an (|objects|, |rooms|) matrix of room-object priors in [0, 1]."""
    M = np.zeros((len(OBJECT_CLASSES), len(ROOM_TYPES)), dtype=np.float32)
    for i, obj in enumerate(OBJECT_CLASSES):
        primary = _OBJECT_TO_PRIMARY_ROOM[obj]
        M[i, ROOM_TYPES.index(primary)] = 1.0
        for room, w in _SECONDARY_AFFINITIES.get(obj, {}).items():
            M[i, ROOM_TYPES.index(room)] = max(M[i, ROOM_TYPES.index(room)], w)
    return M


def primary_room(obj: str) -> str:
    return _OBJECT_TO_PRIMARY_ROOM[obj]


def objects_for_room(room: str) -> List[str]:
    return [o for o, r in _OBJECT_TO_PRIMARY_ROOM.items() if r == room]
