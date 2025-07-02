# modules/emotion_layer.py

"""
Emotion Layer for SelfDoubt.AI

This module provides empathetic responses based on the model's confidence label.
"""

import random

def emotional_response(label):
    """
    Returns a human-like emotional response based on the model's predicted doubt level.

    Args:
        label (str): One of 'CONFIDENT', 'DOUBTFUL', or 'NEUTRAL'

    Returns:
        str: An empathetic response
    """
    label = label.upper()

    if label == "DOUBTFUL":
        responses = [
            "I'm unsure... I don't have enough context.",
            "I'm hesitant — this could go either way.",
            "I feel uncertain about this one.",
            "There's too much ambiguity here for me to decide."
        ]

    elif label == "CONFIDENT":
        responses = [
            "I'm confident! That seems quite clear.",
            "I strongly believe this is correct.",
            "Yes — that feels right to me.",
            "I have no doubts about this one."
        ]

    elif label == "NEUTRAL":
        responses = [
            "I'm neither sure nor unsure — this feels neutral.",
            "It doesn't strongly trigger confidence or doubt.",
            "This is somewhere in the middle.",
            "I'm balanced on this — not leaning either way."
        ]

    else:
        responses = [
            "Hmm... I'm not sure how I feel about this.",
            "I can't quite place this in terms of doubt or certainty."
        ]

    return random.choice(responses)
