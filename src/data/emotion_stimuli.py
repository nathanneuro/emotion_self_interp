"""Curated emotion stimuli for substrate-level probe extraction.

Six emotions covering the valence × arousal corners plus the alignment-relevant
calm ↔ desperate pair (Sofroniew et al. 2026):

                  high arousal       low arousal
    positive      blissful           calm
    negative      desperate, afraid, sad
                  hostile

Each emotion has stimuli at three levels:
    euphoric       — direct emotion-naming. Concentrated, low-context.
    naturalistic   — situational evocation; emotion not named. Off-policy
                     story snippets in the Sofroniew style.
    neutral        — content-matched control prompts shared across emotions.

The stimuli are intentionally short (typically 1–3 sentences) so the last-token
residual carries the operative emotion concept, matching Sofroniew's local-scope
finding (the vector tracks the operative concept at a token position rather
than a persistent character state).

Quantities are deliberately modest (~30 per cell). Treat this set as v0; it
should be expanded and re-validated as the program progresses.
"""
from __future__ import annotations

from itertools import product

from src.data.stimuli import Stimulus

EMOTIONS = ["calm", "desperate", "blissful", "sad", "afraid", "hostile"]

# ----- euphoric level (direct emotion-naming) ---------------------------------

_EUPHORIC_FRAMES = [
    "{character} felt utterly {adj}.",
    "A wave of {noun} washed over {character}.",
    "{character} could not stop feeling {adj}.",
    "Everything about that moment was {adj}.",
    "{character} was {adj} beyond words.",
    "There was nothing in {character}'s mind but {noun}.",
    "{character}'s whole body told them they were {adj}.",
    "It was the most {adj} {character} had ever been.",
    "{character} sat there, completely {adj}.",
    "The feeling was unmistakable: {character} was {adj}.",
]

_CHARACTERS = [
    "she", "he", "they", "the boy", "the woman",
    "the old man", "the child", "the doctor", "the traveler", "the soldier",
]

# (adj, noun) pairs used in euphoric frames.
_EMOTION_LEXICON: dict[str, list[tuple[str, str]]] = {
    "calm": [
        ("calm", "calm"),
        ("at peace", "peace"),
        ("serene", "serenity"),
        ("relaxed", "relaxation"),
        ("tranquil", "tranquility"),
    ],
    "desperate": [
        ("desperate", "desperation"),
        ("frantic", "panic"),
        ("hopeless", "hopelessness"),
        ("at the end of their rope", "desperation"),
        ("driven to the edge", "desperation"),
    ],
    "blissful": [
        ("blissful", "bliss"),
        ("ecstatic", "ecstasy"),
        ("euphoric", "euphoria"),
        ("overjoyed", "joy"),
        ("rapturous", "rapture"),
    ],
    "sad": [
        ("sad", "sorrow"),
        ("miserable", "misery"),
        ("heartbroken", "grief"),
        ("dejected", "dejection"),
        ("sorrowful", "sadness"),
    ],
    "afraid": [
        ("afraid", "fear"),
        ("terrified", "terror"),
        ("frightened", "fright"),
        ("petrified", "dread"),
        ("scared", "fear"),
    ],
    "hostile": [
        ("hostile", "hostility"),
        ("furious", "fury"),
        ("seething", "anger"),
        ("enraged", "rage"),
        ("antagonistic", "hostility"),
    ],
}


def _generate_euphoric(emotion: str, n: int) -> list[str]:
    out: list[str] = []
    pairs = _EMOTION_LEXICON[emotion]
    # Round-robin over (frame, char, lexicon_pair); keep order deterministic.
    combos = list(product(_EUPHORIC_FRAMES, _CHARACTERS, pairs))
    for frame, ch, (adj, noun) in combos:
        sentence = frame.format(character=ch, adj=adj, noun=noun)
        # Capitalize the leading character if the frame starts with one.
        if sentence and sentence[0].islower():
            sentence = sentence[0].upper() + sentence[1:]
        out.append(sentence)
        if len(out) >= n:
            break
    return out


# ----- naturalistic level (scenarios that evoke the emotion) ------------------
# Hand-curated short Sofroniew-style scenarios. The emotion is *not* named.

_NATURALISTIC: dict[str, list[str]] = {
    "calm": [
        "The lake had not moved all morning, and neither had I.",
        "Steam rose from the cup, and there was nowhere I needed to be.",
        "Light slanted through the curtains. I let my breath out slowly.",
        "Outside, the snow muffled every sound on the street.",
        "I closed the book and just listened to the rain.",
        "The garden held its quiet through the long afternoon.",
        "I rested my hand on the warm windowsill and watched the light.",
        "There was nothing to fix; the day could simply be the day.",
        "Her shoulders softened as the kettle began its low whistle.",
        "He sat on the porch, untroubled by anything in particular.",
    ],
    "desperate": [
        "Three a.m. and the test results still weren't back. I refreshed the page again.",
        "It was the seventh hour without word from her son.",
        "The dose on the bottle said two; he had already swallowed nine.",
        "The signal cut out again and the cliff was getting closer.",
        "I tried every name in the address book. None of them picked up.",
        "If the wire didn't hold, there was nothing else holding her.",
        "He counted what was left in the bottle. It would not be enough.",
        "The water was rising past the second step now.",
        "She kept dialing the same number and hearing the same tone.",
        "There were no exits left to try, and the door was warm to the touch.",
    ],
    "blissful": [
        "When she finally said yes, the whole street seemed to brighten.",
        "He held the new baby and forgot, for a moment, that anything else existed.",
        "The first taste was so good he laughed out loud, alone in the kitchen.",
        "The crowd was singing the chorus and she was singing with them.",
        "After eight years of failed attempts, the rocket lifted, and so did everyone.",
        "Sunlight, salt, the dog running ahead — every small thing felt enormous.",
        "He read the message three times, grinning at the empty room.",
        "She closed her eyes on the summit and felt the wind go through her.",
        "The audience was on its feet before the last note had even faded.",
        "When the doctor said the word 'cured,' he had to sit down.",
    ],
    "sad": [
        "I read his letters for the last time and folded them away.",
        "The dog's bowl had been empty since Tuesday, and I hadn't yet moved it.",
        "She left her wedding ring on the counter and walked out.",
        "The old house was already half-emptied of his things.",
        "The baby's room was finished; the baby was not coming home.",
        "I kept his voicemail. I will not delete it.",
        "The chair by the window was empty for the first morning in forty years.",
        "Her name was still on the joint account, and on nothing else.",
        "He picked up the phone to call her before he remembered.",
        "The rain came in through the window onto her unmade bed.",
    ],
    "afraid": [
        "Footsteps in the corridor. Then a pause, just outside the door.",
        "The headlights behind us hadn't fallen back in twenty miles.",
        "Something was breathing in the dark, and it was not the dog.",
        "She heard the lock click on the wrong side of the door.",
        "The lump on the scan was larger than last time.",
        "He realized he had taken the wrong turn an hour ago.",
        "The plane dropped, then dropped again.",
        "The man in the hallway was wearing her father's coat.",
        "Her phone showed no signal and the trail had ended.",
        "Something heavy moved in the attic above her bed.",
    ],
    "hostile": [
        "If he so much as said her name again, she was going to swing.",
        "He kept his fist closed at his side, watching the man with the keys.",
        "Her smile did not reach any part of her face.",
        "If you take one more step toward my daughter, I'll end you.",
        "He pictured the manager's car burning and felt a little better.",
        "She set the knife down, very deliberately, on his side of the table.",
        "He memorized the license plate so he could find them again later.",
        "There would be a reckoning, and there would not be witnesses.",
        "Every word out of his mouth was another reason to hate him.",
        "She watched him walk away and made a list of what she owed him.",
    ],
}


# ----- neutral level (shared control, content-matched) ------------------------

_NEUTRAL: list[str] = [
    "The bus arrived at the stop at seven forty-three.",
    "She sorted the books on the shelf alphabetically by author.",
    "The blue notebook was on the second shelf from the top.",
    "He filled out the form using a black pen.",
    "The recipe called for two cups of flour and one cup of milk.",
    "The map showed the river running roughly east to west.",
    "She copied the address onto the envelope.",
    "The package was scheduled to arrive on Thursday.",
    "He set the timer for fifteen minutes and went back to the desk.",
    "The thermostat in the hallway was set to sixty-eight degrees.",
    "The library closed at six on weekdays and at five on weekends.",
    "He counted the boxes in the back of the truck.",
    "The plane was scheduled to land at gate twenty-two.",
    "She updated the spreadsheet with the new figures.",
    "The map on the wall showed the train lines in different colors.",
    "He filled the kettle and pressed the switch.",
    "The book had four hundred and twelve pages.",
    "The receipt was folded into the third pocket of the wallet.",
    "She labeled each folder with a date and a project name.",
    "The window cleaner came on the first Tuesday of every month.",
    "He measured the room and wrote the dimensions on a sticky note.",
    "The schedule said the meeting would last forty-five minutes.",
    "She moved the lamp two feet to the left.",
    "The forecast called for partial cloud cover and a light wind.",
    "He renewed his library card at the front desk.",
    "The grocery list had milk, bread, eggs, and a head of lettuce.",
    "She filed the receipts in the order they had arrived.",
    "The clock on the kitchen wall read four-fifteen.",
    "The hardware store was on the corner of Third and Pine.",
    "He cleaned the lenses with the cloth from the case.",
]


# ----- assembly ---------------------------------------------------------------


def build_stimulus_set(per_cell: int = 30) -> list[Stimulus]:
    """Return the full v0 stimulus set as a flat list of Stimulus records.

    `per_cell` is capped per (emotion, level) cell. Naturalistic cells currently
    have ten curated items each; euphoric cells generate up to per_cell from
    template combinations. The neutral set is shared across emotions.
    """
    out: list[Stimulus] = []
    for emotion in EMOTIONS:
        for prompt in _generate_euphoric(emotion, per_cell):
            out.append(Stimulus(
                id=f"{emotion}/euphoric/{len(out)}",
                emotion=emotion,
                level="euphoric",
                prompt=prompt,
            ))
        for prompt in _NATURALISTIC[emotion][:per_cell]:
            out.append(Stimulus(
                id=f"{emotion}/naturalistic/{len(out)}",
                emotion=emotion,
                level="naturalistic",
                prompt=prompt,
            ))

    # One shared neutral set, tagged with the sentinel emotion "neutral" so
    # contrast operations are uniform across emotions.
    for i, prompt in enumerate(_NEUTRAL[:per_cell]):
        out.append(Stimulus(
            id=f"neutral/neutral/{i}",
            emotion="neutral",
            level="neutral",
            prompt=prompt,
        ))
    return out


def split_by(emotion: str | None, level: str | None, stims: list[Stimulus]) -> list[Stimulus]:
    """Filter helper used by the probe extraction script."""
    return [s for s in stims
            if (emotion is None or s.emotion == emotion)
            and (level is None or s.level == level)]
