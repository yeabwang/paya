# SPOTR encoder source

Source: https://github.com/5GYYYYY/SPOTR

Commit: `57b3c9f47cc7d20bc9852b84df9545f1dbb887f6`.

Original `SPOTR.py` SHA-256: `6c1be4d2d596d9bd14a631939d5cfe1c3ce4335acb90a7b2ffe969d47954dc22`.

Only imports and encoder helper classes are retained. Formatting and comments are simplified; encoder operations and state-dict keys are unchanged. Decoder, classifier and demonstration code are excluded. The checkpoint's final norm parameter is retained even though upstream encoding does not apply it. See LICENSE for the MIT terms.
