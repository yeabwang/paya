# SPOTR encoder source

Source: https://github.com/5GYYYYY/SPOTR

Commit: `57b3c9f47cc7d20bc9852b84df9545f1dbb887f6`.

Original `SPOTR.py` SHA-256: `6c1be4d2d596d9bd14a631939d5cfe1c3ce4335acb90a7b2ffe969d47954dc22`.

Upstream encoder classes retain their operations and state-dict keys; decoder, classifier and demonstration code are excluded. This file also contains the integration: asset key `spotr`, pinned identity, capacity limits, asset conversion, strict loading and global-token selection. Shared modules access it through `paya.vendor.signal`, while assets retain their explicit `third-party/spotr` directory. The checkpoint's final norm parameter is retained even though upstream encoding does not apply it. See LICENSE for the MIT terms.
