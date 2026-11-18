# Input data

```
data/
  manifest.csv              # one row per clip: setup and conditions (tracked in git)
  clips/                    # videos (gitignored)
    tune/                   # clips used to choose thresholds
      <YYYY-MM-DD>_<place>/ # one folder per recording session
        <phone filename>.mp4
    test/                   # held-out clips: scored once, at the end, never used to choose thresholds
      <YYYY-MM-DD>_<place>/
    excluded/               # kept but not used: drill not performed properly (reason in manifest notes)
      <YYYY-MM-DD>_<place>/
  labels/                   # hand-labelled ground truth (tracked in git)
    <clip>.csv              # same stem as the clip
    TEMPLATE.csv
```

## Clips
- Keep the phone's original filename (e.g. `20261101_183403.mp4`). It is unique, and the cache and output
  folders are named after it, so never reuse a name across sessions.
- Copy the original file off the phone by cable or AirDrop. Never send it through a messaging app, which re-compresses it.
- Don't trim or edit the video. Instead, record where the drill starts and ends in `manifest.csv`.
- Speed-check rolls are clips too: put them in the session folder and set `take` to `speed-roll`.

## manifest.csv
Add a row for every clip on the day it is recorded, while the setup is fresh. Leave unknown fields blank.

| Column | Values |
|---|---|
| `take` | `baseline`, `head-up`, `head-down`, `head-timed`, `head-angles`, `eyes-only`, `look-at-phone`, `slow-fast`, `repeat-touches`, `lost-in-frame`, `lost-out-of-frame`, `near-miss`, `no-touch`, `speed-roll` |
| `camera` | `selfie` or `rear` |
| `fps` | Frame rate as recorded: `60` or `30` |
| `height_m`, `distance_m` | Phone lens height and distance to the player, in meters |
| `angle_deg` | 0 = straight on |
| `drill_start_s`, `drill_end_s` | Time of the clap, and when the drill ends (before anyone walks to the phone) |
| `split` | `tune`, `test` or `excluded`, matching the clip's folder. Exclude a clip only for the recording itself (drill not done properly), never because the pipeline scores badly on it. Hold out whole clips, ideally a whole player, never single frames. Decide before looking at a clip's results; test clips should span the same mix of players and places as tune clips. |

## Labels
One CSV per labelled clip, `labels/<clip>.csv`, with times in seconds from the start of the video:

| `kind` | `t_s` | `end_s` | `foot` |
|---|---|---|---|
| `touch` | push-off: when the ball is clearly sent back (tick-tocks only) | blank | `left` / `right` (the player's foot) |
| `lost_control` | ball escapes | ball back under control | blank |
| `reset` | play restarts | blank | blank |
| `head_down` | head goes down | head comes back up | blank |

Label from the original video at full frame rate, stepping frame by frame around each touch.

The easiest way is `python -m src.label_clip data/clips/<tune|test>/<session>/<clip>.mp4`, which stamps each key
press with the frame's timestamp and saves `labels/<clip>.csv` after every change:

| Key | Label |
|---|---|
| `l` / `r` | touch with the player's left / right foot (the player's own foot, not the screen side) |
| `x` / `y` | lost control starts / control regained (`y` also writes the `reset` row) |
| `h` / `u` | head down starts / head back up |
| `e` / `z` | erase the nearest label / undo |
| space, `,` `.`, `j` `k`, `[` `]` | play-pause, one frame back-forward, one second back-forward, slower-faster |

Label in three passes, one kind of event at a time, all saving into the same `labels/<clip>.csv`:
`--focus touches` (keys `l` `r`, starts at 0.25x), `--focus lost` (`x` `y`, 0.5x) and `--focus head`
(`h` `u`, 0.5x). A pass ignores the other passes' keys and only erases its own labels; their labels
stay dimmed on the timeline. Run one pass at a time per clip, since each pass rewrites the file.

For touches, pause and step to the contact frame before pressing, rather than pressing while it plays.
On quit it prints the touch gaps and interval durations, a first hint for the thresholds in `config.py`.
