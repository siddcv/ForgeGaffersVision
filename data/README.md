# Input data

```
data/
  manifest.csv              # one row per clip: setup and conditions (tracked in git)
  clips/                    # videos (gitignored)
    <YYYY-MM-DD>_<place>/   # one folder per recording session
      <phone filename>.mp4
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
| `split` | `train` or `test`. Hold out whole clips, ideally a whole player, never single frames. |

## Labels
One CSV per labelled clip, `labels/<clip>.csv`, with times in seconds from the start of the video:

| `kind` | `t_s` | `end_s` | `foot` |
|---|---|---|---|
| `touch` | moment of contact | blank | `left` / `right` (the player's foot) |
| `lost_control` | ball escapes | ball back under control | blank |
| `reset` | play restarts | blank | blank |
| `head_down` | head goes down | head comes back up | blank |

Label from the original video at full frame rate, stepping frame by frame around each touch.
