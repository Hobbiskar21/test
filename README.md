# Cricket Video Test Export

This branch contains two runners:

```text
cricket_bowling_analysis/batsman.py
cricket_bowling_analysis/bowler.py
```

## Input Videos

Put batting videos here:

```text
input_videos/test/batsman/
```

Put side-on bowling videos here:

```text
input_videos/test/bowling/
```

## Run

Open terminal in:

```text
cricket_bowling_analysis/
```

Install requirements from the repo root:

```bat
pip install -r requirements.txt
```

The YOLO pose weights are not committed to this repo. If needed, place `yolov8m-pose.pt` inside `cricket_bowling_analysis/`, or let Ultralytics download the model when the pose detector runs.

For batting:

```bat
python batsman.py
```

For one bowling video:

```bat
python bowler.py
```

For all bowling videos:

```bat
python bowler.py --all
```

To clear old outputs:

```bat
python clear.py
```

## Outputs

Batting outputs are saved under:

```text
outputs/test_output/batsman/
```

Bowling outputs are saved under:

```text
outputs/test_output/bowling/
```

## Metrics

Batting metrics:

```text
Weight Transfer (COM Shift)
Front Knee Angle at Impact
Head Stability Index
```

Bowling metrics:

```text
Front knee angle at Front Foot Contact
Delivery stride length
Trunk lateral flexion at release
```

The metric CSVs are saved in each output folder, and the important metric overlays are shown in the final videos.
