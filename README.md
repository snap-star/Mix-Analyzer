# Mix-Analyzer
Mixing analyzer with reports

# How to Use It
1. Install dependencies
```bash
pip install numpy scipy matplotlib
```
2. How To Use It
```bash
python mix_analyzer.py "your_song.wav"
```
3. What it outputs
Two PNG charts saved next to your audio file:
- your_song_analysis.png — Waveform, stereo field, frequency balance, dynamics
- your_song_diagnosis.png — Problem highlights + fix recommendations

Plus a full text report in the terminal:
```txt
FREQUENCY BALANCE ANALYSIS
Sub Bass        (   20-   60 Hz):   -31.5 dB
Bass            (   60-  120 Hz):   -26.0 dB
...
DIAGNOSIS
1. LOW-MID BUILDUP (120-250 Hz): -33.6 dB
   vs Bass (60-120 Hz):           -26.0 dB
   DIFFERENCE: -7.7 dB (target: -3 to -6 dB)
   --> OK
...
STEREO WIDTH BY BAND
Bass           : Side - Mid =  -21.1 dB  (NEARLY MONO)
...
DYNAMIC RANGE BY SECTION
Intro          : Avg= -25.6 dB | Range=10.0 dB
...
```

# Customizing the Script
## Option A: Define your own song sections
Uncomment and edit the `sections` list in `__main__`:
```python
sections = [
    ("Intro", 0, 15),
    ("Verse 1", 15, 45),
    ("Pre-Chorus", 45, 60),
    ("Chorus 1", 60, 90),
    ("Verse 2", 90, 120),
    ("Chorus 2", 120, 150),
    ("Bridge", 150, 180),
    ("Final Chorus", 180, 210),
    ("Outro", 210, 240),
]
generate_report(file_path, target_sr=22050, sections=sections)
```
## Option B: Change the target slope
In the `generate_report` function, find this line:
```python
target = bass_level - 4.5 * np.log2(freqs / 100)
```

- Increase 4.5 for a steeper slope (darker target)
- Decrease 4.5 for a flatter slope (brighter target)

## Option C: Full-resolution analysis (slower)
Change `target_sr` from `22050` to `None` to analyze at the file's native sample rate:
```python
generate_report(file_path, target_sr=None)
```
This is more accurate but uses more RAM on long files.

# What the numbers mean

| Metric                 | Good Range   | If Too High         | If Too Low               |
| ---------------------- | ------------ | ------------------- | ------------------------ |
| **Low-Mid vs Bass**    | -3 to -6 dB  | Muddy, masks detail | Thin, weak body          |
| **Air vs Detail**      | -3 to -6 dB  | Harsh, hissy        | Closed in, no room       |
| **Ultra-Air**          | > -35 dB     | —                   | No shine, dimensionless  |
| **Dynamic Range**      | > 6 dB       | —                   | Flat, lifeless, no depth |
| **Stereo Width (Air)** | -12 to -8 dB | Too diffuse         | Narrow, centered         |

The script handles both mono and stereo files, auto-downsamples for memory efficiency, and works with any standard WAV format.
