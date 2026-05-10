#!/usr/bin/env python3
"""
Cold Orbit Mix Analyzer
========================
Analyzes stereo audio files for mixing issues:
- Frequency balance (bass, mids, highs, ultra-air)
- Stereo width by frequency band
- Dynamic range by time section
- Visual diagnosis charts

Requirements: numpy, scipy, matplotlib
Install: pip install numpy scipy matplotlib

Usage: python mix_analyzer.py your_song.wav
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, lfilter, resample


def load_audio(path, target_sr=None):
    """Load WAV file. Optionally downsample to save memory."""
    sr, data = wavfile.read(path)

    if len(data.shape) == 1:
        left = data.astype(np.float64)
        right = left.copy()
        is_stereo = False
    else:
        left = data[:, 0].astype(np.float64)
        right = data[:, 1].astype(np.float64)
        is_stereo = True

    # Downsample if requested (saves memory on long files)
    if target_sr is not None and sr != target_sr:
        num_samples = int(len(left) * target_sr / sr)
        left = resample(left, num_samples)
        right = resample(right, num_samples)
        sr = target_sr

    mono = (left + right) / 2.0

    # Normalize each channel independently for analysis
    left = left / (np.max(np.abs(left)) + 1e-10)
    right = right / (np.max(np.abs(right)) + 1e-10)
    mono = mono / (np.max(np.abs(mono)) + 1e-10)

    return sr, left, right, mono, is_stereo


def band_level_from_fft(psd, freqs, f_low, f_high):
    """Calculate total energy in a frequency band from PSD."""
    mask = (freqs >= f_low) & (freqs < f_high)
    return 10 * np.log10(np.sum(psd[mask]) + 1e-10)


def compute_fft_spectrum(data, sr):
    """Compute power spectral density via FFT."""
    n = len(data)
    fft_vals = np.abs(rfft(data)) / n
    freqs = rfftfreq(n, 1/sr)
    psd = fft_vals ** 2
    return freqs, psd


def analyze_frequency_balance(freqs, psd_mono, psd_mid, psd_side):
    """Print frequency band analysis."""
    bands = [
        ("Sub Bass", 20, 60),
        ("Bass", 60, 120),
        ("Low Mid", 120, 250),
        ("Mid", 250, 500),
        ("Upper Mid", 500, 1000),
        ("Presence", 1000, 2500),
        ("Detail", 2500, 6000),
        ("Air", 6000, 12000),
        ("Ultra Air", 12000, 20000),
    ]

    print("=" * 60)
    print("FREQUENCY BALANCE ANALYSIS")
    print("=" * 60)

    results = {}
    for name, lo, hi in bands:
        lvl = band_level_from_fft(psd_mono, freqs, lo, hi)
        results[name] = lvl
        print(f"{name:15s} ({lo:5d}-{hi:5d} Hz): {lvl:7.1f} dB")

    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)

    bass = results["Bass"]
    lowmid = results["Low Mid"]
    detail = results["Detail"]
    air = results["Air"]
    ultra = results["Ultra Air"]

    # Low-mid buildup check
    print(f"\n1. LOW-MID BUILDUP (120-250 Hz): {lowmid:.1f} dB")
    print(f"   vs Bass (60-120 Hz):           {bass:.1f} dB")
    diff = lowmid - bass
    print(f"   DIFFERENCE: {diff:.1f} dB (target: -3 to -6 dB)")
    if diff > -3:
        print("   --> TOO MUCH LOW-MID. MUDDY. MASKS DETAIL.")
    elif diff < -10:
        print("   --> LOW-MID TOO THIN. MIX MAY SOUND HOLLOW.")
    else:
        print("   --> OK")

    # Detail region
    presence = results["Presence"]
    print(f"\n2. DETAIL REGION (2.5-6 kHz):    {detail:.1f} dB")
    print(f"   vs Presence (1-2.5 kHz):       {presence:.1f} dB")
    if detail < presence - 6:
        print("   --> DETAIL DROPPING OFF TOO FAST. LACKS CLARITY.")
    else:
        print("   --> ENERGY EXISTS")

    # Air region
    print(f"\n3. AIR REGION (6-12 kHz):        {air:.1f} dB")
    print(f"   vs Detail (2.5-6 kHz):         {detail:.1f} dB")
    diff_air = air - detail
    print(f"   DIFFERENCE: {diff_air:.1f} dB (target: -3 to -6 dB)")
    if diff_air > -1:
        print("   --> AIR IS TOO LOUD. MAY SOUND HARSH/HISSY.")
    elif diff_air < -9:
        print("   --> SEVERE LACK OF AIR. MIX SOUNDS CLOSED IN.")
    elif diff_air < -6:
        print("   --> LOW AIR. MISSING ROOM AND SPACE.")
    else:
        print("   --> OK")

    # Ultra air
    print(f"\n4. ULTRA AIR (12-20 kHz):        {ultra:.1f} dB")
    if ultra < -40:
        print("   --> ESSENTIALLY MISSING. NO SHINE OR DIMENSION.")
    elif ultra < -35:
        print("   --> LOW ULTRA-AIR. ADD HIGH SHELF / SATURATION.")
    else:
        print("   --> OK")

    return results


def analyze_stereo_width(freqs, psd_mid, psd_side):
    """Analyze stereo width per frequency band."""
    bands = [
        ("Sub Bass", 20, 60),
        ("Bass", 60, 120),
        ("Low Mid", 120, 250),
        ("Mid", 250, 500),
        ("Upper Mid", 500, 1000),
        ("Presence", 1000, 2500),
        ("Detail", 2500, 6000),
        ("Air", 6000, 12000),
        ("Ultra Air", 12000, 20000),
    ]

    print("\n" + "=" * 60)
    print("STEREO WIDTH BY BAND (Side vs Mid energy)")
    print("=" * 60)

    width_results = {}
    for name, lo, hi in bands:
        mid_lvl = band_level_from_fft(psd_mid, freqs, lo, hi)
        side_lvl = band_level_from_fft(psd_side, freqs, lo, hi)
        diff = side_lvl - mid_lvl
        width_results[name] = diff

        if diff < -20:
            status = "NEARLY MONO"
        elif diff < -12:
            status = "NARROW"
        elif diff < -8:
            status = "MODERATE"
        else:
            status = "WIDE"

        print(f"{name:15s}: Side - Mid = {diff:6.1f} dB  ({status})")

    return width_results


def analyze_dynamics(mono, sr, sections=None):
    """Analyze dynamic range by song section."""
    ws = sr * 3      # 3-second window
    hop = sr         # 1-second hop
    n = (len(mono) - ws) // hop + 1

    loudness = []
    time_loud = []
    for i in range(n):
        s = i * hop
        e = s + ws
        block = mono[s:e]
        rms = np.sqrt(np.mean(block**2))
        loudness.append(20 * np.log10(rms + 1e-10))
        time_loud.append((s + ws/2) / sr)

    print("\n" + "=" * 60)
    print("DYNAMIC RANGE BY SECTION (3-second RMS windows)")
    print("=" * 60)

    if sections is None:
        # Auto-divide into 8 equal sections
        duration = len(mono) / sr
        sec_len = duration / 8
        sections = [
            ("Intro", 0, sec_len),
            ("Build", sec_len, sec_len*2),
            ("Chorus1", sec_len*2, sec_len*3),
            ("Verse2", sec_len*3, sec_len*4),
            ("Chorus2", sec_len*4, sec_len*5),
            ("Bridge", sec_len*5, sec_len*6),
            ("FinalCh", sec_len*6, sec_len*7),
            ("Outro", sec_len*7, duration),
        ]

    section_data = []
    for name, start, end in sections:
        s_idx = int(start * sr)
        e_idx = int(end * sr)
        if e_idx > len(mono):
            e_idx = len(mono)

        rms_vals = []
        for i in range((e_idx - s_idx - ws) // hop + 1):
            block = mono[s_idx + i*hop : s_idx + i*hop + ws]
            rms = np.sqrt(np.mean(block**2))
            rms_vals.append(20 * np.log10(rms + 1e-10))

        if rms_vals:
            avg_rms = np.mean(rms_vals)
            min_rms = np.min(rms_vals)
            max_rms = np.max(rms_vals)
            dr = max_rms - min_rms
            section_data.append((name, avg_rms, dr, start, end))
            print(f"{name:15s}: Avg={avg_rms:6.1f} dB | Range={dr:4.1f} dB")
        else:
            print(f"{name:15s}: (too short)")

    return np.array(time_loud), np.array(loudness), section_data


def generate_report(path, target_sr=22050, sections=None, output_prefix=None):
    """Run full analysis and generate charts."""
    if output_prefix is None:
        output_prefix = path.replace(".wav", "").replace(".mp3", "").replace(".flac", "")

    print(f"Loading: {path}")
    sr, left, right, mono, is_stereo = load_audio(path, target_sr=target_sr)
    duration = len(mono) / sr

    print(f"Sample Rate: {sr} Hz | Duration: {duration:.1f}s | Stereo: {is_stereo}")

    rms = np.sqrt(np.mean(mono**2))
    peak = np.max(np.abs(mono))
    crest = 20 * np.log10(peak / (rms + 1e-10))
    print(f"Peak: {20*np.log10(peak):.1f} dBFS | RMS: {20*np.log10(rms):.1f} dBFS | Crest: {crest:.1f} dB")

    # FFT analysis
    freqs, psd_mono = compute_fft_spectrum(mono, sr)
    if is_stereo:
        _, psd_left = compute_fft_spectrum(left, sr)
        _, psd_right = compute_fft_spectrum(right, sr)
        psd_mid = np.abs((np.sqrt(psd_left) + np.sqrt(psd_right)) / 2) ** 2
        psd_side = np.abs((np.sqrt(psd_left) - np.sqrt(psd_right)) / 2) ** 2
    else:
        psd_mid = psd_mono
        psd_side = psd_mono * 1e-10  # effectively zero

    # Text analysis
    freq_results = analyze_frequency_balance(freqs, psd_mono, psd_mid, psd_side)
    width_results = analyze_stereo_width(freqs, psd_mid, psd_side)
    time_loud, loudness, section_data = analyze_dynamics(mono, sr, sections)

    # === CHART 1: Full Analysis ===
    fig, axes = plt.subplots(4, 1, figsize=(16, 14))
    time = np.arange(len(mono)) / sr
    skip = max(1, len(time) // 50000)  # downsample for plotting

    # 1. Waveform
    axes[0].plot(time[::skip], left[::skip], color='steelblue', alpha=0.5, linewidth=0.15)
    if is_stereo:
        axes[0].plot(time[::skip], right[::skip], color='coral', alpha=0.5, linewidth=0.15)
    axes[0].set_xlim(0, duration)
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].set_title(f'Mix Analyzer: {path.split("/")[-1]} - Waveform', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.2)

    # 2. Stereo correlation
    if is_stereo:
        ws = sr * 2
        hop_c = sr
        n_win = (len(mono) - ws) // hop_c + 1
        corr_vals = []
        width_vals = []
        t_corr = []
        for i in range(n_win):
            s = i * hop_c
            e = s + ws
            l = left[s:e]
            r = right[s:e]
            corr_vals.append(np.corrcoef(l, r)[0,1])
            side = (l - r) / 2
            width_vals.append(20*np.log10(np.sqrt(np.mean(side**2)) + 1e-10))
            t_corr.append((s + ws/2) / sr)

        ax2 = axes[1]
        ax2_t = ax2.twinx()
        ax2.plot(t_corr, corr_vals, 'g-', linewidth=1.5, label='Correlation')
        ax2_t.plot(t_corr, width_vals, 'purple', linewidth=1.5, alpha=0.6, label='Side Energy')
        ax2.axhline(0, color='k', ls='--', alpha=0.3)
        ax2.set_xlim(0, duration)
        ax2.set_ylim(-1.05, 1.05)
        ax2.set_ylabel('L/R Correlation', color='green')
        ax2_t.set_ylabel('Side Energy (dB)', color='purple')
        ax2.set_title('Stereo Field Analysis', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper left')
        ax2_t.legend(loc='upper right')
        ax2.grid(True, alpha=0.2)
    else:
        axes[1].text(0.5, 0.5, 'Mono File - No Stereo Analysis', ha='center', va='center', fontsize=14)
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].axis('off')

    # 3. Frequency balance
    avg_db = 10 * np.log10(psd_mono + 1e-10)
    axes[2].semilogx(freqs, avg_db, color='darkblue', linewidth=2.2, label='Your Mix')

    # Target slope
    idx_100 = np.argmin(np.abs(freqs - 100))
    bass_level = avg_db[idx_100]
    target = bass_level - 4.5 * np.log2(freqs / 100)
    target[freqs < 100] = avg_db[freqs < 100]
    axes[2].semilogx(freqs, target, 'r--', linewidth=1.8, alpha=0.7, label='Target Indie Slope')

    axes[2].axvspan(2500, 6000, alpha=0.12, color='orange')
    axes[2].axvspan(7000, 16000, alpha=0.12, color='cyan')
    axes[2].text(4000, np.max(avg_db)+2, 'DETAIL', ha='center', fontsize=9, color='darkorange', fontweight='bold')
    axes[2].text(11000, np.max(avg_db)+2, 'AIR', ha='center', fontsize=9, color='darkcyan', fontweight='bold')
    axes[2].set_xlim(30, 20000)
    axes[2].set_ylim(np.min(avg_db)-6, np.max(avg_db)+6)
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Magnitude (dB)')
    axes[2].set_title('Average Frequency Balance vs. Target', fontsize=12, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, which='both')

    # 4. Dynamics
    axes[3].plot(time_loud, loudness, color='navy', linewidth=1.2)
    axes[3].fill_between(time_loud, loudness, -50, alpha=0.25, color='navy')
    axes[3].axhline(np.mean(loudness), color='red', ls='--', alpha=0.5, label=f'Mean: {np.mean(loudness):.1f} dB')
    axes[3].set_xlim(0, duration)
    axes[3].set_ylim(-42, -12)
    axes[3].set_xlabel('Time (seconds)')
    axes[3].set_ylabel('3s RMS Loudness (dBFS)')
    axes[3].set_title('Dynamic Range / Loudness Variation', fontsize=12, fontweight='bold')
    axes[3].legend()
    axes[3].grid(True, alpha=0.2)

    plt.tight_layout()
    chart1_path = f"{output_prefix}_analysis.png"
    plt.savefig(chart1_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {chart1_path}")

    # === CHART 2: Diagnosis ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Frequency balance summary
    band_names = ["Sub Bass", "Bass", "Low Mid", "Mid", "Upper Mid", "Presence", "Detail", "Air", "Ultra Air"]
    band_freqs = [40, 90, 185, 375, 750, 1750, 4250, 9000, 16000]
    band_levels = [freq_results.get(n, -50) for n in band_names]
    target_levels = [band_levels[1] - 4.5 * np.log2(f / 100) if f >= 100 else band_levels[0] for f in band_freqs]

    axes[0,0].semilogx(band_freqs, band_levels, 'bo-', linewidth=2.5, markersize=10, label='Your Mix')
    axes[0,0].semilogx(band_freqs, target_levels, 'r--', linewidth=2, alpha=0.7, label='Target')
    axes[0,0].axvspan(12000, 20000, alpha=0.2, color='red')
    axes[0,0].text(15000, np.mean(band_levels[-2:])-3, 'MISSING\nULTRA-AIR', 
                   ha='center', fontsize=11, color='darkred', fontweight='bold')
    axes[0,0].set_xlim(30, 20000)
    axes[0,0].set_ylim(-50, -20)
    axes[0,0].set_xlabel('Frequency (Hz)')
    axes[0,0].set_ylabel('Level (dB)')
    axes[0,0].set_title('Problem 1: Missing Ultra-High Frequencies', fontsize=11, fontweight='bold')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3, which='both')

    # 2. Dynamics by section
    if section_data:
        sec_names = [s[0] for s in section_data]
        sec_avg = [s[1] for s in section_data]
        sec_range = [s[2] for s in section_data]

        axes[0,1].bar(sec_names, sec_avg, color='steelblue', alpha=0.7)
        axes[0,1].set_ylabel('Avg Loudness (dBFS)', color='steelblue')
        axes[0,1].tick_params(axis='x', rotation=45)
        ax_twin = axes[0,1].twinx()
        ax_twin.plot(sec_names, sec_range, 'ro-', linewidth=2, markersize=8)
        ax_twin.set_ylabel('Dynamic Range (dB)', color='red')
        ax_twin.axhline(y=6, color='orange', linestyle='--', alpha=0.5)
        axes[0,1].set_title('Problem 2: Flat Dynamics in Body', fontsize=11, fontweight='bold')

    # 3. Stereo width
    if is_stereo:
        width_names = list(width_results.keys())
        width_vals = list(width_results.values())
        colors = ['green' if w < -12 else 'orange' if w < -8 else 'red' for w in width_vals]

        axes[1,0].barh(width_names, width_vals, color=colors, alpha=0.7)
        axes[1,0].axvline(x=-12, color='orange', linestyle='--', alpha=0.7)
        axes[1,0].axvline(x=-8, color='red', linestyle='--', alpha=0.7)
        axes[1,0].set_xlabel('Side vs Mid Energy (dB)')
        axes[1,0].set_title('Problem 3: Stereo Width by Band', fontsize=11, fontweight='bold')
        axes[1,0].invert_xaxis()
    else:
        axes[1,0].text(0.5, 0.5, 'Mono File', ha='center', va='center', fontsize=12)
        axes[1,0].axis('off')

    # 4. Fix recommendations
    axes[1,1].axis('off')
    fix_text = """
    QUICK FIXES

    1. RESTORE ULTRA-AIR (12-20 kHz)
       Master Bus: High Shelf +3-4 dB @ 12-16 kHz
       Use saturation to generate harmonics

    2. OPEN UP DYNAMICS
       Automate verse faders down 2-3 dB
       Gentle bus compression (1.5:1) for headroom

    3. WIDEN THE AIR
       Bright reverb with high-freq content
       Widen guitars/synths above 6 kHz

    4. ADD ROOM
       Separate reverb buses per instrument group
       Pre-delay on vocals (25-35 ms)
    """
    axes[1,1].text(0.05, 0.95, fix_text, transform=axes[1,1].transAxes, fontsize=9,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    chart2_path = f"{output_prefix}_diagnosis.png"
    plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {chart2_path}")

    return freq_results, width_results, section_data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mix_analyzer.py <audio_file.wav>")
        print("Optional: python mix_analyzer.py <file> --sr 22050")
        sys.exit(1)

    file_path = sys.argv[1]

    # Optional: custom sections
    # sections = [
    #     ("Intro", 0, 30),
    #     ("Verse", 30, 60),
    #     ("Chorus", 60, 90),
    # ]

    generate_report(file_path, target_sr=22050)
