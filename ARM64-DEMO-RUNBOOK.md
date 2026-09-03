# ARM64 Machine Runbook — Sync, Test, and Record the Demo Video

This is the **checked-in prompt/guide** for the next step: running this
submission's real hardware validation *locally* on a physical Windows-on-Arm
machine (Copilot+ PC, Surface Pro X, etc.) and turning it into a recorded
demo video. Everything referenced here already exists in this repo (`Generated
Files/demo/*`) or on the live fork
(`https://github.com/yeelam-gordon/torchcodec`, branch `winarm64-hackathon`).

This does not replace the real hardware CI evidence already obtained on
GitHub's hosted `windows-11-arm` runners (see
`Generated Files/hackathon-fleet/agents/agent6/output/arm-readiness.r1.md`)
— it *adds* a second, independent, local proof point plus the recorded
artifact needed for a demo video.

## 0. What you need

- A physical Windows-on-Arm (aarch64) machine — Copilot+ PC, Surface Pro X/11,
  Dev Kit 2023, etc. — with admin rights.
- [GitHub Copilot CLI](https://docs.github.com/copilot/how-tos/set-up/install-copilot-cli)
  installed and authenticated (`copilot` on `PATH`, `gh auth login` done as
  your own GitHub account — no special repo permission needed, this is all
  read/fork-owned).
- `git`, `gh` CLI on `PATH`.
- (For the slidecast render step, run on this **or any** machine — it does
  not need to be ARM64): `pip install -r scripts/requirements.txt` inside
  the `slidecast` skill folder (edge-tts), `npm install` there too
  (Playwright+Chromium), and `ffmpeg`/`ffprobe` on `PATH`.

## 1. Sync the forked repo

```powershell
git clone https://github.com/yeelam-gordon/torchcodec.git
cd torchcodec
git checkout winarm64-hackathon
git log --oneline -5
# expect HEAD = 541f218 "Fix FFmpeg arm64 build: don't run under vcvarsall ..."
```

This branch already contains the working Arm64 CI fixes (`711ff55`,
`22993a4`, `541f218`) proven green on GitHub-hosted `windows-11-arm` runners.
Running the same builder scripts **locally on real Arm64 hardware** is the
natural next validation step — same code path, different (physical) machine.

## 2. Run the Copilot CLI prompt on the ARM64 machine

On the Arm64 machine, in the cloned `torchcodec` directory, run:

```powershell
copilot
```

Then paste this prompt (also usable non-interactively via
`copilot -p "<prompt>"` — check `copilot --help` for the current flag on
your installed version):

> You are on a real Windows-on-Arm (aarch64) machine, in a clone of
> `yeelam-gordon/torchcodec` on branch `winarm64-hackathon`. This repo adds
> native Arm64 CI builder scripts for FFmpeg and libavif
> (`packaging/build_ffmpeg_arm64.bat`, `packaging/build_libavif_arm64.bat`),
> already verified green on GitHub's hosted `windows-11-arm` runners (see
> `packaging/build_ffmpeg.sh` and `packaging/build_libavif.sh` for the
> shared, architecture-aware build logic; MSYS2's CLANGARM64 subsystem
> supplies the native aarch64 clang toolchain).
>
> Your task:
> 1. Confirm this machine is genuinely ARM64: run
>    `(Get-CimInstance Win32_Processor).Architecture` (expect `12`, ARM64)
>    and/or `$env:PROCESSOR_ARCHITECTURE` (expect `ARM64`). Record the exact
>    output.
> 2. Install MSYS2 if not already present (`choco install -y msys2 --package-parameters "/NoUpdate"`,
>    or use an existing install).
> 3. Run `packaging/build_libavif_arm64.bat` from a `cmd.exe` or PowerShell
>    shell in the repo root. Capture full console output to a log file
>    (e.g. `Tee-Object -FilePath libavif-local-run.log`).
> 4. Run `packaging/build_ffmpeg_arm64.bat` the same way, logging to
>    `ffmpeg-local-run.log`.
> 5. Report pass/fail for each, with the log tail (last ~30 lines) as
>    evidence, and confirm the expected output artifacts exist
>    (`libavif.tar.gz`, `ffmpeg.tar.gz`, or whatever the scripts produce
>    under their working directories — check the script source if unsure).
> 6. Do not modify any files unless a genuine local-only environment issue
>    (e.g. a missing MSYS2 package) requires it — if so, explain exactly
>    what and why before changing anything, and keep the change minimal.
> 7. Summarize: architecture confirmed, both builds pass/fail, artifact
>    paths, and total wall-clock time for each build (useful for the demo
>    narration's "how long does a native Arm64 build take" beat).

This closes the loop: the same fixes that passed on GitHub's hosted
`windows-11-arm` fleet are now also verified on a real physical Arm64
device you control, with a live terminal session you can record.

## 3. Record the terminal session (for the video's "live proof" shot)

Use your OS screen recorder (Xbox Game Bar `Win+G`, OBS, etc.) while running
step 2's Copilot CLI session, or re-run the two `.bat` scripts directly and
record that terminal window. Save the recording as
`Generated Files/demo/assets/arm64-live-run.mp4` (create the `assets/`
folder if it doesn't exist) — this is the raw clip the slidecast video will
embed.

## 4. Render the narrated demo video with slidecast

**UPDATE: the slidecast HTML deck, storyboard, and rendered MP4 have already
been built in this repo** — see `Generated Files/demo/deck/deck.html`,
`Generated Files/demo/deck/storyboard.json`, and the finished
`Generated Files/demo/deck/build/final.mp4` (150s, 1920x1080, H.264/AAC,
burned subtitles, 7 slides matching `narration.txt`/`shot-list.md`). This
covers everything that does **not** require the physical ARM64 machine. The
steps below remain accurate for reference/re-render, and for adding the
step-3 live-Arm64-terminal clip once it exists (currently the deck does not
yet embed that clip — see "What's left" below).

The demo *content* (script, shot list, narration, subtitles) was already
fully authored in `Generated Files/demo/`:

- `demo-script.md` — narration beats and section timing
- `shot-list.md` — the exact shot sequence (title → gap evidence → code
  diffs → **live GitHub Actions run** → **live local Arm64 terminal** →
  bug-hunt recap → closing)
- `narration.txt` — the voiceover script text
- `subtitles.srt` — pre-timed subtitle cues matching the narration
- `impact-evidence.md` — every number/claim's source, for slide content

To turn these into an actual slidecast HTML deck + narrated MP4:

```powershell
$slidecastRoot = "C:\Users\yeelam\OneDrive - Microsoft\Documents\.copilot\skills\slidecast"

# 1. Scaffold a deck package next to the demo assets (one-time):
mkdir "Generated Files\demo\deck"
Copy-Item "$slidecastRoot\templates\*" "Generated Files\demo\deck" -Recurse

# 2. Author Generated Files\demo\deck\deck.html slides from shot-list.md's
#    12 shots (title, gap-evidence, code diffs, live GH Actions run,
#    live local ARM64 run, bug-hunt recap, closing) using the blocks in
#    templates/partials/blocks.html. Embed the step-3 recording as a
#    slide's embeddedVideo clip for the "live local Arm64 run" shot.

# 3. Write Generated Files\demo\deck\storyboard.json from narration.txt +
#    subtitles.srt timing (one storyboard step per narration.txt paragraph,
#    cue.toLabel pointing at each slide's sc.cue() label).

# 4. Build:
python "$slidecastRoot\scripts\build.py" `
  --storyboard "Generated Files\demo\deck\storyboard.json" `
  --deck "Generated Files\demo\deck\deck.html" `
  --package-root "Generated Files\demo\deck" `
  --out "Generated Files\demo\deck\build"
# -> Generated Files\demo\deck\build\final.mp4
```

See the `slidecast` skill's own `SKILL.md` for the full authoring workflow
(style preset selection, `sc.*` animation helpers, storyboard schema, and
the `[[cue]]` narration-sync token) — this runbook only maps *this
project's* existing demo content onto that pipeline; it does not duplicate
slidecast's own instructions.

## 5. What "done" looks like

- ✅ `Generated Files/demo/deck/build/final.mp4` — **already rendered**: the
  finished narrated demo video (150s, 1920x1080, burned subtitles synced to
  `narration.txt`, 7 animated slides per `shot-list.md`). This part required
  no ARM64 hardware and is complete.
- ⬜ `libavif-local-run.log` and `ffmpeg-local-run.log` — still needed:
  showing pass on a real *local* Arm64 device (independent confirmation,
  complementing the already-obtained GitHub-hosted `windows-11-arm` CI
  results in `arm-readiness.r1.md`).
- ⬜ `Generated Files/demo/assets/arm64-live-run.mp4` — still needed: the raw
  recorded proof from step 3, to optionally be composited as an embedded
  clip into a future revision of the deck (e.g. replacing/extending slide
  s5's "bug hunt" beat), re-running `build.py` after adding it to
  `storyboard.json`'s `embeddedVideo` field per slide.

## What's left (only requires the physical machine)

Steps 1–3 (sync, run the Copilot CLI prompt, record the terminal session)
are the only remaining items — they need physical Windows-on-Arm hardware
that this authoring session does not have. Everything else (deck, storyboard,
rendered video) is done and checked into this repo.

## 6. PR status (companion to this runbook)

There are **two** open PRs carrying the 3 real-hardware bugfix commits:

- **Upstream** (for eventual real merge):
  https://github.com/meta-pytorch/torchcodec/pull/1673
  (from `yeelam-gordon:winarm64-hackathon` into `meta-pytorch:main`). This
  account is an external contributor here with no Triage/Write permission,
  so the `copilot-pr-autopilot` skill's review-request API call correctly
  fails with `FORBIDDEN` — a documented, accepted limitation, not a bug.
- **Fork-local** (primary target for the automated review loop):
  https://github.com/yeelam-gordon/torchcodec/pull/1
  (`main` ← `winarm64-hackathon`, entirely within the fork, where this
  account has full admin/write). The autopilot's `01-request-review.ps1`
  runs here without permission errors, but Copilot Code Review has not yet
  started reviewing as of this writing (`LatestCopilotReview: null`,
  `CopilotPending: true`) even after the mutation call succeeded — this may
  require either the UI's 🔄 button next to `copilot-pull-request-reviewer`
  in the Reviewers panel (the skill's own docs note the internal endpoint
  behind that button is not in the public API), a `synchronize` event from
  a new push, or Copilot Code Review may simply not be enabled for this
  personal/fork repo — check Settings → Code & automation → Copilot on
  https://github.com/yeelam-gordon/torchcodec/settings if the loop still
  does not start after a fresh push.

To retry: push a substantive commit (e.g. the local-Arm64-run log/results
from this runbook) to either PR's branch to fire a `synchronize` event,
then re-run
`copilot-pr-autopilot\scripts\02-check-review-status.ps1 -Owner yeelam-gordon -Repo torchcodec -PrNumber 1`
(use `pwsh`, not Windows PowerShell 5.1, to avoid a parser bug in that
script) to check convergence, and drive any resulting review threads
through the skill's steps 3–8 as a single iteration.
