#!/usr/bin/env bash
# Sets up the RoboTwin 2.0 evaluation harness in third_party/RoboTwin: upstream RoboTwin at the commit used by
# FastWAM, FastWAM's harness changes (robotwin_eval.patch: configurable episode count and output directory,
# skipped observations within an action chunk, unstable seeds skipped during setup), and the LoopedWAM policy.
# Afterwards install RoboTwin's dependencies and assets as described in its README.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DEST=$ROOT/third_party/RoboTwin
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/RoboTwin-Platform/RoboTwin.git "$DEST"
  git -C "$DEST" checkout bf44be51cf5717a5595ce59447f2cf5263d2aa95
  git -C "$DEST" apply "$ROOT/experiments/robotwin/robotwin_eval.patch"
fi
ln -sfn "$ROOT/experiments/robotwin/loopedwam_policy" "$DEST/policy/loopedwam_policy"
echo "RoboTwin harness ready in $DEST"
