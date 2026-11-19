# Equivs control file for building the build dependencies meta-package
#
# Copyright 2026 林博仁(Buo-ren Lin) <buo.ren.lin@gmail.com>
# SPDX-License-Identifier: CC-BY-SA-3.0
Source: avisynthplus
Section: misc
Priority: optional
Homepage: https://github.com/AviSynth/AviSynthPlus
Standards-Version: 3.9.2

Package: avisynthplus-build-deps
Depends:
 cmake,
 git,
 g++,
 libdevil-dev,
 libsoundtouch-dev,
 pkg-config,
 python3-sphinx
Description: Meta-package for the build dependencies of AviSynth+
 This allows clean removal of the build dependency packages after they
 are no longer used.
