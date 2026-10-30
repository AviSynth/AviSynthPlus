Changes from 3.7.6 to 3.7.7
---------------------------

Additions, changes
~~~~~~~~~~~~~~~~~~

Build environment, Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bugfixes
~~~~~~~~
- Fix: Layer(op="Subtract") with YUVA clips: the blend weight is the original overlay alpha, not the inverted
  one. Regression since 3.7.6 (test build r4589)

Optimizations
~~~~~~~~~~~~~

Documentation
~~~~~~~~~~~~~

Please report bugs at `github AviSynthPlus page`_ - or - `Doom9's AviSynth+
forum`_

$Date: 2026/10/30 12:00:00 $

.. _github AviSynthPlus page:
    https://github.com/AviSynth/AviSynthPlus
.. _Doom9's AviSynth+ forum:
    https://forum.doom9.org/showthread.php?t=181351
