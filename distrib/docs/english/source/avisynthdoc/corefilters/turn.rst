
TurnLeft / TurnRight / Turn180
==============================

Set of filters to rotate:

* **TurnLeft** 	rotates the clip 90 degrees counterclockwise.
* **TurnRight** rotates the clip 90 degrees clockwise.
* **Turn180** 	rotates the clip 180 degrees.

Syntax and Parameters
----------------------

::

    TurnLeft (clip)
    TurnRight (clip)
    Turn180 (clip)

.. describe:: clip

    Source clip.


Changelog
----------

+------------------+-------------------------------------------------+
| Version          | Changes                                         |
+==================+=================================================+
| AviSynth+ r2728  | Fix: RGB64 Turnleft/Turnright (which are also   |
|                  | used in RGB64 Resizers).                        |
+------------------+-------------------------------------------------+
| AviSynth+ r2322  | Make Planar RGB TurnLeft, TurnRight work again. |
|                  | (2016/11/20)                                    |
+------------------+-------------------------------------------------+
| AviSynth+ <r1555 | Code from FTurn is now integrated into the core | 
|                  | (with some additional optimizations and new     |
|                  | RGB32 routines). (2013/11/24)                   |
+------------------+-------------------------------------------------+
| AviSynth 2.5.5   | ``Turn180`` added.                              |
+------------------+-------------------------------------------------+

$Date: 2022/02/04 04:52:52 $
