
WriteFile
=========

**WriteFile** and related functions evaluate each `expression`_, convert the
result to a string and put the concatenated results into a file, followed by a
newline.

* **WriteFile** evaluates the *expressions* and generates output for each frame
  rendered by the filter.

* **WriteFileIf** is similar, but generates output only if the first expression
  evaluates to true.

  * In both cases, there is no output at script opening or closure.
  * Note that with :ref:`ScriptClip`, script opening and closure occurs on every
    frame.
  * Note that since output is produced only for rendered frames, there will be
    no output at all if the result of the filter is |not used| in deriving the
    final result of the script.

* **WriteFileStart** and **WriteFileEnd** generate output only on script opening
  and closure respectively.

  * There is no action on each frame, unless used within :ref:`ScriptClip`.
  * The expressions are evaluated exactly once, at the location of the filter
    in the script.


Syntax and Parameters
----------------------

::

    WriteFile (clip, string filename, string expression1 [, string expression2 [, ...]], bool "append", bool "flush")

    WriteFileIf (clip, string filename, string expression1 [, string expression2 [, ...]], bool "append", bool "flush")

    WriteFileStart (clip, string filename, string expression1 [, string expression2 [, ...]], bool "append")

    WriteFileEnd (clip, string filename, string expression1 [, string expression2 [, ...]], bool "append"*)

.. describe:: clip

    Source clip.

.. describe:: filename

    Path and filename of the file to be saved. If path is omitted the file will
    be saved in the same location as the script.

.. describe:: expression1, expression2, ...

    Specify the expressions to use. Usage is best explained in the `Examples`_
    section.

    The |runtime| variable *current_frame* is set so that you can use it in
    expressions (as with :ref:`ScriptClip`).

    * *current_frame* is set to -1 when the script is loaded and to -2 when
      the script is closed.

.. describe:: append

    When ``append=true``, the results will be appended to any existing file
    filename; if false, a new file is created and the old one is overwritten.

    * Only script opening and closure are affected; **WriteFile** and
      **WriteFileIf** per-frame execution always append.
    * The default for ``append`` is *true*, except for **WriteFileStart**,
      where it is *false*.



.. describe:: flush

    When ``flush=true``, after each operation a flush is performed: any
    unwritten data is written to disk, and the file is closed and reopened.

    * After flushing you may read the updated file immediately, either through
      :doc:`conditionalreader` or an external application.
    * The default for ``flush`` is *true* for **WriteFile** and **WriteFileIf**
      and always *true* (no user option) for **WriteFileStart** and
      **WriteFileEnd**.
    * Note that flushing after every frame may be significantly slower.


Examples
--------

Usage is best explained with some simple examples:

::

    filename = "c:\myprojects\output.txt"
    # create a test video to get frames
    Version()

    # the expression here is only a variable, which is evaluated and put in the file
    # you will get a file with the framenumber in each line
    WriteFile(filename, "current_frame")

    # this line is written when the script is opened
    WriteFileStart(filename, """ "This is the header" """)

    # and this when the script is closed
    WriteFileEnd(filename, """ "Now the script was closed" """)

Look how you can use triple-quotes to type a string in a string!

If the expression cannot be evaluated, the error message is written instead.
In case this happens with the If-expression in **WriteFileIf** the result is
assumed to be true.

::

    # will result in "I don't know what "this" means"
    WriteFile(filename, "this is nonsense")

--------


There are easier ways to write numbers in a file, BUT ... with this example you
can see how to use the *runtime function* |AverageLuma|:

::

    # create a test video to get different frames
    Version().FadeIn(50).ConvertToYV12()

    # this will print the frame number, a ":" and the average luma for that frame
    colon = ": "
    WriteFile("C:\text.log", "current_frame", "colon", "AverageLuma")

Or maybe you want the actual time printed too:

::

    # create a test video to get different frames
    Version().FadeIn(50).ConvertToYV12()

    # this will print the frame number, the current time and the average luma for that frame
    # the triple quotes are necessary to put quotes inside a string
    WriteFile(last, "text.log", "current_frame", """ time(" %H:%M:%S") """, "AverageLuma")

--------

**More examples**

In **WriteFileIf** the FIRST expression is expected to be boolean (true or
false). Only if it is TRUE the other expressions are evaluated and the line
is printed. (Remember: && is AND, || is OR, == is EQUAL, != is NOT EQUAL)
That way you can omit lines completely from your file.

::

    # create a test video to get different frames
    Version().FadeIn(50).ConvertToYV12()

    # this will print the frame number, but only of frames where AverageLuma is between 30 and 60
    WriteFileIf(last, "text.log", "(AverageLuma>30) && (AverageLuma<60)", "current_frame", """ ":" """, "AverageLuma")


Changelog
---------
+----------------+------------------------------------------------------------+
| Version        | Changes                                                    |
+================+============================================================+
| AviSynth 2.6.0 | Number of expressions changed from 16 to nearly unlimited. |
+----------------+------------------------------------------------------------+
| AviSynth 2.5.5 | Initial release.                                           |
+----------------+------------------------------------------------------------+

$Date: 2022/02/24 16:01:28 $

.. _expression:
    http://avisynth.nl/index.php/Grammar

.. |not used| replace:: :doc:`not used <../script_ref/script_ref_execution_model>`
.. |runtime| replace:: :ref:`runtime <Special runtime variables and functions>`
.. |AverageLuma| replace:: :doc:`AverageLuma <../syntax/syntax_internal_functions_runtime>`
