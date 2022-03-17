==========
SwapFields
==========

The **SwapFields** filter swaps image line 0 with line 1, line 2 with line 3,
and so on, thus effectively swapping the two fields in an interlaced frame.
It's the same as :ref:`SeparateFields() <SeparateFields>`.\
:ref:`ComplementParity() <ComplementParity>`.\ :ref:`Weave() <weave>` (and it's
implemented that way).


Syntax and Parameters
----------------------

::

    SwapFields (clip)

.. describe:: clip

    Source clip; all color formats supported.


$Date: 2022/03/17 22:07:09 $
