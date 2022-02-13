
ConvertToMono
=============

Converts a multichannel audio signal to mono by averaging all channels with 
equal weights. If the signal is already in mono, it is returned untouched. 


Syntax and Parameters
----------------------

::

    ConvertToMono (clip)

.. describe:: clip

    | Source clip. Supported audio sample types: 16-bit integer and 32-bit float. 
    | Other sample types (8-, 24- and 32-bit integer) are automatically 
      :doc:`converted <convertaudio>` to 32-bit float. 


$Date: 2022/02/05 22:44:06 $
