==========
TCPDeliver
==========

**TCPDeliver** includes the `TCPServer`_ and `TCPSource`_ filters that enable
you to send and receive clips over your network. You can connect several clients
to the same machine.

.. note::

    AviSynth+ does not include the TCPDeliver plugin in the installation.
    Download the `updated TCPDeliver`_, which supports all planar RGB(A) and \
    YUV(A) color formats.


TCPServer
---------

**TCPServer** spawns a server thread on the current machine running on the
specified port. You will get output in the application you open your script in,
but the server will only be running as long as the application (VDub for instance)
is open.

Example::

    ColorBars(512, 256)
    TCPServer()

This will start a server.

.. rubric:: Syntax and Parameters

::

    TCPServer (clip, int "port")

.. describe:: clip

    Source clip; all color formats supported.

.. describe:: port

    Port default is 22050.


TCPSource
---------

**TCPSource** connects to the machine with the given address (IP-number for
instance) to a server running on the given port.

Example::

    TCPSource("127.0.0.1")
    Info()

This will connect to the local machine, if a server is running. "127.0.0.1" is
the localhost, or "local loopback".

.. rubric:: Syntax and Parameters

::

    TCPSource (string hostname, int "port", string "compression")

.. describe:: hostname

    ``hostname`` can be a computer name or an IP address.

.. describe:: port

    Port default is 22050.

.. describe:: compression

    Choose the compression used for the video:

    +------------------+----------------------------------------------------------------------------+
    | Compression Type | Description                                                                |
    +==================+============================================================================+
    | None             | Use no compression. Fastest option - video will not be compressed before   |
    |                  | being sent over the network.                                               |
    +------------------+----------------------------------------------------------------------------+
    | LZO              | Use `LZO`_ dictionary compression. Fairly fast, but only compresses well   |
    |                  | on artificial sources, like cartoons and anime with very uniform surfaces. |
    +------------------+----------------------------------------------------------------------------+
    | Huffman          | Uses a fairly slow Huffman routine by `Marcus Geelnard`_. Compresses       |
    |                  | natural video better than LZO.                                             |
    +------------------+----------------------------------------------------------------------------+
    | GZip             | Uses a `Gzip`_ Huffman only compression. Works much like Huffman           |
    |                  | setting, but seems faster.                                                 |
    +------------------+----------------------------------------------------------------------------+
    | RLE              | RLE (`Run Length Encoding`_) is the simplest possible lossless compression |
    |                  | method by `Marcus Geelnard`_. The compression is particularly well suited  |
    |                  | for palette-based bitmapped images.                                        |
    +------------------+----------------------------------------------------------------------------+

    If no compression is given, GZip is currently used by default. Interlaced
    material compresses worse than non-interlaced due to downwards delta-encoding.
    If network speed is a problem you might want to use :doc:`SeparateFields <separatefields>`.


Examples
--------

You can use this to run each/some filters on different PC's. For example::

    #Clustermember 1:
    AVISource()
    Deinterlacer()
    TCPServer()

    # Clustermember 2:
    TCPSource()
    Sharpener()
    TCPServer()

    # Clustermember 3:
    TCPSource()
    # client app -> video codec -> final file

See the Doom9 thread: "`Can't get TCPServer() and TCPSource() to work`_" for
more information.


Usability Notes
---------------

Once you have added a TCPServer, you cannot add more filters to the chain, or
use the output from the filter. The server runs in a separate thread, but
since AviSynth isn't completely thread-safe you cannot reliably run multiple
servers. This should **not** be used:

::

    AviSource("avi.avi")
    TCPServer(1001)
    TCPServer(1002) # This is NOT a good idea

So the basic rule is **never more than one TCPServer per script**.

Using commands after TCPServer is also a bad idea:

::

    AviSource("avi.avi")
    TCPServer(1001)
    AviSource("avi2.avi") # Do not do this, this will disable the server.

AviSynth detects that the output of TCPServer isn't used, so it kills the
Server filter. **TCPServer should always be the last filter.**


Changelog
---------

TCPDeliver is based from the AviSynth 2.6 source.

+---------+-------------------------------------------------------------------+
| Version | Changes                                                           |
+=========+===================================================================+
| v0.2    || Support for YUVA/PlanarRGBA colorspaces;                         |
|         || HBD formats should be handled better if compression is involved; |
|         || Dropped old garbage from source code;                            |
|         || No need in the old VS runtime.                                   |
+---------+-------------------------------------------------------------------+
| v0.1    | Initial release.                                                  |
+---------+-------------------------------------------------------------------+

$Date: 2022/03/26 14:51:17 $

.. _updated TCPDeliver:
    https://github.com/DJATOMs-archive/TCPDeliver/releases
.. _LZO:
    http://www.oberhumer.com/opensource/lzo/
.. _Marcus Geelnard:
    https://web.archive.org/web/20181025055443/http://bcl.comli.eu/
.. _Gzip:
    http://www.gzip.org/
.. _Run Length Encoding:
    https://en.wikipedia.org/wiki/Run-length_encoding
.. _Can't get TCPServer() and TCPSource() to work:
    https://forum.doom9.org/showthread.php?t=174478
