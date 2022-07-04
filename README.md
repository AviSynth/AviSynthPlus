AviSynth+
=========

**AviSynth+** is an improved version of the [AviSynth frameserver](http://avisynth.nl/index.php/Main_Page), with improved
features and developer friendliness.

Visit our [forum thread](http://forum.doom9.org/showthread.php?t=181351) for compilation instructions and support.

Building the documentation:
---------------------------
(Note: the bundled documentation lags behind the descriptions found in the wiki.
You can always check the online documentation at http://avisynth.nl/index.php/Main_Page)

AviSynth+'s documentation can be generated into HTML by using Sphinx.

### Set-up:

Make sure that Sphinx is installed. This requires that Python is already
installed and the pip tool is available.  Sphinx 1.3 is the recommended
version.

>pip install sphinx

For various Linux distributions, a Sphinx package should be available
in the distro's repositories.  Often under the name 'python-sphinx'
(as it is in Ubuntu's repositories).

There is currently a fallback so that distros that only provide
Sphinx 1.2 can still build the documentation.  It will look
different than when built with Sphinx 1.3, because the theme
used with Sphinx 1.3 (bizstyle) had not yet been added to the main
Sphinx package.

### Building the documentation

Once Sphinx is installed, we can build the documentation.

> cd distrib/docs/english
> <br>make html


Headers for applications which dynamically load AviSynth+:
----------------------------------------------------------

The expected use-case of AviSynth+ is as a dynamically
loaded library (using LoadLibrary on Windows or dlopen
everywhere else).

Due to this, it's not actually necessary to build the
AviSynth+ library itself in order for applications using
it this way to find it.

To facilitate this, we support using CMake to do a
limited, headers-only install.  The GNUmakefile is
deprecated and will eventually be removed.

### Using CMake:

#### To install:

> mkdir avisynth-build && cd avisynth-build
> <br>cmake ../ -DHEADERS_ONLY:bool=on
> <br>make VersionGen install

`-DCMAKE_INSTALL_PREFIX` can be used to override the
install location if need be.

#### To uninstall:

>make uninstall

### Using GNUmakefile (legacy):

#### To install:

>make install

#### To install to a non-standard location:

>make install PREFIX=/path/to/location

#### To uninstall:

>make uninstall PREFIX=/path/to/location
