Documentation for the Spectrum AWG Labscript module
===================================================

This labscript device controls AWGs made by Spectrum Instrumentation https://spectrum-instrumentation.com/products/families/66xx_m4i_pci.php

Change log
----------

* v2.0.0 (2023-11-10)

   * Renamed :meth:`SpectrumAwgOut.loop` to :meth:`SpectrumAwgOut.set_wave_and_enable` and :meth:`SpectrumAwgOut.wait` to :meth:`SpectrumAwgOut.disable` to be more consistent with the naming conventions of labscript.

* v1.0.1 (2023-07-24)

   * Improved front-panel.

   * Fixed bug where the tab would crash when not commanded to do anything.


* v1.0.0 (2023-05-12)

   * Initial upload.

Labscript API
-------------

.. automodule:: __init__
   :members:
   :undoc-members:
   :show-inheritance: