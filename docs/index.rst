##########
Flood Tool
##########

This package implements a flood risk prediction and visualization tool.

Installation Instructions
-------------------------

The flood tool package can be downloaded and installed as follows:

First the relevant github repoistry should be cloned:

``git glone https://github.com/ese-msc-2021/ads-deluge-ouse.git``

After moving to the relevant directory associated with the repository the following command can be run to install the flood tool package:

``python setup.py install --user``


Usage guide
-----------


The flood tool package can be used for both calcuation and visualisation of the annual flood risk for a set of user defined locations specified by postcode. 

``import flood_tool``

``tool = Tool()``

``risk_labels=tool.get_flood_class(self, postcodes, method=1, update=False)``

This generates a series predicting flood probability classification for a collection of postcodes.
the method argument (shown above) choses the  classification alogorithm with which to train flood risk data with the following conversion: KNN: method=1, RandomForest: method=2, Neural Network : method=3)

``annual_flood_risk= tool.get_annual_flood_risk(postcodes,risk_labels)``

where the user can define postcodes to be a list of postcodes they wish to find corresponding data for.
This generates a series of total annual flood risk estimates indexed by locations based on the following formula:

``annual flood risk= 0.05x(total postcode property value)x(postcode flood probability)``

The visualisations can be viewed by viewing the DataVisualisation.ipynd notebook which, when run shows maps visualising annual flood risk of postcodes.

Geodetic Transformations
------------------------

For historical reasons, multiple coordinate systems exist in in current use in
British mapping circles. The Ordnance Survey has been mapping the British Isles
since the 18th Century and the last major retriangulation from 1936-1962 produced
the Ordance Survey National Grid (otherwise known as **OSGB36**), which defined
latitude and longitude for all points across the island of Great Britain [1]_.
For convenience, a standard Transverse Mercator projection [2]_ was also defined,
producing a notionally flat 2D gridded surface, with gradations called eastings
and northings. The scale for these gradations was identified with metres, which
allowed local distances to be defined with a fair degree of accuracy.


The OSGB36 datum is based on the Airy Ellipsoid of 1830, which defines
semimajor axes for its model of the earth, :math:`a` and :math:`b`, a scaling
factor :math:`F_0` and ellipsoid height, :math:`H`.

.. math::
    a &= 6377563.396, \\
    b &= 6356256.910, \\
    F_0 &= 0.9996012717, \\
    H &= 24.7.

The point of origin for the transverse Mercator projection is defined in the
Ordnance Survey longitude-latitude and easting-northing coordinates as

.. math::
    \phi^{OS}_0 &= 49^\circ \mbox{ north}, \\
    \lambda^{OS}_0 &= 2^\circ \mbox{ west}, \\
    E^{OS}_0 &= 400000 m, \\
    N^{OS}_0 &= -100000 m.

More recently, the world has gravitated towards the use of satellite based GPS
equipment, which uses the (globally more appropriate) World Geodetic System
1984 (also known as **WGS84**). This datum uses a different ellipsoid, which offers a
better fit for a global coordinate system (as well as North America). Its key
properties are:

.. math::
    a_{WGS} &= 6378137,, \\
    b_{WGS} &= 6356752.314, \\
    F_0 &= 0.9996.

For a given point on the WGS84 ellipsoid, an approximate mapping to the
OSGB36 datum can be found using a Helmert transformation [3]_,

.. math::
    \mathbf{x}^{OS} = \mathbf{t}+\mathbf{M}\mathbf{x}^{WGS}.


Here :math:`\mathbf{x}` denotes a coordinate in Cartesian space (i.e in 3D)
as given by the (invertible) transformation

.. math::
    \nu &= \frac{aF_0}{\sqrt{1-e^2\sin^2(\phi^{OS})}} \\
    x &= (\nu+H) \sin(\lambda)\cos(\phi) \\
    y &= (\nu+H) \cos(\lambda)\cos(\phi) \\
    z &= ((1-e^2)\nu+H)\sin(\phi)

and the transformation parameters are

.. math::
    :nowrap:

    \begin{eqnarray*}
    \mathbf{t} &= \left(\begin{array}{c}
    -446.448\\ 125.157\\ -542.060
    \end{array}\right),\\
    \mathbf{M} &= \left[\begin{array}{ c c c }
    1+s& -r_3& r_2\\
    r_3 & 1+s & -r_1 \\
    -r_2 & r_1 & 1+s
    \end{array}\right], \\
    s &= 20.4894\times 10^{-6}, \\
    \mathbf{r} &= [0.1502'', 0.2470'', 0.8421''].
    \end{eqnarray*}

Given a latitude, :math:`\phi^{OS}` and longitude, :math:`\lambda^{OS}` in the
OSGB36 datum, easting and northing coordinates, :math:`E^{OS}` & :math:`N^{OS}`
can then be calculated using the following formulae (see "A guide to coordinate
systems in Great Britain, Appendix C1):

.. math::
    \rho &= \frac{aF_0(1-e^2)}{\left(1-e^2\sin^2(\phi^{OS})\right)^{\frac{3}{2}}} \\
    \eta &= \sqrt{\frac{\nu}{\rho}-1} \\
    M &= bF_0\left[\left(1+n+\frac{5}{4}n^2+\frac{5}{4}n^3\right)(\phi^{OS}-\phi^{OS}_0)\right. \\
    &\quad-\left(3n+3n^2+\frac{21}{8}n^3\right)\sin(\phi-\phi_0)\cos(\phi^{OS}+\phi^{OS}_0) \\
    &\quad+\left(\frac{15}{8}n^2+\frac{15}{8}n^3\right)\sin(2(\phi^{OS}-\phi^{OS}_0))\cos(2(\phi^{OS}+\phi^{OS}_0)) \\
    &\left.\quad-\frac{35}{24}n^3\sin(3(\phi-\phi_0))\cos(3(\phi^{OS}+\phi^{OS}_0))\right] \\
    I &= M + N^{OS}_0 \\
    II &= \frac{\nu}{2}\sin(\phi^{OS})\cos(\phi^{OS}) \\
    III &= \frac{\nu}{24}\sin(\phi^{OS})cos^3(\phi^{OS})(5-\tan^2(phi^{OS})+9\eta^2) \\
    IIIA &= \frac{\nu}{720}\sin(\phi^{OS})cos^5(\phi^{OS})(61-58\tan^2(\phi^{OS})+\tan^4(\phi^{OS})) \\
    IV &= \nu\cos(\phi^{OS}) \\
    V &= \frac{\nu}{6}\cos^3(\phi^{OS})\left(\frac{\nu}{\rho}-\tan^2(\phi^{OS})\right) \\
    VI &= \frac{\nu}{120}\cos^5(\phi^{OS})(5-18\tan^2(\phi^{OS})+\tan^4(\phi^{OS}) \\
    &\quad+14\eta^2-58\tan^2(\phi^{OS})\eta^2) \\
    E^{OS} &= E^{OS}_0+IV(\lambda^{OS}-\lambda^{OS}_0)+V(\lambda-\lambda^{OS}_0)^3+VI(\lambda^{OS}-\lambda^{OS}_0)^5 \\
    N^{OS} &= I + II(\lambda^{OS}-\lambda^{OS}_0)^2+III(\lambda-\lambda^{OS}_0)^4+IIIA(\lambda^{OS}-\lambda^{OS}_0)^6

The inverse transformation can be generated iteratively using a fixed point process:

1. Set :math:`M=0` and :math:`\phi^{OS} = \phi_0^{OS}`.
2. Update :math:`\phi_{i+1}^{OS} = \frac{N-N_0-M}{aF_0}+\phi_i^{OS}`
3. Calculate :math:`M` using the formula above.
4. If :math:`\textrm{abs}(N-N_0-M)> 0.01 mm` go to 2, otherwise halt.

With :math:`M` calculated we now improve our estimate of :math:`\phi^{OS}`. First calculate
:math:`\nu`, :math:`\rho` and :math:`\eta` using our previous formulae. Next

.. math::

    VII &= \frac{\tan(\phi^{OS})}{2\rho\nu},\\
    VIII &= \frac{\tan(\phi^{OS})}{24\rho\nu^3}\left(5+3\tan^2(\phi^{OS})+\eta^2-9\tan^2(\phi^{OS})\eta^2\right),\\
    IX &= \frac{\tan(\phi^{OS})}{720\rho\nu^5}\left(61+90\tan^2(\phi^{OS})+45\tan^4(\phi^{OS})\right),\\
    X &= \frac{\sec\phi^{OS}}{\nu}, \\
    XI &= \frac{\sec\phi^{OS}}{6\nu^3}\left(\frac{\nu}{\rho}+2\tan^2(\phi^{OS})\right), \\
    XII &= \frac{\sec\phi^{OS}}{120\nu^5}\left(5+28\tan^2(\phi^{OS})+24\tan^4(\phi^{OS})\right), \\
    XIIA &= \frac{\sec\phi^{OS}}{5040\nu^5}\left(61+662\tan^2(\phi^{OS})+1320\tan^4(\phi^{OS})+720\tan^6(\phi^{OS})\right).

Finally, the corrected values for :math:`\phi^{OS}` and :math:`\lambda^{OS}` are:

.. math::
    \phi_{\textrm{final}}^{OS} &= \phi^{OS} -VII(E-E_0)^2 +VIII(E-E_0)^4 -IX(E-E_0)^6, \\
    \lambda_{\textrm{final}}^{OS} &= \lambda_0^{OS}+X(E-E_0)-XI(E-E_0)^3+ XII(E-E_0)^5-XII(E-E_0)^7.




Function APIs
-------------

.. automodule:: flood_tool
  :members:
  :imported-members:


.. rubric:: References

.. [1] A guide to coordinate systems in Great Britain, Ordnance Survey
.. [2] Map projections - A Working Manual, John P. Snyder, https://doi.org/10.3133/pp1395
.. [3] Computing Helmert transformations, G Watson, http://www.maths.dundee.ac.uk/gawatson/helmertrev.pdf
