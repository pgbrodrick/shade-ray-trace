# Creation of time-of-flight shade masks from LiDAR data for NEON AOP

This set of code generates shade masks for NEON AOP data by combining sensor and solar view angles with surface models through a ray tracing algorithm. Shade masks can be generated either based on the digital surface model or from the digital terrain model combined with the canopy height model. This code was created as part of an effort to generate foliar trait maps throughout the Department of Energy (DOE) Watershed Function Scientific Focus Area (WF-SFA) site in Crested Butte, CO in association with NEON's Assignable Asset program.<br> A full description of the effort can be found at:

> K. Dana Chadwick, Philip Brodrick, Kathleen Grant, Tristan Goulden, Amanda Henderson, Nicola Falco, Haruko Wainwright, Kenneth H. Williams, Markus Bill, Ian Breckheimer, Eoin L. Brodie, Heidi Steltzer, C. F. Rick Williams, Benjamin Blonder, Jiancong Chen, Baptiste Dafflon, Joan Damerow, Matt Hancher, Aizah Khurram, Jack Lamb, Corey Lawrence, Maeve McCormick. John Musinsky, Samuel Pierce, Alexander Polussa, Maceo Hastings Porro, Andea Scott, Hans Wu Singh, Patrick O. Sorensen, Charuleka Varadharajan, Bizuayehu Whitney, Katharine Maher. Integrating airborne remote sensing and field campaigns for ecology and Earth system science. Integrating airborne remote sensing and field campaigns for ecology and earth system science. <i>In Review</i>, 2020.

and use of this code should cite that manuscript.

## Additional relevant repositories:

### Atmospheric correction wrapper: 
https://github.com/pgbrodrick/acorn_atmospheric_correction

### Shade ray tracing: 
https://github.com/pgbrodrick/shade-ray-trace

### Conifer Modeling:
https://github.com/pgbrodrick/conifer_modeling

### Trait Model Generation:
https://github.com/kdchadwick/east_river_trait_modeling

### PLSR Ensembling:
https://github.com/pgbrodrick/ensemblePLSR
