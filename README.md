# path_dependent_missions

Right now this repo is mostly about the basic thermal model in `simple_heat`.
It's diagrammed below.
The idea is that we have a fuel tank that can burn or circulate fuel, and can't let the fuel get above a certain temperature.

![fuel thermal component diagram](/path_dependent_missions/simple_heat/fuel_thermal_diagram.png?raw=true)

There are three example files in the `simple_heat/examples` directory, each with slightly different heat inputs (q values) to the components.
