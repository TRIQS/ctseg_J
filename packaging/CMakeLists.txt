# Configure the version of packaging files
configure_file(conda/meta.yaml.in conda/meta.yaml)
configure_file(conda/build.sh conda/build.sh COPYONLY)
configure_file(conda/conda_build_config.yaml conda/conda_build_config.yaml COPYONLY)
configure_file(TRIQS-ctseg-foss-2021b.eb.in TRIQS-ctseg-foss-2021b.eb)


