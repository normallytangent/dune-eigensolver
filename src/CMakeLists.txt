add_executable("dune-eigensolver" dune-eigensolver.cc)
target_link_dune_default_libraries("dune-eigensolver")
dune_symlink_to_source_files(FILES dune-eigensolver.ini)
