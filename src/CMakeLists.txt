find_package(Eigen3 3.3 REQUIRED NO_MODULE)
add_executable("dune-eigensolver" dune-eigensolver.cc)
target_link_libraries("dune-eigensolver" Eigen3::Eigen)
dune_symlink_to_source_files(FILES dune-eigensolver.ini)
