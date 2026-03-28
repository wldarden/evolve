add_library(evolve STATIC
    src/genome.cpp
    src/population.cpp
    src/innovation.cpp
    src/structured_genome.cpp
)
target_include_directories(evolve PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(evolve PRIVATE project_warnings)

add_subdirectory(tests)
