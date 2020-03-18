cmake_minimum_required (VERSION 3.2)

# Project name
project (qdldl_amd)

# Needed for compilation to succeed
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/out)

add_subdirectory(qdldl EXCLUDE_FROM_ALL)

set(
    amd_src
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/include/amd_internal.h
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/include/amd.h
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/include/SuiteSparse_config.h
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/include/perm.h
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_1.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_aat.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_control.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_defaults.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_info.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_order.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_post_tree.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_postorder.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_preprocess.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_valid.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/SuiteSparse_config.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/perm.c
)


# Add QDLDL+AMD library
add_library(qdldlamd STATIC ${amd_src} $<TARGET_OBJECTS:qdldlobject>)
target_include_directories(qdldlamd PRIVATE
	                   ${CMAKE_CURRENT_SOURCE_DIR}/amd/include
	                   ${CMAKE_CURRENT_SOURCE_DIR}/qdldl/include
			    )
