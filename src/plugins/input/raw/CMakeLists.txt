project(ipfixprobe-input-raw VERSION 1.0.0 DESCRIPTION "ipfixprobe-input-raw plugin")

add_library(ipfixprobe-input-raw MODULE
	src/raw.cpp
	src/raw.hpp
	../parser/parser.cuh
)

set_target_properties(ipfixprobe-input-raw PROPERTIES
	CXX_VISIBILITY_PRESET hidden
	VISIBILITY_INLINES_HIDDEN YES
)

target_include_directories(ipfixprobe-input-raw PRIVATE
	${CMAKE_SOURCE_DIR}/include/
	${CMAKE_SOURCE_DIR}/src/plugins/input/parser
)

add_library(gpu_parser STATIC ../parser/gpu_parser.cu ../parser/parser.cu)
set_target_properties(gpu_parser PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gpu_parser PROPERTIES POSITION_INDEPENDENT_CODE ON)


target_link_libraries(ipfixprobe-input-raw PRIVATE
	telemetry::telemetry
	gpu_parser
)

target_include_directories(gpu_parser PRIVATE
		${CMAKE_SOURCE_DIR}/include/
		${CMAKE_SOURCE_DIR}/src/plugins/input/parser
)

install(TARGETS ipfixprobe-input-raw
	LIBRARY DESTINATION "${INSTALL_DIR_LIB}/ipfixprobe/input/"
)
