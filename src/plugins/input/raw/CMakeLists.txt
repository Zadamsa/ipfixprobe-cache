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

add_library(gpu_parser STATIC  ../parser/gpu_parser.cu)
set_target_properties(gpu_parser PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gpu_parser PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(gpu_parser PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G -g>)
message(STATUS "CUDA Compile Options: ${CMAKE_CUDA_FLAGS}")
target_link_libraries(ipfixprobe-input-raw PRIVATE
	telemetry::telemetry
	#gpu_parser
	gpu_parser
)

target_include_directories(gpu_parser PRIVATE
		${CMAKE_SOURCE_DIR}/include/
		${CMAKE_SOURCE_DIR}/src/plugins/input/parser
)

install(TARGETS gpu_parser
	LIBRARY DESTINATION "${INSTALL_DIR_LIB}/ipfixprobe/input/"
)

install(TARGETS ipfixprobe-input-raw
	LIBRARY DESTINATION "${INSTALL_DIR_LIB}/ipfixprobe/input/"
)
