project(ipfixprobe-storage-cache VERSION 1.0.0 DESCRIPTION "ipfixprobe-storage-cache plugin")

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/)

add_library(ipfixprobe-storage-cache MODULE
	src/cache.hpp
	src/cache.cpp
	src/cacheOptParser.cpp
	src/cacheOptParser.hpp
	src/cacheRowSpan.hpp
	src/cacheRowSpan.cpp
	src/flowRecord.hpp
	src/flowRecord.cpp
	src/flowKeyFactory.hpp
	src/flowKey.tpp
	src/cacheStats.hpp
	src/fragmentationCache/fragmentationCache.cpp
	src/fragmentationCache/fragmentationCache.hpp
	src/fragmentationCache/fragmentationKeyData.hpp
	src/fragmentationCache/fragmentationTable.cpp
	src/fragmentationCache/fragmentationTable.hpp
	src/fragmentationCache/ringBuffer.hpp
	src/fragmentationCache/timevalUtils.hpp
	src/xxhash.c
	src/xxhash.h
)

if (ENABLE_CTT)
	add_definitions(-DWITH_CTT=1)
	target_sources(ipfixprobe-storage-cache PRIVATE 
		src/ctt/cttController.cpp
		src/ctt/cttController.hpp
	)
endif()

set_target_properties(ipfixprobe-storage-cache PROPERTIES
	CXX_VISIBILITY_PRESET hidden
	VISIBILITY_INLINES_HIDDEN YES
)

target_include_directories(ipfixprobe-storage-cache PRIVATE
	${CMAKE_SOURCE_DIR}/include/
)

target_link_libraries(ipfixprobe-storage-cache PRIVATE
	telemetry::telemetry
)

install(TARGETS ipfixprobe-storage-cache
	LIBRARY DESTINATION "${INSTALL_DIR_LIB}/ipfixprobe/storage/"
)
