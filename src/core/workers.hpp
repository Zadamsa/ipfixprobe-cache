/**
 * \file workers.hpp
 * \brief Exporter worker procedures
 * \author Jiri Havranek <havranek@cesnet.cz>
 * \date 2021
 */
/*
 * Copyright (C) 2021 CESNET
 *
 * LICENSE TERMS
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 * 3. Neither the name of the Company nor the names of its contributors
 *    may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 *
 *
 */

#ifndef IPXP_WORKERS_HPP
#define IPXP_WORKERS_HPP

#include "stats.hpp"

#include <atomic>
#include <future>

#include <ipfixprobe/inputPlugin.hpp>
#include <ipfixprobe/outputPlugin.hpp>
#include <ipfixprobe/packet.hpp>
#include <ipfixprobe/processPlugin.hpp>
#include <ipfixprobe/ring.h>
#include <ipfixprobe/storagePlugin.hpp>

namespace ipxp {

#define MICRO_SEC 1000000L

struct WorkerResult {
	bool error;
	std::string msg;
};

struct WorkPipeline {
	struct {
		std::shared_ptr<InputPlugin> inputPlugin;
		std::thread* thread;
		std::promise<WorkerResult>* promise;
		std::atomic<InputStats>* stats;
	} input;
	struct {
		std::shared_ptr<StoragePlugin> storagePlugin;
		std::vector<ProcessPlugin*> plugins;
	} storage;
};

struct OutputWorker {
	std::shared_ptr<OutputPlugin> outputPlugin;
	std::thread* thread;
	std::promise<WorkerResult>* promise;
	std::atomic<OutputStats>* stats;
	ipx_ring_t* queue;
};

void input_storage_worker(
	std::shared_ptr<InputPlugin> inputPlugin,
	std::shared_ptr<StoragePlugin> storagePlugin,
	size_t queue_size,
	uint64_t pkt_limit,
	std::promise<WorkerResult>* out,
	std::atomic<InputStats>* out_stats);
void output_worker(
	std::shared_ptr<OutputPlugin> outputPlugin,
	ipx_ring_t* queue,
	std::promise<WorkerResult>* out,
	std::atomic<OutputStats>* out_stats,
	uint32_t fps);

} // namespace ipxp

#endif /* IPXP_WORKERS_HPP */
