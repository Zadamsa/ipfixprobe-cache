/**
 * @file
 * @brief Pcap reader based on libpcap
 * @author Jiri Havranek <havranek@cesnet.cz>
 * @author Pavel Siska <siska@cesnet.cz>
 * @date 2025
 *
 * Copyright (c) 2025 CESNET
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "pcap.hpp"

#include "parser.hpp"

#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>

#include <ipfixprobe/pluginFactory/pluginManifest.hpp>
#include <ipfixprobe/pluginFactory/pluginRegistrar.hpp>
#include <pcap/pcap.h>

namespace ipxp {

// Read only 1 packet into packet block
constexpr size_t PCAP_PACKET_BLOCK_SIZE = 1;

struct UserData {
	parser_opt_t* opt;
	ParserStats& stats;
};

static const PluginManifest pcapPluginManifest = {
	.name = "pcap",
	.description = "Pcap input plugin for reading packets from network interface or pcap file.",
	.pluginVersion = "1.0.0",
	.apiVersion = "1.0.0",
	.usage =
		[]() {
			PcapOptParser parser;
			parser.usage(std::cout);
		},
};

/**
 * \brief Parsing callback function for pcap_dispatch() call. Parse packets up to transport layer.
 * \param [in,out] arg Serves for passing pointer to Packet user data structure into callback
 * function. \param [in] h Contains timestamp and packet size. \param [in] data Pointer to the
 * captured packet data.
 */
void packet_handler(u_char* arg, const struct pcap_pkthdr* h, const u_char* data)
{
	UserData* user_data = reinterpret_cast<UserData*>(arg);

#ifdef __CYGWIN__
	// WinPcap, uses Microsoft's definition of struct timeval, which has `long` data type
	// used for both tv_sec and tv_usec and has 32 bit even on 64 bit platform.
	// Cygwin uses 64 bit tv_sec and tv_usec, thus a little reinterpretation of bytes needs to be
	// used.
	struct pcap_pkthdr new_h;
	new_h.ts.tv_sec = *reinterpret_cast<const uint32_t*>(h);
	new_h.ts.tv_usec = *(reinterpret_cast<const uint32_t*>(h) + 1);
	new_h.caplen = *(reinterpret_cast<const uint32_t*>(h) + 2);
	new_h.len = *(reinterpret_cast<const uint32_t*>(h) + 3);
	parse_packet(user_data->opt, user_data->stats, new_h.ts, data, new_h.len, new_h.caplen);
#else
	parse_packet(user_data->opt, user_data->stats, h->ts, data, h->len, h->caplen);
#endif
}

PcapReader::PcapReader(const std::string& params)
	: m_handle(nullptr)
	, m_snaplen(-1)
	, m_datalink(0)
	, m_live(false)
	, m_netmask(PCAP_NETMASK_UNKNOWN)
{
	init(params.c_str());
}

PcapReader::~PcapReader()
{
	close();
}

void PcapReader::init(const char* params)
{
	PcapOptParser parser;
	try {
		parser.parse(params);
	} catch (ParserError& e) {
		throw PluginError(e.what());
	}

	if (parser.m_list) {
		print_available_ifcs();
		throw PluginExit();
	}

	if (parser.m_ifc.empty() && parser.m_file.empty()) {
		throw PluginError("specify network interface or pcap file path");
	}
	if (!parser.m_ifc.empty() && !parser.m_file.empty()) {
		throw PluginError("only one input can be specified");
	}

	m_snaplen = parser.m_snaplen;
	if (m_snaplen < MIN_SNAPLEN) {
		std::cerr << "setting snapshot length to minimum value " << MIN_SNAPLEN << std::endl;
		m_snaplen = MIN_SNAPLEN;
	}

	if (!parser.m_ifc.empty()) {
		open_ifc(parser.m_ifc);
	} else {
		open_file(parser.m_file);
	}

	if (!parser.m_filter.empty()) {
		set_filter(parser.m_filter);
	}
}

void PcapReader::close()
{
	if (m_handle != nullptr) {
		pcap_close(m_handle);
		m_handle = nullptr;
	}
}

void PcapReader::open_file(const std::string& file)
{
	char errbuf[PCAP_ERRBUF_SIZE];

	m_handle = pcap_open_offline(file.c_str(), errbuf);
	if (m_handle == nullptr) {
		throw PluginError(std::string("unable to open file: ") + errbuf);
	}

	m_datalink = pcap_datalink(m_handle);
	m_live = false;

	check_datalink(m_datalink);
}

void PcapReader::open_ifc(const std::string& ifc)
{
	char errbuf[PCAP_ERRBUF_SIZE];
	errbuf[0] = 0;

	m_handle = pcap_open_live(ifc.c_str(), m_snaplen, 1, READ_TIMEOUT, errbuf);
	if (m_handle == nullptr) {
		throw PluginError(std::string("unable to open ifc: ") + errbuf);
	}
	if (errbuf[0] != 0) {
		std::cerr << errbuf << std::endl; // Print warning
	}
	if (pcap_setnonblock(m_handle, 1, errbuf) < 0) {
		close();
		throw PluginError(std::string("unable to set nonblocking mode: ") + errbuf);
	}

	m_datalink = pcap_datalink(m_handle);
	check_datalink(m_datalink);

	bpf_u_int32 net;
	if (pcap_lookupnet(ifc.c_str(), &net, &m_netmask, errbuf) != 0) {
		m_netmask = PCAP_NETMASK_UNKNOWN;
	}

	m_live = true;
}

void PcapReader::check_datalink(int datalink)
{
	(void) datalink;
	if (m_datalink != DLT_EN10MB && m_datalink != DLT_LINUX_SLL && m_datalink != DLT_RAW) {
#ifdef DLT_LINUX_SLL2
		if (m_datalink == DLT_LINUX_SLL2) {
			// DLT_LINUX_SLL2 is also supported
			return;
		} else {
			close();
			throw PluginError(
				"unsupported link type detected, supported types are: DLT_EN10MB, DLT_LINUX_SLL, "
				"DLT_LINUX_SLL2, and DLT_RAW");
		}
#endif
		close();
		throw PluginError(
			"unsupported link type detected, supported types are DLT_EN10MB and DLT_LINUX_SLL and "
			"DLT_RAW");
	}
}

void PcapReader::print_available_ifcs()
{
	char errbuf[PCAP_ERRBUF_SIZE];
	pcap_if_t* devs;
	pcap_if_t* d;
	int max_width = 0;
	int i = 0;

	if (pcap_findalldevs(&devs, errbuf) == -1) {
		throw PluginError(std::string("error in pcap_findalldevs: ") + errbuf);
	}

	if (devs != nullptr) {
		std::cout << "List of available interfaces:" << std::endl;
	}

	for (d = devs; d != nullptr; d = d->next) {
		int len = strlen(d->name);
		if (len > max_width) {
			max_width = len;
		}
	}
	for (d = devs; d != nullptr; d = d->next) {
#ifdef PCAP_IF_UP
		if (!(d->flags & PCAP_IF_UP)) {
			continue;
		}
#endif
		std::cout << std::setw(2) << ++i << ".  " << std::setw(max_width) << d->name;
		if (d->description) {
			std::cout << "    " << d->description << std::endl;
		} else {
			std::cout << std::endl;
		}
	}
	if (i == 0) {
		std::cout << "No available interfaces found" << std::endl;
	}

	pcap_freealldevs(devs);
}

void PcapReader::set_filter(const std::string& filter_str)
{
	struct bpf_program filter;
	if (pcap_compile(m_handle, &filter, filter_str.c_str(), 0, m_netmask) == -1) {
		throw PluginError(
			"couldn't parse filter " + filter_str + ": " + std::string(pcap_geterr(m_handle)));
	}
	if (pcap_setfilter(m_handle, &filter) == -1) {
		pcap_freecode(&filter);
		throw PluginError(
			"couldn't parse filter " + filter_str + ": " + std::string(pcap_geterr(m_handle)));
	}

	pcap_freecode(&filter);
}

InputPlugin::Result PcapReader::get(PacketBlock& packets)
{
	parser_opt_t opt = {&packets, false, false, m_datalink};
	int ret;

	UserData user_data = {&opt, m_parser_stats};

	if (m_handle == nullptr) {
		throw PluginError("no interface capture or file opened");
	}

	packets.cnt = 0;
	ret = pcap_dispatch(m_handle, PCAP_PACKET_BLOCK_SIZE, packet_handler, (u_char*) (&user_data));
	if (m_live) {
		if (ret == 0) {
			return Result::TIMEOUT;
		}
		if (ret > 0) {
			m_seen += ret;
			m_parsed += opt.pblock->cnt;
			return opt.packet_valid ? Result::PARSED : Result::NOT_PARSED;
		}
	} else {
		if (opt.pblock->cnt) {
			m_seen += ret ? ret : opt.pblock->cnt;
			m_parsed += opt.pblock->cnt;
			return Result::PARSED;
		} else if (ret == 0) {
			return Result::END_OF_FILE;
		}
	}
	if (ret < 0) {
		throw PluginError(pcap_geterr(m_handle));
	}
	return Result::NOT_PARSED;
}

static const PluginRegistrar<PcapReader, InputPluginFactory> pcapRegistrar(pcapPluginManifest);

} // namespace ipxp
