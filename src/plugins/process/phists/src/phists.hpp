/**
 * @file
 * @brief Plugin for parsing phists traffic.
 * @author Karel Hynek <hynekkar@fit.cvut.cz>
 * @author Pavel Siska <siska@cesnet.cz>
 * @date 2025
 *
 * Copyright (c) 2025 CESNET
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <limits>
#include <sstream>
#include <string>

#ifdef WITH_NEMEA
#include "fields.h"
#endif

#include <ipfixprobe/flowifc.hpp>
#include <ipfixprobe/ipfix-basiclist.hpp>
#include <ipfixprobe/ipfix-elements.hpp>
#include <ipfixprobe/packet.hpp>
#include <ipfixprobe/processPlugin.hpp>

namespace ipxp {

#ifndef PHISTS_MINLEN
#define PHISTS_MINLEN 1
#endif

#define HISTOGRAM_SIZE 8

#define PHISTS_UNIREC_TEMPLATE "S_PHISTS_SIZES,S_PHISTS_IPT,D_PHISTS_SIZES,D_PHISTS_IPT"

UR_FIELDS(
	uint32* S_PHISTS_SIZES,
	uint32* S_PHISTS_IPT,
	uint32* D_PHISTS_SIZES,
	uint32* D_PHISTS_IPT)

class PHISTSOptParser : public OptionsParser {
public:
	bool m_include_zeroes;

	PHISTSOptParser()
		: OptionsParser("phists", "Processing plugin for packet histograms")
		, m_include_zeroes(false)
	{
		register_option(
			"i",
			"includezeroes",
			"",
			"Include zero payload packets",
			[this](const char* arg) {
				(void) arg;
				m_include_zeroes = true;
				return true;
			},
			OptionFlags::NoArgument);
	}
};

/**
 * \brief Flow record extension header for storing parsed PHISTS packets.
 */
struct RecordExtPHISTS : public RecordExt {
	typedef enum eHdrFieldID {
		SPhistsSizes = 1060,
		SPhistsIpt = 1061,
		DPhistsSizes = 1062,
		DPhistsIpt = 1063
	} eHdrSemantic;

	uint32_t size_hist[2][HISTOGRAM_SIZE];
	uint32_t ipt_hist[2][HISTOGRAM_SIZE];
	uint32_t last_ts[2];

	RecordExtPHISTS(int pluginID)
		: RecordExt(pluginID)
	{
		// inicializing histograms with zeros
		for (int i = 0; i < 2; i++) {
			memset(size_hist[i], 0, sizeof(uint32_t) * HISTOGRAM_SIZE);
			memset(ipt_hist[i], 0, sizeof(uint32_t) * HISTOGRAM_SIZE);
			last_ts[i] = 0;
		}
	}

#ifdef WITH_NEMEA
	virtual void fill_unirec(ur_template_t* tmplt, void* record)
	{
		ur_array_allocate(tmplt, record, F_S_PHISTS_SIZES, HISTOGRAM_SIZE);
		ur_array_allocate(tmplt, record, F_S_PHISTS_IPT, HISTOGRAM_SIZE);
		ur_array_allocate(tmplt, record, F_D_PHISTS_SIZES, HISTOGRAM_SIZE);
		ur_array_allocate(tmplt, record, F_D_PHISTS_IPT, HISTOGRAM_SIZE);
		for (int i = 0; i < HISTOGRAM_SIZE; i++) {
			ur_array_set(tmplt, record, F_S_PHISTS_SIZES, i, size_hist[0][i]);
			ur_array_set(tmplt, record, F_S_PHISTS_IPT, i, ipt_hist[0][i]);
			ur_array_set(tmplt, record, F_D_PHISTS_SIZES, i, size_hist[1][i]);
			ur_array_set(tmplt, record, F_D_PHISTS_IPT, i, ipt_hist[1][i]);
		}
	}

	const char* get_unirec_tmplt() const { return PHISTS_UNIREC_TEMPLATE; }
#endif // ifdef WITH_NEMEA

	virtual int fill_ipfix(uint8_t* buffer, int size)
	{
		int32_t bufferPtr;
		IpfixBasicList basiclist;

		basiclist.hdrEnterpriseNum = IpfixBasicList::CesnetPEM;
		// Check sufficient size of buffer
		int req_size = 4 * basiclist.HeaderSize() /* sizes, times, flags, dirs */
			+ 4 * HISTOGRAM_SIZE * sizeof(uint32_t); /* sizes */

		if (req_size > size) {
			return -1;
		}
		// Fill sizes
		// fill buffer with basic list header and SPhistsSizes
		bufferPtr
			= basiclist.FillBuffer(buffer, size_hist[0], HISTOGRAM_SIZE, (uint32_t) SPhistsSizes);
		bufferPtr += basiclist.FillBuffer(
			buffer + bufferPtr,
			size_hist[1],
			HISTOGRAM_SIZE,
			(uint32_t) DPhistsSizes);
		bufferPtr += basiclist.FillBuffer(
			buffer + bufferPtr,
			ipt_hist[0],
			HISTOGRAM_SIZE,
			(uint32_t) SPhistsIpt);
		bufferPtr += basiclist.FillBuffer(
			buffer + bufferPtr,
			ipt_hist[1],
			HISTOGRAM_SIZE,
			(uint32_t) DPhistsIpt);

		return bufferPtr;
	} // fill_ipfix

	const char** get_ipfix_tmplt() const
	{
		static const char* ipfix_tmplt[] = {IPFIX_PHISTS_TEMPLATE(IPFIX_FIELD_NAMES) nullptr};

		return ipfix_tmplt;
	}

	std::string get_text() const
	{
		std::ostringstream out;
		char dirs_c[2] = {'s', 'd'};

		for (int dir = 0; dir < 2; dir++) {
			out << dirs_c[dir] << "phistsize=(";
			for (int i = 0; i < HISTOGRAM_SIZE; i++) {
				out << size_hist[dir][i];
				if (i != HISTOGRAM_SIZE - 1) {
					out << ",";
				}
			}
			out << ")," << dirs_c[dir] << "phistipt=(";
			for (int i = 0; i < HISTOGRAM_SIZE; i++) {
				out << ipt_hist[dir][i];
				if (i != HISTOGRAM_SIZE - 1) {
					out << ",";
				}
			}
			out << "),";
		}
		return out.str();
	}
};

/**
 * \brief Flow cache plugin for parsing PHISTS packets.
 */
class PHISTSPlugin : public ProcessPlugin {
public:
	PHISTSPlugin(const std::string& params, int pluginID);
	~PHISTSPlugin();
	void init(const char* params);
	void close();
	OptionsParser* get_parser() const { return new PHISTSOptParser(); }
	std::string get_name() const { return "phists"; }
	RecordExt* get_ext() const { return new RecordExtPHISTS(m_pluginID); }
	ProcessPlugin* copy();

	int post_create(Flow& rec, const Packet& pkt);
	int post_update(Flow& rec, const Packet& pkt);

private:
	bool use_zeros;

	void update_record(RecordExtPHISTS* phists_data, const Packet& pkt);
	void update_hist(RecordExtPHISTS* phists_data, uint32_t value, uint32_t* histogram);
	void pre_export(Flow& rec);
	int64_t calculate_ipt(RecordExtPHISTS* phists_data, const struct timeval tv, uint8_t direction);

	static const uint32_t log2_lookup32[32];

	static inline uint32_t fastlog2_32(uint32_t value)
	{
		value |= value >> 1;
		value |= value >> 2;
		value |= value >> 4;
		value |= value >> 8;
		value |= value >> 16;
		return log2_lookup32[(uint32_t) (value * 0x07C4ACDD) >> 27];
	}

	static inline uint32_t no_overflow_increment(uint32_t value)
	{
		if (value == std::numeric_limits<uint32_t>::max()) {
			return value;
		}
		return value + 1;
	}
};

} // namespace ipxp
