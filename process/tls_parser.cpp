/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (C) 2022, CESNET z.s.p.o.
 */

/**
 * \file tls_parser.cpp
 * \brief Class for parsing TLS traffic.
 * \author Andrej Lukacovic lukacan1@fit.cvut.cz
 * \author Karel Hynek <Karel.Hynek@cesnet.cz>
 * \date 2022
 */

#include "tls_parser.hpp"
#include <endian.h>

namespace ipxp {
uint64_t quic_get_variable_length(const uint8_t *start, uint64_t &offset)
{
   // find out length of parameter field (and load parameter, then move offset) , defined in:
   // https://www.rfc-editor.org/rfc/rfc9000.html#name-summary-of-integer-encoding
   // this approach is used also in length field , and other QUIC defined fields.
   uint64_t tmp = 0;

   uint8_t two_bits = *(start + offset) & 0xC0;

   switch (two_bits) {
       case 0:
          tmp     = *(start + offset) & 0x3F;
          offset += sizeof(uint8_t);
          return tmp;

       case 64:
          tmp     = be16toh(*(uint16_t *) (start + offset)) & 0x3FFF;
          offset += sizeof(uint16_t);
          return tmp;

       case 128:
          tmp     = be32toh(*(uint32_t *) (start + offset)) & 0x3FFFFFFF;
          offset += sizeof(uint32_t);
          return tmp;

       case 192:
          tmp     = be64toh(*(uint64_t *) (start + offset)) & 0x3FFFFFFFFFFFFFFF;
          offset += sizeof(uint64_t);
          return tmp;

       default:
          return 0;
   }
} // quic_get_variable_length

bool TLSParser::is_grease_value(uint16_t val)
{
    return val != 0 && !(val & ~(0xFAFA)) && ((0x00FF & val) == (val >> 8));
}

bool TLSParser::parse(const uint8_t* packet, uint32_t length)
{
    m_packet_data = packet;
    m_packet_length = length;
    clear_parsed_data();

    if (!parse_tls_header()){
        return false;
    }
    if (!parse_tls_handshake()){
        return false;
    }

    if (m_handshake->type != TLS_HANDSHAKE_CLIENT_HELLO
        && m_handshake->type != TLS_HANDSHAKE_SERVER_HELLO) {
        return false;
    }

    if (!parse_session_id()) {
        return false;
    }

    if (!parse_cipher_suites()){
        return false;
    }
    if (!parse_compression_methods()){
        return false;
    }

    if (m_handshake->type == TLS_HANDSHAKE_CLIENT_HELLO){
        return parse_client_hello_extensions();
    }
    return parse_server_hello_extensions();
}

bool TLSParser::parse_tls_header() const noexcept
{
    const auto* tls_header = (const TLSHeader*) m_packet_data;
    if (sizeof(TLSHeader) > m_packet_length || tls_header == nullptr || tls_header->type != TLS_HANDSHAKE || tls_header->version.major != 3 || tls_header->version.minor > 3) {
        return false;
    }
    return true;
}

bool TLSParser::parse_tls_handshake() noexcept
{
    auto* handshake = (const TLSHandshake*)(m_packet_data + sizeof(TLSHeader));

    if ( sizeof(TLSHeader) + sizeof(TLSHandshake) > m_packet_length ||
        !(handshake->type == TLS_HANDSHAKE_CLIENT_HELLO || handshake->type == TLS_HANDSHAKE_SERVER_HELLO) ||
        handshake->version.major != 3 ||
        handshake->version.minor < 1 ||
        handshake->version.minor > 3) {
        return false;
    }
    m_handshake = handshake;
    return true;
}

bool TLSParser::parse_session_id() noexcept
{
    const auto session_id_section_offset
        = sizeof(TLSHeader) + sizeof(TLSHandshake) + TLS_RANDOM_BYTES_LENGTH;
    if (session_id_section_offset > m_packet_length){
        return false;
    }

    const auto sessionIdLength = *(m_packet_data + session_id_section_offset);
    m_session_id_section_length = sizeof(sessionIdLength) + sessionIdLength;
    if (session_id_section_offset + m_session_id_section_length > m_packet_length) {
        return false;
    }
    return true;
}

bool TLSParser::parse_cipher_suites() noexcept
{
    const auto cipher_suite_section_offset = sizeof(TLSHeader)
        + sizeof(TLSHandshake) + TLS_RANDOM_BYTES_LENGTH + m_session_id_section_length;
    if (cipher_suite_section_offset + sizeof(uint16_t) > m_packet_length){
        return false;
    }

    if (m_handshake->type == TLS_HANDSHAKE_SERVER_HELLO){
        m_cipher_suites_section_length = sizeof(uint16_t);
        return true;
    }

    // Else parse Client Hello
    const auto client_cipher_suites_length
        = ntohs(*(uint16_t*)(m_packet_data + cipher_suite_section_offset));
    if (cipher_suite_section_offset + sizeof(client_cipher_suites_length) + client_cipher_suites_length
        > m_packet_length){
        return false;
    }

    const auto* cipher_suites_begin
        = m_packet_data + cipher_suite_section_offset + sizeof(client_cipher_suites_length);
    const auto* cipher_suites_end = cipher_suites_begin + client_cipher_suites_length;
    for (const auto* cipher_suite = cipher_suites_begin; cipher_suite < cipher_suites_end;
         cipher_suite += sizeof(uint16_t)) {
        const auto type_id = ntohs(*(uint16_t *)(cipher_suite));
        if (!is_grease_value(type_id)){
            m_cipher_suits.push_back(type_id);
        }
    }
    m_cipher_suites_section_length = sizeof(client_cipher_suites_length) + client_cipher_suites_length;
    return true;
}

bool TLSParser::parse_compression_methods() noexcept
{
    const auto compression_methods_section_offset = sizeof(TLSHeader)
        + sizeof(TLSHandshake) + TLS_RANDOM_BYTES_LENGTH + m_session_id_section_length + m_cipher_suites_section_length;
    if (compression_methods_section_offset > m_packet_length){
        return false;
    }

    if (m_handshake->type == TLS_HANDSHAKE_SERVER_HELLO){
        m_compression_methods_section_length = 1;
        return true;
    }
    // Else parse Client Hello
    auto compression_methods_length
        = *(uint8_t *)(m_packet_data + compression_methods_section_offset);
    if (sizeof(compression_methods_length) + compression_methods_length > m_packet_length) {
        return false;
    }
    m_compression_methods_section_length
        = sizeof(compression_methods_length) + compression_methods_length;
    return true;
}

bool TLSParser::parse_client_hello_extensions() noexcept
{
    return parse_extensions([this](uint16_t extension_type, const uint8_t* extension_payload, uint16_t extension_length){
        if (extension_type == TLS_EXT_SERVER_NAME) {
            parse_server_names(extension_payload, extension_length);
        } else if (extension_type == TLS_EXT_ECLIPTIC_CURVES) {
            parse_elliptic_curves(extension_payload, extension_length);
        } else if (extension_type == TLS_EXT_EC_POINT_FORMATS) {
            parse_elliptic_curve_point_formats(extension_payload, extension_length);
        } else if (extension_type == TLS_EXT_ALPN) {
            parse_alpn(extension_payload, extension_length);
        } else if (extension_type == TLS_EXT_SIGNATURE_ALGORITHMS) {
            parse_signature_algorithms(extension_payload, extension_length);
        } else if (extension_type == TLS_EXT_SUPPORTED_VER) {
            parse_supported_versions(extension_payload, extension_length);
        }

        m_extensions.emplace_back(TLSExtension{extension_type, extension_length});
    });
}

void TLSParser::parse_server_names(const uint8_t* extension_data, uint16_t extension_length)
{
    if (sizeof(uint16_t) > extension_length){
        return;
    }
    const auto servername_list_length = ntohs(*(uint16_t *) extension_data);
    if (sizeof(servername_list_length) + servername_list_length > extension_length){
        return;
    }
    const auto* sni_begin = extension_data + sizeof(servername_list_length);
    const auto* sni_end = sni_begin + servername_list_length;

    for (auto* sni = sni_begin; sni + sizeof(TLSExtensionSNI) <= sni_end;) {
        const auto sni_length = ntohs(((TLSExtensionSNI*)sni)->length);

        if (sni + sizeof(TLSExtensionSNI) + sni_length > extension_data + extension_length) {
            break;
        }
        m_serverNames.emplace_back((const char*)sni + sizeof(TLSExtensionSNI), sni_length);

        sni += sizeof(TLSExtensionSNI) + sni_length;
        m_objects_parsed++;
    }
}

void TLSParser::parse_elliptic_curves(const uint8_t* extension_payload, uint16_t extension_length) noexcept
{
    if (sizeof(uint16_t) > extension_length){
        return;
    }
    const auto supported_groups_length = ntohs(*(uint16_t *) extension_payload);
    if (sizeof(supported_groups_length) + supported_groups_length > extension_length){
        return;
    }

    const auto* supported_groups_begin = extension_payload + sizeof(supported_groups_length);
    const auto* supported_groups_end = supported_groups_begin + supported_groups_length;

    for (auto* supported_group = supported_groups_begin; supported_group < supported_groups_end;
         supported_group += sizeof(uint16_t)) {
        const auto supported_group_type = ntohs(*(uint16_t *) supported_group);
        if (!is_grease_value(supported_group_type)) {
            m_elliptic_curves.push_back(supported_group_type);
        }
    }
}

void TLSParser::parse_elliptic_curve_point_formats(const uint8_t* extension_payload, uint16_t extension_length) noexcept
{
    if (sizeof(uint8_t) > extension_length){
        return;
    }
    const auto supported_formats_length = *extension_payload;
    if (sizeof(supported_formats_length) + supported_formats_length > extension_length){
        return;
    }

    const auto* supportedFormatsBegin = extension_payload + sizeof(supported_formats_length);
    const auto* supportedFormatsEnd = supportedFormatsBegin + supported_formats_length;
    std::string supportedFormats;

    for (auto* supported_format_pointer = supportedFormatsBegin;
         supported_format_pointer < supportedFormatsEnd;
         supported_format_pointer++) {
        const auto supported_format = *supported_format_pointer;
        if (!is_grease_value(supported_format)) {
            m_elliptic_curve_point_formats.push_back(supported_format);
        }
    }
}

void TLSParser::parse_alpn(const uint8_t* extension_data, uint16_t extension_length)
{
    if (sizeof(uint16_t) > extension_length){
        return;
    }
    const auto alpnExtensionLength = ntohs(*(uint16_t*) extension_data);
    if (sizeof(uint16_t) + alpnExtensionLength > extension_length){
        return;
    }

    const auto* alpn_begin = extension_data + sizeof(uint16_t);
    const auto* alpn_end = alpn_begin + alpnExtensionLength;

    for (auto* alpn = alpn_begin; alpn + sizeof(uint8_t) <= alpn_end; ) {
        const auto alpn_length = *alpn;

        if (alpn + sizeof(alpn_length) + alpn_length > alpn_begin + extension_length) {
            break;
        }
        m_alpns.emplace_back((const char*)alpn + sizeof(alpn_length), alpn_length);
        alpn += sizeof(uint8_t) + alpn_length;
        m_objects_parsed++;
    }
}

void TLSParser::parse_signature_algorithms(const uint8_t* extension_data, uint16_t extension_length) noexcept
{
    const auto* signature_algorithm = (const uint16_t*) extension_data;
    std::for_each_n(signature_algorithm,
                    extension_length /sizeof(uint16_t),[this](uint16_t algorithm){
                        m_signature_algorithms.push_back(ntohs(algorithm));
                    });
}

bool TLSParser::parse_server_hello_extensions() noexcept
{
    return parse_extensions([this](
                                uint16_t extension_type,
                                const uint8_t* extension_payload,
                                uint16_t extension_length) {
        if (extension_type == TLS_EXT_ALPN) {
            parse_alpn(extension_payload, extension_length);
        } else if (extension_type == TLS_EXT_SUPPORTED_VER) {
            parse_supported_versions(extension_payload, extension_length);
        }
    });
}

void TLSParser::parse_supported_versions(const uint8_t* extension_data, uint16_t extension_length) noexcept
{
    if (m_handshake->type == TLS_HANDSHAKE_SERVER_HELLO){
        if (sizeof(uint16_t) > extension_length){
            return;
        }
        m_supported_versions.push_back(ntohs(*(uint16_t*) extension_data));
        return;
    }
    //Else parse client hello
    if (sizeof(uint8_t) > extension_length){
        return;
    }
    const auto versions_length = *extension_data;
    if (sizeof(uint8_t) + versions_length > extension_length){
        return;
    }

    const auto version = (const uint16_t*) (extension_data + sizeof(versions_length));
    std::for_each_n(version, versions_length /2,[this](auto version){
        if (!is_grease_value(version)){
            m_supported_versions.push_back(ntohs(version));
        }
    });
}

bool TLSParser::parse_extensions(const std::function<void(uint16_t, const uint8_t*, uint16_t)>& callable) noexcept{
    if (!has_valid_extension_length()){
        return false;
    }
    const auto extensions_section_offset = sizeof(TLSHeader)
        + sizeof(TLSHandshake) + TLS_RANDOM_BYTES_LENGTH + m_session_id_section_length + m_cipher_suites_section_length
        + m_compression_methods_section_length;
    const auto extensions_section_length
        = ntohs(*(uint16_t*)(m_packet_data + extensions_section_offset));

    const auto* extensions_begin
        = m_packet_data + extensions_section_offset + sizeof(extensions_section_length);
    const auto* extensions_end = extensions_begin + extensions_section_length;

    for (auto* extension_ptr = extensions_begin; extension_ptr < extensions_end;) {
        const auto* extension    = (TLSExtension*) extension_ptr;
        const auto extension_length = ntohs(extension->length);
        const auto extension_type = ntohs(extension->type);

        if (extension_ptr + sizeof(TLSExtension) + extension_length > extensions_end) {
            break;
        }

        const auto* extensionPayload = extension_ptr + sizeof(TLSExtension);
        callable(extension_type, extensionPayload, extension_length);

        extension_ptr += sizeof(TLSExtension) + extension_length;
    }
    return true;
}

bool TLSParser::has_valid_extension_length() const noexcept
{
    const auto extensions_section_offset = sizeof(TLSHeader)
        + sizeof(TLSHandshake) + TLS_RANDOM_BYTES_LENGTH + m_session_id_section_length + m_cipher_suites_section_length
        + m_compression_methods_section_length;
    if (extensions_section_offset > m_packet_length){
        return false;
    }
    const auto extension_section_length
        = ntohs(*(uint16_t*)(m_packet_data + extensions_section_offset));
    if (extensions_section_offset + extension_section_length > m_packet_length){
        return false;
    }
   return true;
}

TLSHandshake TLSParser::get_handshake() const noexcept
{
    return *m_handshake;
}

bool TLSParser::is_client_hello() const noexcept
{
    return m_handshake->type == TLS_HANDSHAKE_CLIENT_HELLO;
}

bool TLSParser::is_server_hello() const noexcept
{
    return m_handshake->type == TLS_HANDSHAKE_SERVER_HELLO;
}

const std::vector<TLSExtension>& TLSParser::get_extensions() const noexcept
{
    return m_extensions;
}

const std::vector<uint16_t>& TLSParser::get_cipher_suits() const noexcept
{
    return m_cipher_suits;
}

const std::vector<uint16_t>& TLSParser::get_elliptic_curves() const noexcept
{
    return m_elliptic_curves;
}

const std::vector<uint16_t>& TLSParser::get_elliptic_curve_point_formats() const noexcept
{
    return m_elliptic_curve_point_formats;
}

const std::vector<std::pair<const char*, uint16_t>>& TLSParser::get_alpns() const noexcept
{
    return m_alpns;
}

const std::vector<std::pair<const char*, uint16_t>>& TLSParser::get_server_names() const noexcept
{
    return m_serverNames;
}

const std::vector<uint16_t>& TLSParser::get_supported_versions() const noexcept
{
    return m_supported_versions;
}

const std::vector<uint16_t>& TLSParser::get_signature_algorithms() const noexcept
{
    return m_signature_algorithms;
}

static void save_to_buffer(char* destination, const std::vector<std::pair<const char*, uint16_t>>& source, uint32_t size, char delimiter) noexcept
{
    std::for_each(source.begin(), source.end(), [destination, write_pos = 0UL, size,delimiter](const std::pair<const char*, uint16_t>& alpn) mutable {
        if (alpn.second + 2U > size - write_pos){
            destination[write_pos] = 0;
            return;
        }
        const auto bytes_to_write = std::min(size - write_pos - 2U, alpn.second + 2UL);
        memcpy(destination + write_pos, alpn.first, bytes_to_write);
        write_pos += alpn.second;
        destination[write_pos++] = delimiter;
    });
}

void TLSParser::save_server_names(char* destination, uint32_t size) const noexcept
{
    save_to_buffer(destination, m_serverNames, size, 0);
}

void TLSParser::save_alpns(char* destination, uint32_t size) const noexcept
{
    save_to_buffer(destination, m_alpns, size, 0);
}

void TLSParser::save_quic_user_agent(char* destination, uint32_t size) const noexcept
{
    save_to_buffer(destination, m_serverNames, size, 0);
}

void TLSParser::parse_quic_user_agent(const uint8_t* extension_payload, uint16_t extension_length) noexcept
{
    // compute end of quic_transport_parameters
    if (sizeof(uint16_t) > extension_length){
        return;
    }
    const auto quic_transport_parameters_length = ntohs(*(uint16_t *) extension_payload);
    if (sizeof(quic_transport_parameters_length) + quic_transport_parameters_length > extension_length) {
        return;
    }

    const auto quic_transport_parameters_begin = extension_payload + sizeof(quic_transport_parameters_length);
    const auto quic_transport_parameters_end = quic_transport_parameters_begin + quic_transport_parameters_length;
    for (const auto* parameter = quic_transport_parameters_begin; parameter < quic_transport_parameters_end; ) {
        auto offset = 0UL;
        const auto parameter_id  = quic_get_variable_length(parameter, offset);
        const auto parameter_length = quic_get_variable_length(parameter, offset);
        if (parameter + parameter_length > quic_transport_parameters_end) {
            return;
        }
        if (parameter_id == TLS_EXT_GOOGLE_USER_AGENT) {
            m_objects_parsed++;
        }
        parameter += parameter_length;
    }
}

void TLSParser::clear_parsed_data() noexcept
{
    m_extensions.clear();
    m_cipher_suits.clear();
    m_signature_algorithms.clear();
    m_elliptic_curves.clear();
    m_elliptic_curve_point_formats.clear();
    m_alpns.clear();
    m_supported_versions.clear();
    m_serverNames.clear();
}

}
