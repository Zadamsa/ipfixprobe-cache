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


#include <cstdint>
#include <cstring>
#include <utility>
#include <ipfixprobe/process.hpp>

#define TLS_HANDSHAKE_CLIENT_HELLO           1
#define TLS_HANDSHAKE_SERVER_HELLO           2
#define TLS_EXT_SERVER_NAME                  0
#define TLS_EXT_ALPN                         16
// draf-33, draft-34 a rfc9001, have this value defined as 0x39 == 57
#define TLS_EXT_QUIC_TRANSPORT_PARAMETERS_V1 0x39
// draf-13 az draft-32 have this value defined as 0xffa5 == 65445
#define TLS_EXT_QUIC_TRANSPORT_PARAMETERS    0xffa5
// draf-02 az draft-12 have this value defined as 0x26 == 38
#define TLS_EXT_QUIC_TRANSPORT_PARAMETERS_V2 0x26
#define TLS_EXT_GOOGLE_USER_AGENT            0x3129
#define MAX_TLS_EXT_LEN                      30UL

#define TLS_EXT_SERVER_NAME      0
#define TLS_EXT_ECLIPTIC_CURVES  10 // AKA supported_groups
#define TLS_EXT_EC_POINT_FORMATS 11
#define TLS_EXT_SIGNATURE_ALGORITHMS 13
#define TLS_EXT_ALPN             16
#define TLS_EXT_SUPPORTED_VER    43

namespace ipxp {
typedef struct TLSData {
   const uint8_t *start;
   const uint8_t *end;
   int            obejcts_parsed;
} TLSData;

struct __attribute__((packed)) TLSExtensionSNI {
   uint8_t  type;
   uint16_t length;
   /* Hostname bytes... */
};

struct __attribute__((packed)) TLSExtension {
   uint16_t type;
   uint16_t length;
   /* Extension specific data... */
};

union __attribute__((packed)) TLSVersion {
   uint16_t version;
   struct {
      uint8_t major;
      uint8_t minor;
   };
};

struct __attribute__((packed)) TLSHandshake {
   uint8_t     type;
   uint8_t     length1; // length field is 3 bytes long...
   uint16_t    length2;
   TLSVersion  version;

   /* Handshake data... */
};

#define TLS_HANDSHAKE 22
struct __attribute__((packed)) TLSHeader {
   uint8_t     type;
   TLSVersion  version;
   uint16_t    length;
   /* Record data... */
};

class TLSParser
{
public:
    /**
    * @brief Parses given payload as TLS packet.
    * @param packet Pointer to the payload.
    * @param length Length of the payload.
    * @return True if parsed succesfully, false otherwise.
    */
   bool parse(const uint8_t* packet, uint32_t length);

   /**
    * @brief Provide custom extensions parser of TLS Client or Server Hello packet.
    * @param callable Callable that will be called for each extension, takes type of extension,
    * pointer to its begin and its length.
    * @return True if extensions section has valid data, false otherwise.
    */
   bool parse_extensions(const std::function<void(uint16_t, const uint8_t*, uint16_t)>& callable) noexcept;

   /**
    * @brief Parses TLS SNI extension.
    * @param extension_data Pointer to the extension's begin.
    * @param extension_length Extension's length.
    */
   void parse_server_names(const uint8_t* extension_data, uint16_t extension_length);

   /**
    * @brief Parses TLS QUIC transport parameters extension.
    * @param extension_data Pointer to the extension's begin.
    * @param extension_length Extension's length.
    */
   void parse_quic_user_agent(const uint8_t* extension_payload, uint16_t extension_length) noexcept;

   /**
    * @brief Checks if given TLS packet is Client Hello.
    * @return True if it is, false otherwise.
    */
   bool is_client_hello() const noexcept;

   /**
    * @brief Checks if given TLS packet is Server Hello.
    * @return True if it is, false otherwise.
    */
   bool is_server_hello() const noexcept;

   /**
    * @brief Getter for TLS packet handshake section.
    * @return Handshake section.
    */
   TLSHandshake get_handshake() const noexcept;

   /**
    * @brief Getter for parsed extension.
    * @return Parsed extension.
    */
   const std::vector<TLSExtension>& get_extensions() const noexcept;

   /**
    * @brief Getter for parsed cipher suits.
    * @return Parsed cipher suits types.
    */
   const std::vector<uint16_t>& get_cipher_suits() const noexcept;

   /**
    * @brief Getter for parsed elliptic curves.
    * @return Parsed elliptic curves ids.
    */
   const std::vector<uint16_t>& get_elliptic_curves() const noexcept;

   /**
    * @brief Getter for parsed elliptic curves point formats.
    * @return Parsed elliptic curves point formats types.
    */
   const std::vector<uint16_t>& get_elliptic_curve_point_formats() const noexcept;

   /**
    * @brief Getter for parsed application layer protocol negotiations.
    * @return Pointers to parsed protocol names and its lengths.
    */
   const std::vector<std::pair<const char*, uint16_t>>& get_alpns() const noexcept;

   /**
    * @brief Getter for parsed supported versions.
    * @return Parsed supported versions.
    */
   const std::vector<uint16_t>& get_supported_versions() const noexcept;

   /**
    * @brief Getter for parsed server names from SNI extension.
    * @return Pointers to SNI names and its lengths.
    */
   const std::vector<std::pair<const char*, uint16_t>>& get_server_names() const noexcept;

   /**
    * @brief Getter for parsed signature algorithms.
    * @return Parsed signature algorithms types.
    */
   const std::vector<uint16_t>& get_signature_algorithms() const noexcept;

   /**
    * @brief Save parsed alpns to given buffer restricted with buffer length.
    * @param destination Destination buffer.
    * @param size Destination buffer size.
    */
   void save_alpns(char* destination, uint32_t size) const noexcept;

   /**
    * @brief Save parsed server names from SNI extension restricted with buffer length.
    * @param destination Destination buffer.
    * @param size Destination buffer size.
    */
   void save_server_names(char* destination, uint32_t size) const noexcept;

   /**
    * @brief Save parsed QUIC user agent from QUIC transport parameters extension restricted with buffer length.
    * @param destination Destination buffer.
    * @param size Destination buffer size.
    */
   void save_quic_user_agent(char* destination, uint32_t size) const noexcept;

   /**
    * @brief Checks if given value is GREASE.
    * @param value Value to check.
    * @return True if value is GREASE, false otherwise
    */
   static bool is_grease_value(uint16_t value);
private:
    bool parse_client_hello_extensions() noexcept;
    bool parse_server_hello_extensions() noexcept;
    void parse_alpn(const uint8_t* extension_data, uint16_t extension_length);
    bool parse_tls_handshake() noexcept;
    bool parse_cipher_suites() noexcept;
    bool parse_tls_header() const noexcept;
    bool parse_session_id() noexcept;
    bool parse_compression_methods()noexcept;
    bool has_valid_extension_length() const noexcept;
    void parse_elliptic_curves(const uint8_t* extension_payload, uint16_t extension_length)noexcept;
    void
    parse_elliptic_curve_point_formats(const uint8_t* extension_payload, uint16_t extension_length) noexcept;
    void parse_supported_versions(const uint8_t* extension_data, uint16_t extension_length)noexcept;
    void parse_signature_algorithms(const uint8_t* extension_data, uint16_t extension_length)noexcept;
    void clear_parsed_data() noexcept;

   const uint8_t* m_packet_data{nullptr};
   uint32_t m_packet_length{0};

   constexpr static inline const uint32_t TLS_RANDOM_BYTES_LENGTH = 32;
   uint32_t m_session_id_section_length {0};
   uint32_t m_cipher_suites_section_length {0};
   uint32_t m_compression_methods_section_length {0};

   std::vector<TLSExtension> m_extensions;
   std::vector<uint16_t> m_cipher_suits;
   std::vector<uint16_t> m_signature_algorithms;
   std::vector<uint16_t> m_elliptic_curves;
   std::vector<uint16_t> m_elliptic_curve_point_formats;
   std::vector<std::pair<const char*, uint16_t>> m_alpns;
   std::vector<uint16_t> m_supported_versions;
   std::vector<std::pair<const char*, uint16_t>> m_serverNames;

   const TLSHandshake* m_handshake;
   uint16_t m_objects_parsed {0};
};
}
