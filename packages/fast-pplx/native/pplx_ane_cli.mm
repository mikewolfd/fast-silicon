#import <Foundation/Foundation.h>

#include "fast_shared_apple.hpp"
#include "pplx_coreml_runner.h"
#include "pplx_fast_io.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zlib.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

inline uint16_t read_le16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

inline uint32_t read_le32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

bool decode_npy_to_f32(const std::vector<uint8_t>& raw, std::vector<float>& out) {
    if (raw.size() < 10 || std::memcmp(raw.data(), "\x93NUMPY", 6) != 0) {
        return false;
    }
    const uint8_t major = raw[6];
    size_t hdr_len = 0;
    size_t pos = 0;
    if (major == 1) {
        hdr_len = read_le16(raw.data() + 8);
        pos = 10 + hdr_len;
    } else if (major == 2 || major == 3) {
        if (raw.size() < 12) {
            return false;
        }
        hdr_len = read_le32(raw.data() + 8);
        pos = 12 + hdr_len;
    } else {
        return false;
    }
    if (pos > raw.size()) {
        return false;
    }
    std::string npy_hdr(reinterpret_cast<const char*>(raw.data() + (pos - hdr_len)), hdr_len);
    const size_t data_bytes = raw.size() - pos;
    if (npy_hdr.find("<f4") != std::string::npos) {
        if ((data_bytes % 4) != 0) {
            return false;
        }
        out.resize(data_bytes / 4);
        std::memcpy(out.data(), raw.data() + pos, data_bytes);
        return true;
    }
    if (npy_hdr.find("<f2") != std::string::npos) {
        if ((data_bytes % 2) != 0) {
            return false;
        }
        const size_t n = data_bytes / 2;
        out.resize(n);
        const auto* src = reinterpret_cast<const uint16_t*>(raw.data() + pos);
        for (size_t i = 0; i < n; ++i) {
            _Float16 h;
            std::memcpy(&h, &src[i], sizeof(h));
            out[i] = static_cast<float>(h);
        }
        return true;
    }
    return false;
}

// Read an uncompressed or deflate-compressed .npy float32/float16 entry from an .npz zip archive.
// Uses the ZIP Central Directory (EOCD -> CD -> Local Header) so entries written with
// general-purpose bit 3 (data descriptor, where local header sizes are 0) are parsed accurately.
bool read_npz_f32_entry(const std::string& npz_path, const std::string& target_name, std::vector<float>& out) {
    FILE* f = std::fopen(npz_path.c_str(), "rb");
    if (!f) {
        return false;
    }
    if (::fseeko(f, 0, SEEK_END) != 0) {
        std::fclose(f);
        return false;
    }
    const off_t file_size_off = ::ftello(f);
    if (file_size_off < 22) {
        std::fclose(f);
        return false;
    }
    const uint64_t file_size = static_cast<uint64_t>(file_size_off);

    // Search tail up to max EOCD comment size (65535) + 22 bytes for EOCD signature PK\x05\x06.
    const uint64_t tail_len = std::min<uint64_t>(file_size, 65557);
    const uint64_t tail_off = file_size - tail_len;
    if (::fseeko(f, static_cast<off_t>(tail_off), SEEK_SET) != 0) {
        std::fclose(f);
        return false;
    }
    std::vector<uint8_t> tail(static_cast<size_t>(tail_len));
    if (std::fread(tail.data(), 1, tail.size(), f) != tail.size()) {
        std::fclose(f);
        return false;
    }

    int64_t eocd_pos = -1;
    for (int64_t i = static_cast<int64_t>(tail.size()) - 22; i >= 0; --i) {
        if (tail[static_cast<size_t>(i)] == 0x50 &&
            tail[static_cast<size_t>(i + 1)] == 0x4b &&
            tail[static_cast<size_t>(i + 2)] == 0x05 &&
            tail[static_cast<size_t>(i + 3)] == 0x06) {
            const uint16_t comment_len = read_le16(tail.data() + i + 20);
            if (static_cast<size_t>(i) + 22 + comment_len == tail.size()) {
                eocd_pos = i;
                break;
            }
        }
    }
    if (eocd_pos < 0) {
        std::fclose(f);
        return false;
    }

    const uint8_t* eocd = tail.data() + eocd_pos;
    const uint16_t cd_entries = read_le16(eocd + 10);
    const uint32_t cd_size = read_le32(eocd + 12);
    const uint32_t cd_offset = read_le32(eocd + 16);
    if (cd_size == 0xFFFFFFFFu || cd_offset == 0xFFFFFFFFu ||
        static_cast<uint64_t>(cd_offset) + static_cast<uint64_t>(cd_size) > file_size) {
        std::fprintf(stderr, "Unsupported ZIP64 or truncated central directory in %s\n", npz_path.c_str());
        std::fclose(f);
        return false;
    }

    if (::fseeko(f, static_cast<off_t>(cd_offset), SEEK_SET) != 0) {
        std::fclose(f);
        return false;
    }
    std::vector<uint8_t> cd_buf(cd_size);
    if (cd_size > 0 && std::fread(cd_buf.data(), 1, cd_size, f) != cd_size) {
        std::fclose(f);
        return false;
    }

    const std::string want = target_name + ".npy";
    size_t off = 0;
    for (uint16_t idx = 0; idx < cd_entries && off + 46 <= cd_buf.size(); ++idx) {
        const uint8_t* rec = cd_buf.data() + off;
        if (rec[0] != 0x50 || rec[1] != 0x4b || rec[2] != 0x01 || rec[3] != 0x02) {
            break;
        }
        const uint16_t flags = read_le16(rec + 8);
        const uint16_t method = read_le16(rec + 10);
        const uint32_t comp_sz = read_le32(rec + 20);
        const uint32_t uncomp_sz = read_le32(rec + 24);
        const uint16_t name_len = read_le16(rec + 28);
        const uint16_t extra_len = read_le16(rec + 30);
        const uint16_t comment_len = read_le16(rec + 32);
        const uint32_t local_off = read_le32(rec + 42);

        const size_t rec_total = 46u + name_len + extra_len + comment_len;
        if (off + rec_total > cd_buf.size()) {
            break;
        }

        std::string name(reinterpret_cast<const char*>(rec + 46), name_len);
        off += rec_total;
        if (name != want) {
            continue;
        }

        if (comp_sz == 0xFFFFFFFFu || uncomp_sz == 0xFFFFFFFFu || local_off == 0xFFFFFFFFu) {
            std::fprintf(stderr, "Unsupported ZIP64 entry %s in %s\n", name.c_str(), npz_path.c_str());
            std::fclose(f);
            return false;
        }
        if ((flags & 0x0001u) != 0) {
            std::fprintf(stderr, "Unsupported encrypted ZIP entry %s in %s\n", name.c_str(), npz_path.c_str());
            std::fclose(f);
            return false;
        }

        // Read local file header to advance past local name + local extra field (authoritative sizes come from CD).
        if (static_cast<uint64_t>(local_off) + 30 > file_size ||
            ::fseeko(f, static_cast<off_t>(local_off), SEEK_SET) != 0) {
            std::fclose(f);
            return false;
        }
        uint8_t lhdr[30];
        if (std::fread(lhdr, 1, 30, f) != 30 ||
            lhdr[0] != 0x50 || lhdr[1] != 0x4b || lhdr[2] != 0x03 || lhdr[3] != 0x04) {
            std::fclose(f);
            return false;
        }
        const uint16_t local_name_len = read_le16(lhdr + 26);
        const uint16_t local_extra_len = read_le16(lhdr + 28);
        const uint64_t data_off = static_cast<uint64_t>(local_off) + 30u + local_name_len + local_extra_len;
        if (data_off + static_cast<uint64_t>(comp_sz) > file_size ||
            ::fseeko(f, static_cast<off_t>(data_off), SEEK_SET) != 0) {
            std::fclose(f);
            return false;
        }

        std::vector<uint8_t> comp(comp_sz);
        if (comp_sz > 0 && std::fread(comp.data(), 1, comp_sz, f) != comp_sz) {
            std::fclose(f);
            return false;
        }
        std::fclose(f);

        std::vector<uint8_t> raw;
        if (method == 0) {
            raw = std::move(comp);
        } else if (method == 8) {
            raw.resize(uncomp_sz);
            z_stream zs{};
            zs.next_in = comp.data();
            zs.avail_in = static_cast<uInt>(comp.size());
            zs.next_out = raw.data();
            zs.avail_out = static_cast<uInt>(raw.size());
            if (inflateInit2(&zs, -MAX_WBITS) != Z_OK) {
                return false;
            }
            int rc = inflate(&zs, Z_FINISH);
            inflateEnd(&zs);
            if (rc != Z_STREAM_END) {
                return false;
            }
        } else {
            return false;
        }
        return decode_npy_to_f32(raw, out);
    }

    std::fclose(f);
    return false;
}

bool mmap_safetensors_weight(
    const std::string& path,
    int& out_fd,
    void*& out_map,
    size_t& out_map_len,
    const uint16_t*& out_ptr,
    int& out_rows,
    int& out_cols) {
    out_fd = ::open(path.c_str(), O_RDONLY);
    if (out_fd < 0) {
        return false;
    }
    struct stat st {};
    if (::fstat(out_fd, &st) != 0 || st.st_size < 16) {
        ::close(out_fd);
        out_fd = -1;
        return false;
    }
    out_map_len = static_cast<size_t>(st.st_size);
    out_map = ::mmap(nullptr, out_map_len, PROT_READ, MAP_PRIVATE, out_fd, 0);
    if (out_map == MAP_FAILED) {
        ::close(out_fd);
        out_fd = -1;
        out_map = nullptr;
        return false;
    }
    uint64_t hdr_len = 0;
    std::memcpy(&hdr_len, out_map, 8);
    if (hdr_len == 0 || 8 + hdr_len > out_map_len) {
        ::munmap(out_map, out_map_len);
        ::close(out_fd);
        out_fd = -1;
        out_map = nullptr;
        return false;
    }
    NSData* json_data = [NSData dataWithBytesNoCopy:const_cast<uint8_t*>(static_cast<const uint8_t*>(out_map) + 8)
                                            length:static_cast<NSUInteger>(hdr_len)
                                      freeWhenDone:NO];
    NSError* err = nil;
    id parsed = [NSJSONSerialization JSONObjectWithData:json_data options:0 error:&err];
    NSDictionary* hdr = [parsed isKindOfClass:[NSDictionary class]] ? static_cast<NSDictionary*>(parsed) : nil;
    if (!hdr || err != nil) {
        if (err != nil) {
            std::fprintf(stderr, "Failed to parse safetensors JSON header: %s\n",
                         [[err localizedDescription] UTF8String]);
        }
        ::munmap(out_map, out_map_len);
        ::close(out_fd);
        out_fd = -1;
        out_map = nullptr;
        return false;
    }

    NSArray<NSString*>* candidate_keys = @[
        @"language_model.embed_tokens.weight",
        @"model.embed_tokens.weight",
        @"embed_tokens.weight",
        @"weight",
    ];
    NSDictionary* info = nil;
    for (NSString* key in candidate_keys) {
        id entry = hdr[key];
        if ([entry isKindOfClass:[NSDictionary class]]) {
            info = static_cast<NSDictionary*>(entry);
            break;
        }
    }
    if (!info || ![info[@"dtype"] isEqualToString:@"F16"]) {
        ::munmap(out_map, out_map_len);
        ::close(out_fd);
        out_fd = -1;
        out_map = nullptr;
        return false;
    }
    NSArray* shape = info[@"shape"];
    NSArray* offsets = info[@"data_offsets"];
    if (![shape isKindOfClass:[NSArray class]] || shape.count != 2 ||
        ![offsets isKindOfClass:[NSArray class]] || offsets.count != 2) {
        ::munmap(out_map, out_map_len);
        ::close(out_fd);
        out_fd = -1;
        out_map = nullptr;
        return false;
    }
    out_rows = [shape[0] intValue];
    out_cols = [shape[1] intValue];
    const uint64_t b0 = [offsets[0] unsignedLongLongValue];
    const uint64_t b1 = [offsets[1] unsignedLongLongValue];
    const uint64_t expected = static_cast<uint64_t>(out_rows) * static_cast<uint64_t>(out_cols) * sizeof(uint16_t);
    if (out_rows <= 0 || out_cols <= 0 || b1 < b0 || (b1 - b0) != expected ||
        8 + hdr_len + b1 > out_map_len) {
        ::munmap(out_map, out_map_len);
        ::close(out_fd);
        out_fd = -1;
        out_map = nullptr;
        return false;
    }
    out_ptr = reinterpret_cast<const uint16_t*>(
        static_cast<const uint8_t*>(out_map) + 8 + hdr_len + b0);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        std::string weights_npz;
        std::string embed_sf;
        std::vector<std::string> stage_paths;
        int seq_len = 128;
        int chunk_size = 32;
        int pipe_depth = 4;
        int compute_units = 0;  // 0=CPU_AND_NE, 1=CPU_ONLY, 2=ALL, 3=CPU_AND_GPU
        int iters = 20;
        bool inspect_only = false;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--weights-npz" && i + 1 < argc) {
                weights_npz = argv[++i];
            } else if (arg == "--embed-safetensors" && i + 1 < argc) {
                embed_sf = argv[++i];
            } else if (arg == "--stage" && i + 1 < argc) {
                stage_paths.emplace_back(argv[++i]);
            } else if (arg == "--seq-len" && i + 1 < argc) {
                seq_len = std::atoi(argv[++i]);
            } else if (arg == "--chunk-size" && i + 1 < argc) {
                chunk_size = std::atoi(argv[++i]);
            } else if (arg == "--pipe-depth" && i + 1 < argc) {
                pipe_depth = std::atoi(argv[++i]);
            } else if (arg == "--compute-units" && i + 1 < argc) {
                compute_units = std::atoi(argv[++i]);
            } else if (arg == "--iters" && i + 1 < argc) {
                iters = std::atoi(argv[++i]);
            } else if (arg == "--inspect") {
                inspect_only = true;
            }
        }

        if (stage_paths.empty()) {
            std::fprintf(stderr, "Usage: pplx-ane-cli --stage path0.mlmodelc [--stage path1.mlmodelc ...] [--seq-len N] [--inspect]\n");
            return 1;
        }

        if (inspect_only) {
            char buf[512];
            int exit_code = 0;
            for (size_t k = 0; k < stage_paths.size(); ++k) {
                buf[0] = '\0';
                int rc = pplx_ane_inspect_plan(stage_paths[k].c_str(), compute_units, buf, sizeof(buf));
                std::printf("stage %zu (%s): %s\n", k, stage_paths[k].c_str(),
                            buf[0] != '\0' ? buf : "{\"error\":\"inspect_failed\"}");
                if (rc <= 0) {
                    exit_code = 1;
                }
            }
            return exit_code;
        }

        if (chunk_size <= 0 || seq_len <= 0 || (seq_len % chunk_size) != 0) {
            std::fprintf(
                stderr,
                "Invalid alignment: seq_len (%d) must be positive and divisible by chunk_size (%d).\n",
                seq_len,
                chunk_size);
            return 1;
        }
        if (iters <= 0 || pipe_depth <= 0) {
            std::fprintf(stderr, "Invalid parameters: iters (%d) and pipe_depth (%d) must be positive.\n", iters, pipe_depth);
            return 1;
        }

        if (weights_npz.empty() || embed_sf.empty()) {
            std::fprintf(stderr, "Both --weights-npz and --embed-safetensors are required for benchmark mode.\n");
            return 1;
        }

        int embed_fd = -1;
        void* embed_map = nullptr;
        size_t embed_map_len = 0;
        const uint16_t* embed_ptr = nullptr;
        int vocab_size = 0;
        int hidden_dim = 0;
        if (!mmap_safetensors_weight(embed_sf, embed_fd, embed_map, embed_map_len, embed_ptr, vocab_size, hidden_dim)) {
            std::fprintf(stderr, "Failed to mmap F16 embed_tokens safetensors: %s\n", embed_sf.c_str());
            return 1;
        }

        std::vector<float> contextual_proj;
        if (!read_npz_f32_entry(weights_npz, "contextual_projection.weight", contextual_proj) &&
            !read_npz_f32_entry(weights_npz, "contextual_proj", contextual_proj)) {
            std::fprintf(stderr, "Failed to read contextual_projection.weight from %s\n", weights_npz.c_str());
            ::munmap(embed_map, embed_map_len);
            ::close(embed_fd);
            return 1;
        }

        const size_t expected_proj = static_cast<size_t>(hidden_dim) * static_cast<size_t>(hidden_dim);
        if (contextual_proj.size() < expected_proj) {
            std::fprintf(
                stderr,
                "contextual_projection.weight too small (%zu < %zu)\n",
                contextual_proj.size(),
                expected_proj);
            ::munmap(embed_map, embed_map_len);
            ::close(embed_fd);
            return 1;
        }

        std::vector<const char*> c_paths;
        c_paths.reserve(stage_paths.size());
        for (const auto& p : stage_paths) {
            c_paths.push_back(p.c_str());
        }

        const int rope_half = 32;
        void* runner = pplx_ane_load_stages(
            c_paths.data(),
            static_cast<int>(c_paths.size()),
            seq_len,
            hidden_dim,
            chunk_size,
            rope_half,
            pipe_depth,
            compute_units);
        if (!runner) {
            const char* err = pplx_ane_last_error();
            std::fprintf(stderr, "Failed to load CoreML ANE stage pipeline: %s\n",
                         (err && err[0]) ? err : "unknown error");
            ::munmap(embed_map, embed_map_len);
            ::close(embed_fd);
            return 1;
        }

        const int n_chunks = seq_len / chunk_size;
        std::vector<uint16_t> rope_cos(static_cast<size_t>(seq_len) * rope_half);
        std::vector<uint16_t> rope_sin(static_cast<size_t>(seq_len) * rope_half);
        pplx_build_rope_table_f16(rope_cos.data(), rope_sin.data(), seq_len, rope_half, 1e7);

        std::vector<int32_t> tok_ids(seq_len, 100);
        std::vector<int32_t> doc_ids(seq_len, 0);
        std::vector<int32_t> pos_ids(seq_len);
        for (int s = 0; s < seq_len; ++s) {
            pos_ids[s] = s;
        }

        std::vector<uint16_t> h_in(static_cast<size_t>(hidden_dim) * seq_len);
        std::vector<uint16_t> bias(static_cast<size_t>(seq_len) * seq_len);
        std::vector<uint16_t> c32(static_cast<size_t>(seq_len) * rope_half);
        std::vector<uint16_t> s32(static_cast<size_t>(seq_len) * rope_half);
        std::vector<uint16_t> tvalid(seq_len);
        std::vector<uint16_t> carry(n_chunks);
        std::vector<uint16_t> h_out(static_cast<size_t>(hidden_dim) * seq_len);

        pplx_pack_bucket_f16(
            embed_ptr, vocab_size, hidden_dim, seq_len, chunk_size, rope_half,
            tok_ids.data(), doc_ids.data(), pos_ids.data(),
            rope_cos.data(), rope_sin.data(), seq_len,
            h_in.data(), bias.data(), c32.data(), s32.data(), tvalid.data(), carry.data());

        // Warmup
        for (int w = 0; w < 2; ++w) {
            if (pplx_ane_exec(runner, h_in.data(), bias.data(), c32.data(), s32.data(), tvalid.data(), carry.data(), h_out.data()) < 0.0) {
                std::fprintf(stderr, "Warmup failed: %s\n", pplx_ane_last_error());
                pplx_ane_free(runner);
                ::munmap(embed_map, embed_map_len);
                ::close(embed_fd);
                return 1;
            }
        }

        std::vector<std::vector<uint16_t>> h_outs(static_cast<size_t>(iters), std::vector<uint16_t>(static_cast<size_t>(hidden_dim) * seq_len));
        std::vector<void*> jobs;
        jobs.reserve(static_cast<size_t>(iters));
        bool bench_ok = true;
        const double t0 = fast_shared::now_sec();
        for (int k = 0; k < iters; ++k) {
            void* j = pplx_ane_submit(
                runner, h_in.data(), bias.data(), c32.data(), s32.data(), tvalid.data(), carry.data(), h_outs[k].data());
            if (!j) {
                std::fprintf(stderr, "Submit failed at iter %d: %s\n", k, pplx_ane_last_error());
                bench_ok = false;
                break;
            }
            jobs.push_back(j);
        }
        for (void* j : jobs) {
            if (pplx_ane_wait(j) < 0.0) {
                std::fprintf(stderr, "Wait failed: %s\n", pplx_ane_last_error());
                bench_ok = false;
            }
        }
        const double elapsed = fast_shared::now_sec() - t0;

        if (!bench_ok) {
            pplx_ane_free(runner);
            ::munmap(embed_map, embed_map_len);
            ::close(embed_fd);
            return 1;
        }

        int32_t s0 = 0;
        int32_t s1 = seq_len;
        std::vector<float> vec(static_cast<size_t>(hidden_dim));
        pplx_pool_and_project_f32(
            h_outs.back().data(), hidden_dim, seq_len, 1, &s0, &s1,
            nullptr, contextual_proj.data(), hidden_dim, hidden_dim, 0, 1, vec.data());

        const double tok_per_s = static_cast<double>(iters) * seq_len / elapsed;
        std::printf(
            "{\"seq_len\":%d,\"stages\":%zu,\"iters\":%d,\"elapsed_s\":%.4f,\"ms_per_bucket\":%.3f,\"tok_per_s\":%.1f,\"v0\":%.6f}\n",
            seq_len, stage_paths.size(), iters, elapsed, (elapsed / iters) * 1000.0, tok_per_s, vec[0]);

        pplx_ane_free(runner);
        ::munmap(embed_map, embed_map_len);
        ::close(embed_fd);
    }
    return 0;
}
