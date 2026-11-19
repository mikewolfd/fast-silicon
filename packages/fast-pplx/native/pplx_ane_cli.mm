#import <Foundation/Foundation.h>

#include "fast_shared_apple.hpp"
#include "pplx_coreml_runner.h"
#include "pplx_fast_io.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zlib.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Read an uncompressed or deflate-compressed .npy float32/float16 entry from an .npz zip archive.
bool read_npz_f32_entry(const std::string& npz_path, const std::string& target_name, std::vector<float>& out) {
    FILE* f = std::fopen(npz_path.c_str(), "rb");
    if (!f) {
        return false;
    }
    const std::string want = target_name + ".npy";
    uint8_t sig[4];
    while (std::fread(sig, 1, 4, f) == 4) {
        if (sig[0] != 0x50 || sig[1] != 0x4b || sig[2] != 0x03 || sig[3] != 0x04) {
            break;
        }
        uint8_t hdr[26];
        if (std::fread(hdr, 1, 26, f) != 26) {
            break;
        }
        const uint16_t method = hdr[4] | (static_cast<uint16_t>(hdr[5]) << 8);
        uint32_t comp_sz = hdr[14] | (hdr[15] << 8) | (hdr[16] << 16) | (hdr[17] << 24);
        uint32_t uncomp_sz = hdr[18] | (hdr[19] << 8) | (hdr[20] << 16) | (hdr[21] << 24);
        const uint16_t name_len = hdr[22] | (static_cast<uint16_t>(hdr[23]) << 8);
        const uint16_t extra_len = hdr[24] | (static_cast<uint16_t>(hdr[25]) << 8);

        std::string name(name_len, '\0');
        if (name_len > 0 && std::fread(&name[0], 1, name_len, f) != name_len) {
            break;
        }
        if (extra_len > 0) {
            std::fseek(f, extra_len, SEEK_CUR);
        }

        if (name != want) {
            std::fseek(f, comp_sz, SEEK_CUR);
            continue;
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

        if (raw.size() < 10 || std::memcmp(raw.data(), "\x93NUMPY", 6) != 0) {
            return false;
        }
        const uint8_t major = raw[6];
        size_t hdr_len = 0;
        size_t pos = 0;
        if (major == 1) {
            hdr_len = raw[8] | (static_cast<size_t>(raw[9]) << 8);
            pos = 10 + hdr_len;
        } else {
            hdr_len = raw[8] | (raw[9] << 8) | (raw[10] << 16) | (raw[11] << 24);
            pos = 12 + hdr_len;
        }
        std::string npy_hdr(reinterpret_cast<const char*>(raw.data() + (pos - hdr_len)), hdr_len);
        const size_t data_bytes = raw.size() - pos;
        if (npy_hdr.find("<f4") != std::string::npos) {
            out.resize(data_bytes / 4);
            std::memcpy(out.data(), raw.data() + pos, data_bytes);
            return true;
        }
        if (npy_hdr.find("<f2") != std::string::npos) {
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
    if (8 + hdr_len > out_map_len) {
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
    NSDictionary* hdr = [NSJSONSerialization JSONObjectWithData:json_data options:0 error:&err];
    NSDictionary* info = hdr[@"weight"];
    if (!info) {
        info = hdr[@"language_model.embed_tokens.weight"];
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
    out_rows = [shape[0] intValue];
    out_cols = [shape[1] intValue];
    const uint64_t b0 = [offsets[0] unsignedLongLongValue];
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
            for (size_t k = 0; k < stage_paths.size(); ++k) {
                int rc = pplx_ane_inspect_plan(stage_paths[k].c_str(), compute_units, buf, sizeof(buf));
                std::printf("stage %zu (%s): %s\n", k, stage_paths[k].c_str(), rc > 0 ? buf : "{\"error\":\"inspect_failed\"}");
            }
            return 0;
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

        std::vector<const char*> c_paths;
        c_paths.reserve(stage_paths.size());
        for (const auto& p : stage_paths) {
            c_paths.push_back(p.c_str());
        }

        void* runner = pplx_ane_load_stages(
            c_paths.data(),
            static_cast<int>(c_paths.size()),
            seq_len,
            hidden_dim,
            32,
            32,
            pipe_depth,
            compute_units);
        if (!runner) {
            std::fprintf(stderr, "Failed to load CoreML ANE stage pipeline.\n");
            ::munmap(embed_map, embed_map_len);
            ::close(embed_fd);
            return 1;
        }

        const int rope_half = 32;
        const int chunk_size = 32;
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
                std::fprintf(stderr, "Warmup failed\n");
                pplx_ane_free(runner);
                ::munmap(embed_map, embed_map_len);
                ::close(embed_fd);
                return 1;
            }
        }

        std::vector<std::vector<uint16_t>> h_outs(static_cast<size_t>(iters), std::vector<uint16_t>(static_cast<size_t>(hidden_dim) * seq_len));
        std::vector<void*> jobs;
        jobs.reserve(static_cast<size_t>(iters));
        const double t0 = fast_shared::now_sec();
        for (int k = 0; k < iters; ++k) {
            jobs.push_back(pplx_ane_submit(
                runner, h_in.data(), bias.data(), c32.data(), s32.data(), tvalid.data(), carry.data(), h_outs[k].data()));
        }
        for (void* j : jobs) {
            pplx_ane_wait(j);
        }
        const double elapsed = fast_shared::now_sec() - t0;

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
