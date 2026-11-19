#include <arm_neon.h>
#include <libgen.h>
#include <mach-o/dyld.h>
#include <zlib.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "emb2_coreml_runner.h"
#include "emb2_fast_io.h"
#include "fast_shared_apple.hpp"

namespace {

static inline uint16_t rd16(const uint8_t* p) {
    uint16_t v;
    memcpy(&v, p, 2);
    return v;
}

static inline uint32_t rd32(const uint8_t* p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

// Seek-based NPZ entry reader: reads only the target entry without buffering the whole archive.
bool read_npz_entry(std::ifstream& ifs, size_t file_size, const std::string& entry_name, std::vector<uint8_t>& out) {
    if (file_size < 22) return false;
    const size_t tail_sz = std::min<size_t>(file_size, 65557);
    std::vector<uint8_t> tail(tail_sz);
    ifs.seekg(static_cast<std::streamoff>(file_size - tail_sz), std::ios::beg);
    if (!ifs.read(reinterpret_cast<char*>(tail.data()), static_cast<std::streamsize>(tail_sz))) return false;

    size_t eocd_rel = std::string::npos;
    for (size_t i = tail_sz - 22; ; --i) {
        if (tail[i] == 0x50 && tail[i + 1] == 0x4b && tail[i + 2] == 0x05 && tail[i + 3] == 0x06) {
            eocd_rel = i;
            break;
        }
        if (i == 0) break;
    }
    if (eocd_rel == std::string::npos) return false;

    const uint16_t num_entries = rd16(tail.data() + eocd_rel + 10);
    const uint32_t cd_size = rd32(tail.data() + eocd_rel + 12);
    const uint32_t cd_offset = rd32(tail.data() + eocd_rel + 16);
    if (static_cast<size_t>(cd_offset) + cd_size > file_size) return false;

    std::vector<uint8_t> cd(cd_size);
    ifs.seekg(static_cast<std::streamoff>(cd_offset), std::ios::beg);
    if (!ifs.read(reinterpret_cast<char*>(cd.data()), static_cast<std::streamsize>(cd_size))) return false;

    size_t p = 0;
    for (uint16_t i = 0; i < num_entries; ++i) {
        if (p + 46 > cd.size() || rd32(cd.data() + p) != 0x02014b50u) return false;
        const uint16_t method = rd16(cd.data() + p + 10);
        const uint32_t comp_sz = rd32(cd.data() + p + 20);
        const uint32_t uncomp_sz = rd32(cd.data() + p + 24);
        const uint16_t fn_len = rd16(cd.data() + p + 28);
        const uint16_t ex_len = rd16(cd.data() + p + 30);
        const uint16_t cm_len = rd16(cd.data() + p + 32);
        const uint32_t lfh_off = rd32(cd.data() + p + 42);
        std::string name(reinterpret_cast<const char*>(cd.data() + p + 46), fn_len);
        p += 46 + fn_len + ex_len + cm_len;
        if (name != entry_name) continue;

        uint8_t lfh[30];
        ifs.seekg(static_cast<std::streamoff>(lfh_off), std::ios::beg);
        if (!ifs.read(reinterpret_cast<char*>(lfh), 30) || rd32(lfh) != 0x04034b50u) return false;
        const uint16_t l_fn = rd16(lfh + 26);
        const uint16_t l_ex = rd16(lfh + 28);
        const size_t data_off = static_cast<size_t>(lfh_off) + 30 + l_fn + l_ex;
        if (data_off + comp_sz > file_size) return false;

        ifs.seekg(static_cast<std::streamoff>(data_off), std::ios::beg);
        if (method == 0) {
            out.resize(comp_sz);
            return ifs.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(comp_sz)).good();
        }
        if (method == 8) {
            std::vector<uint8_t> comp(comp_sz);
            if (!ifs.read(reinterpret_cast<char*>(comp.data()), static_cast<std::streamsize>(comp_sz))) return false;
            out.resize(uncomp_sz);
            z_stream zs{};
            if (inflateInit2(&zs, -MAX_WBITS) != Z_OK) return false;
            zs.next_in = comp.data();
            zs.avail_in = comp_sz;
            zs.next_out = out.data();
            zs.avail_out = uncomp_sz;
            const int rc = inflate(&zs, Z_FINISH);
            inflateEnd(&zs);
            return rc == Z_STREAM_END;
        }
        return false;
    }
    return false;
}

bool parse_npy_f16(const std::vector<uint8_t>& npy, std::vector<uint16_t>& out, size_t& n_elems) {
    if (npy.size() < 12 || memcmp(npy.data(), "\x93NUMPY", 6) != 0) return false;
    const uint8_t major = npy[6];
    size_t data_off = 0;
    if (major == 1) {
        data_off = 10 + rd16(npy.data() + 8);
    } else if (major >= 2) {
        data_off = 12 + rd32(npy.data() + 8);
    } else {
        return false;
    }
    if (data_off > npy.size()) return false;
    const size_t payload_bytes = npy.size() - data_off;
    n_elems = payload_bytes / sizeof(uint16_t);
    out.resize(n_elems);
    memcpy(out.data(), npy.data() + data_off, n_elems * sizeof(uint16_t));
    return true;
}

std::string default_home_dir() {
    if (const char* env = std::getenv("FAST_EMBEDDINGGEMMA2_HOME")) {
        if (env[0] != '\0') return std::string(env);
    }
    char exe_buf[4096];
    uint32_t sz = sizeof(exe_buf);
    if (_NSGetExecutablePath(exe_buf, &sz) == 0) {
        char real_buf[4096];
        if (realpath(exe_buf, real_buf)) {
            char* d1 = dirname(real_buf);
            std::string dir(d1);
            char* d2 = dirname(const_cast<char*>(dir.c_str()));
            if (d2) return std::string(d2);
        }
    }
    return ".";
}

void build_rope_tables_f16(
    int max_l,
    std::vector<uint16_t>& c256,
    std::vector<uint16_t>& s256,
    std::vector<uint16_t>& c512,
    std::vector<uint16_t>& s512
) {
    c256.resize(static_cast<size_t>(max_l) * 128);
    s256.resize(static_cast<size_t>(max_l) * 128);
    c512.resize(static_cast<size_t>(max_l) * 256);
    s512.resize(static_cast<size_t>(max_l) * 256);

    double inv256[128], inv512[256];
    for (int i = 0; i < 128; ++i) inv256[i] = std::pow(10000.0, -static_cast<double>(2 * i) / 256.0);
    for (int i = 0; i < 256; ++i) inv512[i] = std::pow(1000000.0, -static_cast<double>(2 * i) / 512.0);

    for (int p = 0; p < max_l; ++p) {
        auto* c2 = reinterpret_cast<float16_t*>(c256.data() + static_cast<size_t>(p) * 128);
        auto* s2 = reinterpret_cast<float16_t*>(s256.data() + static_cast<size_t>(p) * 128);
        for (int i = 0; i < 128; ++i) {
            const double ang = static_cast<double>(p) * inv256[i];
            c2[i] = static_cast<float16_t>(static_cast<float>(std::cos(ang)));
            s2[i] = static_cast<float16_t>(static_cast<float>(std::sin(ang)));
        }
        auto* c5 = reinterpret_cast<float16_t*>(c512.data() + static_cast<size_t>(p) * 256);
        auto* s5 = reinterpret_cast<float16_t*>(s512.data() + static_cast<size_t>(p) * 256);
        for (int i = 0; i < 256; ++i) {
            const double ang = static_cast<double>(p) * inv512[i];
            c5[i] = static_cast<float16_t>(static_cast<float>(std::cos(ang)));
            s5[i] = static_cast<float16_t>(static_cast<float>(std::sin(ang)));
        }
    }
}

void print_usage(const char* prog) {
    fprintf(stderr,
            "Usage: %s [--ids \"2,3250,...,1\"] [--seq-len 32|128|256|384|512] [--bench N] [--pipe-depth D] [--compute ane|cpu|all] [--home DIR] [--json]\n",
            prog);
}

}  // namespace

int main(int argc, char** argv) {
    std::string home = default_home_dir();
    std::vector<int32_t> tokens = {2, 3250, 236787, 7293, 1109, 1816, 236787, 1};
    int forced_seq = 0;
    int bench_iters = 0;
    int pipe_depth = 1;
    int compute_mode = static_cast<int>(fast_shared::ComputeMode::kCpuAndNeuralEngine);
    bool json_out = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--home" && i + 1 < argc) {
            home = argv[++i];
        } else if (arg == "--ids" && i + 1 < argc) {
            tokens.clear();
            std::stringstream ss(argv[++i]);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) tokens.push_back(std::stoi(item));
            }
            if (tokens.empty()) tokens = {2, 1};
        } else if (arg == "--seq-len" && i + 1 < argc) {
            forced_seq = std::stoi(argv[++i]);
        } else if (arg == "--bench" && i + 1 < argc) {
            bench_iters = std::stoi(argv[++i]);
        } else if (arg == "--pipe-depth" && i + 1 < argc) {
            pipe_depth = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--compute" && i + 1 < argc) {
            std::string cm = argv[++i];
            if (cm == "ane") compute_mode = static_cast<int>(fast_shared::ComputeMode::kCpuAndNeuralEngine);
            else if (cm == "cpu") compute_mode = static_cast<int>(fast_shared::ComputeMode::kCpuOnly);
            else if (cm == "all") compute_mode = static_cast<int>(fast_shared::ComputeMode::kAll);
            else {
                fprintf(stderr, "Invalid --compute: %s\n", cm.c_str());
                return 2;
            }
        } else if (arg == "--json") {
            json_out = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 2;
        }
    }

    const int sl = std::min<int>(static_cast<int>(tokens.size()), 512);
    const int ane_caps[] = {32, 128, 256, 384, 512};
    int chosen_s = 512;
    if (forced_seq > 0) {
        chosen_s = forced_seq;
    } else {
        for (int c : ane_caps) {
            if (sl <= c) {
                chosen_s = c;
                break;
            }
        }
    }

    const std::string npz_path = home + "/artifacts/weights/embeddinggemma_2_f16.npz";
    const std::string ane_dir = home + "/artifacts/ane_models";
    const std::string model_path = (compute_mode == static_cast<int>(fast_shared::ComputeMode::kCpuOnly))
        ? (ane_dir + "/emb2_cpu_s" + std::to_string(chosen_s) + "_mix_w8.mlmodelc")
        : (ane_dir + "/emb2_pk_s" + std::to_string(chosen_s) + "_w8.mlmodelc");

    std::ifstream ifs(npz_path, std::ios::binary | std::ios::ate);
    if (!ifs) {
        fprintf(stderr, "Failed to open weights: %s\n", npz_path.c_str());
        return 1;
    }
    const size_t file_size = static_cast<size_t>(ifs.tellg());
    std::vector<uint8_t> npy_emb, npy_proj;
    if (!read_npz_entry(ifs, file_size, "embed_tokens.npy", npy_emb) ||
        !read_npz_entry(ifs, file_size, "emb_proj.npy", npy_proj)) {
        fprintf(stderr, "Missing embed_tokens.npy or emb_proj.npy in %s\n", npz_path.c_str());
        return 1;
    }

    std::vector<uint16_t> emb_f16, proj_f16;
    size_t n_emb = 0, n_proj = 0;
    if (!parse_npy_f16(npy_emb, emb_f16, n_emb) || !parse_npy_f16(npy_proj, proj_f16, n_proj)) {
        fprintf(stderr, "Failed to parse float16 arrays from %s\n", npz_path.c_str());
        return 1;
    }
    if (n_proj != 768 * 512) {
        fprintf(stderr, "Unexpected emb_proj size: %zu\n", n_proj);
        return 1;
    }

    // Scale embed_tokens by sqrt(512) in place via NEON float32 multiply
    const float emb_scale = std::sqrt(512.0f);
    const float32x4_t v_sc = vdupq_n_f32(emb_scale);
    auto* emb_ptr = reinterpret_cast<float16_t*>(emb_f16.data());
    for (size_t i = 0; i + 8 <= n_emb; i += 8) {
        const float16x8_t h = vld1q_f16(emb_ptr + i);
        const float32x4_t lo = vmulq_f32(vcvt_f32_f16(vget_low_f16(h)), v_sc);
        const float32x4_t hi = vmulq_f32(vcvt_f32_f16(vget_high_f16(h)), v_sc);
        vst1q_f16(emb_ptr + i, vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi)));
    }

    // Transpose emb_proj [768, 512] f16 -> emb_proj_T [512, 768] f32
    std::vector<float> emb_proj_T(512 * 768);
    const auto* proj_ptr = reinterpret_cast<const float16_t*>(proj_f16.data());
    for (int out_c = 0; out_c < 768; ++out_c) {
        for (int in_c = 0; in_c < 512; ++in_c) {
            emb_proj_T[static_cast<size_t>(in_c) * 768 + out_c] = static_cast<float>(proj_ptr[static_cast<size_t>(out_c) * 512 + in_c]);
        }
    }

    std::vector<uint16_t> c256, s256, c512, s512;
    build_rope_tables_f16(chosen_s, c256, s256, c512, s512);

    void* handle = emb2_pk_load_units(model_path.c_str(), chosen_s, pipe_depth, compute_mode);
    if (!handle) {
        fprintf(stderr, "Failed to load CoreML model: %s\n", model_path.c_str());
        return 1;
    }

    const int iters = (bench_iters > 0) ? bench_iters : 1;
    std::vector<int32_t> ids(static_cast<size_t>(iters) * chosen_s, -1);
    std::vector<int32_t> pos(static_cast<size_t>(iters) * chosen_s, 0);
    std::vector<int32_t> span_offsets(iters + 1, 0);
    std::vector<int32_t> spans_se(static_cast<size_t>(iters) * 2);
    std::vector<int32_t> doc_lane_se(static_cast<size_t>(iters) * 3);

    for (int it = 0; it < iters; ++it) {
        for (int k = 0; k < sl; ++k) {
            ids[static_cast<size_t>(it) * chosen_s + k] = tokens[k];
            pos[static_cast<size_t>(it) * chosen_s + k] = k;
        }
        span_offsets[it] = it;
        spans_se[it * 2] = 0;
        spans_se[it * 2 + 1] = sl;
        doc_lane_se[it * 3] = it;
        doc_lane_se[it * 3 + 1] = 0;
        doc_lane_se[it * 3 + 2] = sl;
    }
    span_offsets[iters] = iters;

    std::vector<uint16_t> h_all(static_cast<size_t>(iters) * 512 * chosen_s);
    std::vector<uint16_t> b_all(static_cast<size_t>(iters) * chosen_s * chosen_s);
    std::vector<uint16_t> cs_all(static_cast<size_t>(iters) * chosen_s * 128);
    std::vector<uint16_t> ss_all(static_cast<size_t>(iters) * chosen_s * 128);
    std::vector<uint16_t> cf_all(static_cast<size_t>(iters) * chosen_s * 256);
    std::vector<uint16_t> sf_all(static_cast<size_t>(iters) * chosen_s * 256);
    std::vector<uint16_t> out_all(static_cast<size_t>(iters) * chosen_s * 512);
    std::vector<float> scratch_pooled(static_cast<size_t>(iters) * 512);
    std::vector<float> out_embs(static_cast<size_t>(iters) * 768);

    const float16_t neg_inf_val = static_cast<float16_t>(-10000.0f);
    uint16_t neg_inf_u16 = 0;
    memcpy(&neg_inf_u16, &neg_inf_val, 2);

    const double t0 = fast_shared::now_sec();
    emb2_prep_batch(
        iters, chosen_s,
        emb_f16.data(), c256.data(), s256.data(), c512.data(), s512.data(),
        ids.data(), pos.data(), span_offsets.data(), spans_se.data(), neg_inf_u16,
        h_all.data(), b_all.data(), cs_all.data(), ss_all.data(), cf_all.data(), sf_all.data()
    );
    emb2_pk_exec_queue(
        handle, iters,
        h_all.data(), b_all.data(), cs_all.data(), ss_all.data(), cf_all.data(), sf_all.data(),
        out_all.data()
    );
    emb2_pool_project_batch(
        chosen_s, out_all.data(), iters, doc_lane_se.data(),
        emb_proj_T.data(), scratch_pooled.data(), out_embs.data()
    );
    const double total_ms = (fast_shared::now_sec() - t0) * 1e3;
    const double ms_per_call = total_ms / iters;

    if (json_out) {
        printf("{\"tokens\": %d, \"bucket\": %d, \"iters\": %d, \"pipe_depth\": %d, \"latency_ms\": %.4f, \"embedding\": [",
               sl, chosen_s, iters, pipe_depth, ms_per_call);
        for (int c = 0; c < 768; ++c) {
            printf("%s%.7g", (c == 0 ? "" : ", "), out_embs[c]);
        }
        printf("]}\n");
    } else {
        printf("tokens=%d bucket=S%d iters=%d pipe=%d latency_ms=%.3f vec[:6]=[%.5f, %.5f, %.5f, %.5f, %.5f, %.5f]\n",
               sl, chosen_s, iters, pipe_depth, ms_per_call,
               out_embs[0], out_embs[1], out_embs[2], out_embs[3], out_embs[4], out_embs[5]);
    }

    emb2_pk_free(handle);
    return 0;
}
