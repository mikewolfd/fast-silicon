#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "fast_shared_apple.hpp"
#include "gemma4_ane_prefix.h"

static void usage(const char* prog) {
    fprintf(stderr,
            "Usage: %s --model PATH.mlmodelc [--S 256] [--p-swa 512] [--p-full 1920] "
            "[--batch 16] [--pipe 4] [--iters 3] [--inspect] [--json]\n",
            prog);
}

int main(int argc, char** argv) {
    std::string model_path;
    int S = 256;
    int p_swa = 512;
    int p_full = 1920;
    int B = 16;
    int pipe = 4;
    int iters = 3;
    bool inspect = false;
    bool as_json = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--S") == 0 && i + 1 < argc) {
            S = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--p-swa") == 0 && i + 1 < argc) {
            p_swa = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--p-full") == 0 && i + 1 < argc) {
            p_full = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            B = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--pipe") == 0 && i + 1 < argc) {
            pipe = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--inspect") == 0) {
            inspect = true;
        } else if (strcmp(argv[i], "--json") == 0) {
            as_json = true;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (model_path.empty()) {
        usage(argv[0]);
        return 2;
    }

    void* h = gemma4_px_load(model_path.c_str(), S, p_swa, p_full, pipe);
    if (!h) {
        fprintf(stderr, "Failed to load CoreML model: %s\n", model_path.c_str());
        return 1;
    }
    const int zero_copy = gemma4_px_is_zero_copy(h);

    int ane_ops = -1;
    int cpu_runtime_ops = -1;
    int cpu_constexpr_ops = -1;
    int gpu_ops = -1;
    if (inspect) {
        ane_ops = gemma4_px_inspect_plan_ex(
            model_path.c_str(),
            static_cast<int>(fast_shared::ComputeMode::kCpuAndNeuralEngine),
            &ane_ops,
            &cpu_runtime_ops,
            &cpu_constexpr_ops,
            &gpu_ops
        );
        if (!as_json) {
            printf("[MLComputePlan] ANE=%d GPU=%d CPU_runtime=%d CPU_constexpr=%d\n",
                   ane_ops, gpu_ops, cpu_runtime_ops, cpu_constexpr_ops);
        }
    }

    int H_dim = 1536, C_pli = 3584, n_swa = 12, n_full_past = 2, n_full_out = 3, n_kv = 1;
    if (gemma4_px_get_dims(h, &H_dim, &C_pli, &n_swa, &n_full_past, &n_full_out, &n_kv) != 0) {
        fprintf(stderr, "Failed to query dimensions for CoreML model: %s\n", model_path.c_str());
        gemma4_px_free(h);
        return 1;
    }

    std::vector<uint16_t> h_in(static_cast<size_t>(B) * H_dim * S, 0x3800);
    std::vector<uint16_t> pli_in(static_cast<size_t>(B) * C_pli * S, 0x3800);
    std::vector<uint16_t> cos_s(static_cast<size_t>(S) * 256, 0x3C00);
    std::vector<uint16_t> sin_s(static_cast<size_t>(S) * 256, 0);
    std::vector<uint16_t> cos_f(static_cast<size_t>(S) * 512, 0x3C00);
    std::vector<uint16_t> sin_f(static_cast<size_t>(S) * 512, 0);
    std::vector<uint16_t> kp_s(static_cast<size_t>(n_swa) * n_kv * p_swa * 256, 0);
    std::vector<uint16_t> vp_s(static_cast<size_t>(n_swa) * n_kv * p_swa * 256, 0);
    std::vector<uint16_t> kp_f(static_cast<size_t>(n_full_past) * n_kv * p_full * 512, 0);
    std::vector<uint16_t> vp_f(static_cast<size_t>(n_full_past) * n_kv * p_full * 512, 0);
    std::vector<uint16_t> ms(static_cast<size_t>(S) * (p_swa + S), 0);
    std::vector<uint16_t> mf(static_cast<size_t>(S) * (p_full + S), 0);

    std::vector<uint16_t> ok_s(static_cast<size_t>(B) * n_swa * n_kv * S * 256);
    std::vector<uint16_t> ov_s(static_cast<size_t>(B) * n_swa * n_kv * S * 256);
    std::vector<uint16_t> ok_f(static_cast<size_t>(B) * n_full_out * n_kv * S * 512);
    std::vector<uint16_t> ov_f(static_cast<size_t>(B) * n_full_out * n_kv * S * 512);

    double best_sec = 1e9;
    double last_sec = -1.0;
    for (int it = 0; it < iters; ++it) {
        double dt = gemma4_px_run_batch(
            h, B,
            h_in.data(), pli_in.data(),
            cos_s.data(), sin_s.data(), cos_f.data(), sin_f.data(),
            kp_s.data(), vp_s.data(), kp_f.data(), vp_f.data(),
            ms.data(), mf.data(),
            ok_s.data(), ov_s.data(), ok_f.data(), ov_f.data(),
            1, 1, pipe
        );
        if (dt < 0) {
            fprintf(stderr, "gemma4_px_run_batch failed on iter %d\n", it);
            gemma4_px_free(h);
            return 1;
        }
        last_sec = dt;
        if (dt < best_sec) best_sec = dt;
    }

    const double tok_s = static_cast<double>(B * S) / best_sec;
    const double ms_per_chunk = (best_sec / B) * 1e3;
    if (as_json) {
        printf("{\"S\":%d,\"batch\":%d,\"pipe\":%d,\"ane_ops\":%d,\"cpu_runtime_ops\":%d,"
               "\"cpu_constexpr_ops\":%d,\"gpu_ops\":%d,\"zero_copy\":%d,\"best_sec\":%.6f,"
               "\"last_sec\":%.6f,\"ms_per_chunk\":%.3f,\"tok_per_sec\":%.1f}\n",
               S, B, pipe, ane_ops, cpu_runtime_ops, cpu_constexpr_ops, gpu_ops, zero_copy,
               best_sec, last_sec, ms_per_chunk, tok_s);
    } else {
        printf("S=%d batch=%d pipe=%d zero_copy=%d | best=%.4fs (%.2f ms/chunk, %.0f prefill tok/s)\n",
               S, B, pipe, zero_copy, best_sec, ms_per_chunk, tok_s);
    }

    gemma4_px_free(h);
    return 0;
}
