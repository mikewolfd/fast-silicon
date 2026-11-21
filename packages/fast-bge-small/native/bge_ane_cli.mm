#import <CoreFoundation/CoreFoundation.h>
#include <arm_neon.h>
#include <zlib.h>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <libgen.h>
#include <mach-o/dyld.h>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "bge_ane_runner.h"

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

// Read whole file into memory.
bool read_file_bytes(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) return false;
    std::streamsize sz = ifs.tellg();
    if (sz < 0) return false;
    ifs.seekg(0, std::ios::beg);
    out.resize(static_cast<size_t>(sz));
    return ifs.read(reinterpret_cast<char*>(out.data()), sz).good();
}

// Minimal NPZ (ZIP + .npy) extractor supporting method 0 (stored) and method 8 (deflate).
bool extract_npz_entry(const std::vector<uint8_t>& zip, const std::string& entry_name, std::vector<uint8_t>& out) {
    if (zip.size() < 22) return false;
    // Locate EOCD
    size_t min_off = (zip.size() > 65557) ? (zip.size() - 65557) : 0;
    size_t eocd = std::string::npos;
    for (size_t i = zip.size() - 22; ; --i) {
        if (zip[i] == 0x50 && zip[i + 1] == 0x4b && zip[i + 2] == 0x05 && zip[i + 3] == 0x06) {
            eocd = i;
            break;
        }
        if (i == min_off) break;
    }
    if (eocd == std::string::npos) return false;
    uint32_t cd_offset = rd32(zip.data() + eocd + 16);
    uint16_t num_entries = rd16(zip.data() + eocd + 10);
    size_t p = cd_offset;
    for (uint16_t i = 0; i < num_entries; ++i) {
        if (p + 46 > zip.size() || rd32(zip.data() + p) != 0x02014b50u) return false;
        uint16_t method = rd16(zip.data() + p + 10);
        uint32_t comp_sz = rd32(zip.data() + p + 20);
        uint32_t uncomp_sz = rd32(zip.data() + p + 24);
        uint16_t fn_len = rd16(zip.data() + p + 28);
        uint16_t ex_len = rd16(zip.data() + p + 30);
        uint16_t cm_len = rd16(zip.data() + p + 32);
        uint32_t lfh_off = rd32(zip.data() + p + 42);
        std::string name(reinterpret_cast<const char*>(zip.data() + p + 46), fn_len);
        p += 46 + fn_len + ex_len + cm_len;
        if (name != entry_name) continue;
        if (lfh_off + 30 > zip.size() || rd32(zip.data() + lfh_off) != 0x04034b50u) return false;
        uint16_t l_fn = rd16(zip.data() + lfh_off + 26);
        uint16_t l_ex = rd16(zip.data() + lfh_off + 28);
        size_t data_off = static_cast<size_t>(lfh_off) + 30 + l_fn + l_ex;
        if (data_off + comp_sz > zip.size()) return false;
        if (method == 0) {
            out.assign(zip.begin() + data_off, zip.begin() + data_off + comp_sz);
            return true;
        }
        if (method == 8) {
            out.resize(uncomp_sz);
            z_stream zs{};
            if (inflateInit2(&zs, -MAX_WBITS) != Z_OK) return false;
            zs.next_in = const_cast<Bytef*>(zip.data() + data_off);
            zs.avail_in = comp_sz;
            zs.next_out = out.data();
            zs.avail_out = uncomp_sz;
            int rc = inflate(&zs, Z_FINISH);
            inflateEnd(&zs);
            return rc == Z_STREAM_END;
        }
        return false;
    }
    return false;
}

bool parse_npy_f16(const std::vector<uint8_t>& npy, std::vector<uint16_t>& out, size_t& elem_count) {
    if (npy.size() < 12 || memcmp(npy.data(), "\x93NUMPY", 6) != 0) return false;
    uint8_t major = npy[6];
    size_t hdr_len = 0;
    size_t hdr_off = 0;
    if (major == 1) {
        hdr_len = rd16(npy.data() + 8);
        hdr_off = 10;
    } else if (major == 2 || major == 3) {
        hdr_len = rd32(npy.data() + 8);
        hdr_off = 12;
    } else {
        return false;
    }
    size_t data_off = hdr_off + hdr_len;
    if (data_off > npy.size()) return false;
    size_t bytes = npy.size() - data_off;
    elem_count = bytes / 2;
    out.resize(elem_count);
    memcpy(out.data(), npy.data() + data_off, elem_count * 2);
    return true;
}

struct WordPieceTokenizer {
    std::unordered_map<std::string, int32_t> vocab;
    int32_t cls_id = 101;
    int32_t sep_id = 102;
    int32_t unk_id = 100;

    bool load(const std::string& vocab_path) {
        std::ifstream ifs(vocab_path);
        if (!ifs) return false;
        std::string line;
        int32_t idx = 0;
        while (std::getline(ifs, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            vocab[line] = idx++;
        }
        if (auto it = vocab.find("[CLS]"); it != vocab.end()) cls_id = it->second;
        if (auto it = vocab.find("[SEP]"); it != vocab.end()) sep_id = it->second;
        if (auto it = vocab.find("[UNK]"); it != vocab.end()) unk_id = it->second;
        return !vocab.empty();
    }

    static bool is_punct(uint32_t cp) {
        if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
            (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) return true;
        return false;
    }

    static std::string normalize_lowercase_strip_accents(const std::string& input) {
        CFStringRef s = CFStringCreateWithBytes(
            kCFAllocatorDefault,
            reinterpret_cast<const UInt8*>(input.data()),
            static_cast<CFIndex>(input.size()),
            kCFStringEncodingUTF8,
            false
        );
        if (!s) return input;
        CFMutableStringRef ms = CFStringCreateMutableCopy(kCFAllocatorDefault, 0, s);
        CFRelease(s);
        if (!ms) return input;
        CFStringTransform(ms, nullptr, kCFStringTransformStripCombiningMarks, false);
        CFStringLowercase(ms, CFLocaleGetSystem());
        CFIndex len = CFStringGetLength(ms);
        CFIndex max_bytes = CFStringGetMaximumSizeForEncoding(len, kCFStringEncodingUTF8) + 1;
        std::string out(static_cast<size_t>(max_bytes), '\0');
        CFIndex used = 0;
        CFStringGetBytes(
            ms, CFRangeMake(0, len), kCFStringEncodingUTF8, 0, false,
            reinterpret_cast<UInt8*>(out.data()), max_bytes, &used
        );
        CFRelease(ms);
        out.resize(static_cast<size_t>(used));
        return out;
    }

    std::vector<int32_t> encode(const std::string& text, int max_len = 512) const {
        std::string norm = normalize_lowercase_strip_accents(text);
        std::vector<std::string> basic_tokens;
        std::string cur;
        for (size_t i = 0; i < norm.size();) {
            unsigned char b = static_cast<unsigned char>(norm[i]);
            if (b < 0x80) {
                if (std::isspace(b)) {
                    if (!cur.empty()) { basic_tokens.push_back(std::move(cur)); cur.clear(); }
                } else if (is_punct(b)) {
                    if (!cur.empty()) { basic_tokens.push_back(std::move(cur)); cur.clear(); }
                    basic_tokens.emplace_back(1, static_cast<char>(b));
                } else {
                    cur.push_back(static_cast<char>(b));
                }
                ++i;
            } else {
                // Keep multi-byte UTF-8 sequence intact in cur
                size_t clen = (b < 0xE0) ? 2 : (b < 0xF0 ? 3 : 4);
                clen = std::min(clen, norm.size() - i);
                cur.append(norm, i, clen);
                i += clen;
            }
        }
        if (!cur.empty()) basic_tokens.push_back(std::move(cur));

        std::vector<int32_t> ids;
        ids.reserve(static_cast<size_t>(max_len));
        ids.push_back(cls_id);
        for (const auto& tok : basic_tokens) {
            if (static_cast<int>(ids.size()) >= max_len - 1) break;
            if (tok.size() > 200) {
                ids.push_back(unk_id);
                continue;
            }
            size_t start = 0;
            bool bad = false;
            std::vector<int32_t> sub_ids;
            while (start < tok.size()) {
                size_t end = tok.size();
                int32_t found = -1;
                while (start < end) {
                    std::string substr = (start > 0) ? ("##" + tok.substr(start, end - start)) : tok.substr(start, end - start);
                    auto it = vocab.find(substr);
                    if (it != vocab.end()) {
                        found = it->second;
                        break;
                    }
                    // Step back by 1 UTF-8 codepoint
                    --end;
                    while (end > start && (static_cast<unsigned char>(tok[end]) & 0xC0) == 0x80) {
                        --end;
                    }
                }
                if (found < 0) {
                    bad = true;
                    break;
                }
                sub_ids.push_back(found);
                start = end;
            }
            if (bad) {
                ids.push_back(unk_id);
            } else {
                for (int32_t sid : sub_ids) {
                    if (static_cast<int>(ids.size()) >= max_len - 1) break;
                    ids.push_back(sid);
                }
            }
        }
        ids.push_back(sep_id);
        return ids;
    }
};

std::string default_home_dir() {
    if (const char* env = std::getenv("FAST_BGE_SMALL_HOME")) {
        if (env[0] != '\0') return std::string(env);
    }
    namespace fs = std::filesystem;
    const std::string cache_name = "fast-bge-small";
    fs::path pkg_dir;
    uint32_t sz = 0;
    _NSGetExecutablePath(nullptr, &sz);
    std::string exe(sz, '\0');
    if (_NSGetExecutablePath(exe.data(), &sz) == 0) {
        char real_buf[4096];
        if (realpath(exe.c_str(), real_buf)) {
            pkg_dir = fs::path(real_buf).parent_path().parent_path();
        } else {
            pkg_dir = fs::path(exe.c_str()).parent_path().parent_path();
        }
        if (fs::exists(pkg_dir / "artifacts")) {
            return pkg_dir.string();
        }
    }
    if (const char* art_env = std::getenv("FAST_ARTIFACTS_ROOT")) {
        if (art_env[0] != '\0') {
            fs::path art_root(art_env);
            if (fs::exists(art_root / cache_name / "artifacts") || !fs::exists(art_root / "artifacts")) {
                return (art_root / cache_name).string();
            }
            return art_root.string();
        }
    }
    if (!pkg_dir.empty()) {
        std::vector<fs::path> sibs;
        if (pkg_dir.parent_path().filename() == "packages") {
            sibs.push_back(pkg_dir.parent_path().parent_path().parent_path() / cache_name);
        }
        if (pkg_dir.filename() != cache_name) {
            sibs.push_back(pkg_dir.parent_path() / cache_name);
        }
        for (const auto& sib : sibs) {
            if (fs::exists(sib / "artifacts")) {
                return sib.string();
            }
        }
    }
    if (const char* home_env = std::getenv("HOME")) {
        if (home_env[0] != '\0') {
            fs::path cache_dir = fs::path(home_env) / ".cache" / cache_name;
            if (fs::exists(cache_dir / "artifacts") || pkg_dir.empty()) {
                return cache_dir.string();
            }
        }
    }
    return pkg_dir.empty() ? "." : pkg_dir.string();
}

void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --text TEXT             Text to embed on Apple Neural Engine (WordPiece uncased)\n"
        "  --query-prefix          Prepend BGE retrieval query prompt before --text\n"
        "  --tokens ID1,ID2,...    Comma-separated token IDs (bypass tokenizer)\n"
        "  --bucket N              Force sequence bucket (24,64,128,192,320,384,448,512) or auto\n"
        "  --bench N               Run N sequential ANE single-query iterations and report latency\n"
        "  --home DIR              Override FAST_BGE_SMALL_HOME root (contains artifacts/)\n"
        "  --json                  Emit JSON object instead of human-readable summary\n",
        prog
    );
}

}  // namespace

int main(int argc, char** argv) {
    std::string home = default_home_dir();
    std::string text;
    bool has_text = false;
    bool query_prefix = false;
    bool json_out = false;
    int forced_bucket = 0;
    int bench_iters = 0;
    std::vector<int32_t> tokens;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--text" && i + 1 < argc) {
            text = argv[++i];
            has_text = true;
        } else if (arg == "--tokens" && i + 1 < argc) {
            std::stringstream ss(argv[++i]);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) tokens.push_back(std::stoi(item));
            }
        } else if (arg == "--bucket" && i + 1 < argc) {
            std::string v = argv[++i];
            forced_bucket = (v == "auto") ? 0 : std::stoi(v);
        } else if (arg == "--bench" && i + 1 < argc) {
            bench_iters = std::stoi(argv[++i]);
        } else if (arg == "--home" && i + 1 < argc) {
            home = argv[++i];
        } else if (arg == "--query-prefix") {
            query_prefix = true;
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

    if (!has_text && tokens.empty()) {
        if (bench_iters > 0) {
            tokens = {101, 2054, 2003, 1996, 15068, 2291, 1029, 102};
        } else {
            print_usage(argv[0]);
            return 2;
        }
    }

    std::string npz_path = home + "/artifacts/weights/bge_small_f16.npz";
    std::string vocab_path = home + "/artifacts/tokenizer/vocab.txt";
    std::string ane_dir = home + "/artifacts/ane_models";

    std::vector<uint8_t> zip_bytes, npy_word, npy_type, npy_pos;
    if (!read_file_bytes(npz_path, zip_bytes)) {
        fprintf(stderr, "Failed to read weights: %s\n", npz_path.c_str());
        return 1;
    }
    if (!extract_npz_entry(zip_bytes, "embeddings.word_embeddings.weight.npy", npy_word) ||
        !extract_npz_entry(zip_bytes, "embeddings.token_type_embeddings.weight.npy", npy_type) ||
        !extract_npz_entry(zip_bytes, "embeddings.position_embeddings.weight.npy", npy_pos)) {
        fprintf(stderr, "Missing expected embedding entries in %s\n", npz_path.c_str());
        return 1;
    }

    std::vector<uint16_t> word_f16, type_f16, pos_f16;
    size_t n_word = 0, n_type = 0, n_pos = 0;
    if (!parse_npy_f16(npy_word, word_f16, n_word) ||
        !parse_npy_f16(npy_type, type_f16, n_type) ||
        !parse_npy_f16(npy_pos, pos_f16, n_pos)) {
        fprintf(stderr, "Failed to parse float16 .npy arrays in %s\n", npz_path.c_str());
        return 1;
    }
    int vocab_size = static_cast<int>(n_word / 384);
    int max_pos = static_cast<int>(n_pos / 384);

    // Precompute wt0 = word_embeddings + token_type_embeddings[0] in FP16
    std::vector<uint16_t> wt0_f16(n_word);
    const float16_t* t0 = reinterpret_cast<const float16_t*>(type_f16.data());
    for (int v = 0; v < vocab_size; ++v) {
        const float16_t* w = reinterpret_cast<const float16_t*>(word_f16.data() + static_cast<size_t>(v) * 384);
        float16_t* dst = reinterpret_cast<float16_t*>(wt0_f16.data() + static_cast<size_t>(v) * 384);
        for (int c = 0; c < 384; c += 8) {
            vst1q_f16(dst + c, vaddq_f16(vld1q_f16(w + c), vld1q_f16(t0 + c)));
        }
    }

    if (bge_ane_init_tables(wt0_f16.data(), vocab_size, pos_f16.data(), max_pos) != 0) {
        fprintf(stderr, "Failed to initialize ANE embedding tables\n");
        return 1;
    }

    if (has_text) {
        WordPieceTokenizer tok;
        if (!tok.load(vocab_path)) {
            fprintf(stderr, "Failed to load vocab: %s\n", vocab_path.c_str());
            return 1;
        }
        std::string full = query_prefix ? ("Represent this sentence for searching relevant passages: " + text) : text;
        tokens = tok.encode(full, 512);
    }

    const int sl = static_cast<int>(tokens.size());
    const int caps[] = {24, 64, 128, 192, 256, 320, 384, 448, 512};
    std::string mf_path = ane_dir + "/bge_ane_w8a16.mlmodelc";
    bool has_mf = std::ifstream(mf_path + "/coremldata.bin").good();
    int chosen_cap = 512;
    if (forced_bucket > 0) {
        chosen_cap = forced_bucket;
    } else {
        for (int c : caps) {
            if (sl <= c) {
                if (c == 256 && !has_mf) {
                    // Use optional B=1 S=256 bundle if present; otherwise advance to 320.
                    std::string p256_w8 = ane_dir + "/bge_b1_s256_w8a16.mlmodelc/coremldata.bin";
                    std::string p256_f16 = ane_dir + "/bge_b1_s256_fp16.mlmodelc/coremldata.bin";
                    if (!std::ifstream(p256_w8).good() && !std::ifstream(p256_f16).good()) continue;
                }
                chosen_cap = c;
                break;
            }
        }
    }

    int bid = -1;
    if (has_mf) {
        char fn_name[64];
        snprintf(fn_name, sizeof(fn_name), "bge_b1_s%d", chosen_cap);
        bid = bge_ane_load_bucket_fn(mf_path.c_str(), fn_name, 1, chosen_cap, 1, 1);
    }
    if (bid < 0) {
        char model_path[2048];
        snprintf(model_path, sizeof(model_path), "%s/bge_b1_s%d_w8a16.mlmodelc", ane_dir.c_str(), chosen_cap);
        if (!std::ifstream(std::string(model_path) + "/coremldata.bin").good()) {
            snprintf(model_path, sizeof(model_path), "%s/bge_b1_s%d_fp16.mlmodelc", ane_dir.c_str(), chosen_cap);
        }
        bid = bge_ane_load_bucket(model_path, 1, chosen_cap, 1, 1);
        if (bid < 0) {
            fprintf(stderr, "Failed to load ANE bucket %s (err=%d: %s)\n", model_path, bid, bge_ane_last_error());
            return 1;
        }
    }

    float vec[384];
    int iters = (bench_iters > 0) ? bench_iters : 1;
    double elapsed_s = bge_ane_query(bid, tokens.data(), sl, vec, iters);
    if (elapsed_s < 0.0) {
        fprintf(stderr, "bge_ane_query failed (rc=%.1f: %s)\n", elapsed_s, bge_ane_last_error());
        bge_ane_free_all();
        return 1;
    }
    double ms_per_call = (elapsed_s / iters) * 1e3;

    if (json_out) {
        printf("{\"tokens\": %d, \"bucket\": %d, \"iters\": %d, \"latency_ms\": %.4f, \"embedding\": [",
               sl, chosen_cap, iters, ms_per_call);
        for (int i = 0; i < 384; ++i) {
            printf("%s%.7g", (i == 0 ? "" : ", "), vec[i]);
        }
        printf("]}\n");
    } else {
        printf("tokens=%d bucket=S%d iters=%d latency_ms=%.3f vec[:6]=[%.5f, %.5f, %.5f, %.5f, %.5f, %.5f]\n",
               sl, chosen_cap, iters, ms_per_call, vec[0], vec[1], vec[2], vec[3], vec[4], vec[5]);
    }

    bge_ane_free_all();
    return 0;
}
