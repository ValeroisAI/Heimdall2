#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr int depth = 32768;
constexpr int steps = 256;
constexpr int batch_size = 512;
constexpr int anchors = 6;
constexpr int history_steps = 64;

struct Capture {
    std::array<std::ofstream, anchors> files;
    std::array<int, anchors> counts{};
    std::array<std::ofstream, anchors> queries, keys, values, history;
    std::array<int, anchors> query_counts{};
    std::array<bool, anchors> saved_keys{}, saved_values{};
    std::array<bool, anchors> saved_history{};
    bool prefill_history = false;
    bool enabled = false;
    bool all_tensors = false;
    bool final_step = false;
    bool failed = false;
};

int parse_layer(std::string_view name, std::string_view prefix, int first) {
    if (name.substr(0, prefix.size()) != prefix) {
        return -1;
    }
    int layer = -1;
    const auto end = name.data() + name.size();
    auto [parsed, status] = std::from_chars(name.data() + prefix.size(), end, layer);
    if (status != std::errc{} || parsed != end || layer < first || (layer - first) % 4 != 0) {
        return -1;
    }
    int index = (layer - first) / 4;
    return index < anchors ? index : -1;
}

bool write_attention(Capture & capture, ggml_tensor * node, std::string_view name, bool ask) {
    int q = parse_layer(name, "Qcur-", 3);
    int k = parse_layer(name, "ret_k_archive-", 3);
    int v = parse_layer(name, "ret_v_archive-", 3);
    if (q < 0 && (!capture.final_step ||
                  !((k >= 0 && !capture.saved_keys[k]) || (v >= 0 && !capture.saved_values[v])))) {
        return !ask;
    }
    if (ask) {
        return true;
    }
    const bool correct_shape = q >= 0
        ? node->type == GGML_TYPE_F32 && node->ne[0] == 256 && node->ne[1] == 8 && node->ne[2] == 1
        : node->type == GGML_TYPE_F16 && node->ne[0] == 512 && node->ne[1] == depth + steps && node->ne[2] == 1;
    if (!correct_shape || node->ne[3] != 1 || !ggml_is_contiguous(node)) {
        std::fprintf(stderr, "Unexpected attention capture layout for %s\n", ggml_get_name(node));
        capture.failed = true;
        return false;
    }
    auto & file = q >= 0 ? capture.queries[q] : k >= 0 ? capture.keys[k] : capture.values[v];
    std::vector<char> bytes(ggml_nbytes(node));
    ggml_backend_tensor_get(node, bytes.data(), 0, bytes.size());
    file.write(bytes.data(), bytes.size());
    if (q >= 0) {
        ++capture.query_counts[q];
    } else if (k >= 0) {
        capture.saved_keys[k] = file.good();
    } else {
        capture.saved_values[v] = file.good();
    }
    capture.failed = !file;
    return file.good();
}

bool observe(ggml_tensor * node, bool ask, void * data) {
    auto & capture = *static_cast<Capture *>(data);
    if (capture.prefill_history) {
        const int anchor = parse_layer(ggml_get_name(node), "Qcur-", 3);
        if (anchor < 0 || capture.saved_history[anchor]) {
            return !ask;
        }
        if (ask) {
            return true;
        }
        constexpr size_t query_bytes = 256 * 8 * sizeof(float);
        if (node->type != GGML_TYPE_F32 || node->ne[0] != 256 || node->ne[1] != 8 ||
            node->ne[2] != batch_size || node->ne[3] != 1 || !ggml_is_contiguous(node)) {
            std::fprintf(stderr, "Unexpected prefill query layout for %s\n", ggml_get_name(node));
            capture.failed = true;
            return false;
        }
        std::vector<char> bytes(history_steps * query_bytes);
        ggml_backend_tensor_get(node, bytes.data(), (batch_size - history_steps) * query_bytes, bytes.size());
        auto & file = capture.history[anchor];
        file.write(bytes.data(), bytes.size());
        capture.saved_history[anchor] = file.good();
        capture.failed = !file;
        return !capture.failed;
    }
    if (!capture.enabled) {
        return !ask;
    }
    const char * name = ggml_get_name(node);
    const int index = parse_layer(name, "attn_output-", 2);
    if (index < 0) {
        return capture.all_tensors ? write_attention(capture, node, name, ask) : !ask;
    }
    if (ask) {
        return true;
    }
    if (node->type != GGML_TYPE_F32 || node->ne[2] != 1 || node->ne[3] != 1) {
        std::fprintf(stderr, "Unexpected GDN readout shape or type for %s\n", name);
        capture.failed = true;
        return false;
    }
    auto & file = capture.files[index];
    std::vector<char> bytes(ggml_nbytes(node));
    ggml_backend_tensor_get(node, bytes.data(), 0, bytes.size());
    for (int64_t head = 0; head < node->ne[1]; ++head) {
        for (int64_t d = 0; d < node->ne[0]; ++d) {
            file.write(bytes.data() + head * node->nb[1] + d * node->nb[0], sizeof(float));
        }
    }
    if (!file) {
        capture.failed = true;
        return false;
    }
    if (++capture.counts[index] == 1) {
        std::fprintf(stderr, "%s capture: %lld heads x %lld values\n", name,
                     static_cast<long long>(node->ne[1]), static_cast<long long>(node->ne[0]));
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    const bool capture_history = argc == 5 && std::strcmp(argv[3], "--capture-with-history") == 0;
    const bool capture_all = capture_history || (argc == 5 && std::strcmp(argv[3], "--capture") == 0);
    const bool capture_gdn = argc == 4 || capture_all;
    if (argc != 3 && !capture_gdn) {
        std::fprintf(stderr, "Usage: %s model.gguf corpus.txt [gdn-prefix | --capture cal-kind-prefix | "
                             "--capture-with-history cal-kind-prefix]\n", argv[0]);
        return 2;
    }
    if (capture_all) {
        const char * route_path = std::getenv("GJALLAR_DIAGNOSTIC_ROUTE");
        if (!route_path) {
            std::fprintf(stderr, "Set GJALLAR_DIAGNOSTIC_ROUTE to an all-full route before capturing Q/K/V.\n");
            return 2;
        }
        std::ifstream route_file(route_path, std::ios::binary);
        std::array<char, anchors * steps> route{};
        if (!route_file.read(route.data(), route.size()) || route_file.peek() != std::ifstream::traits_type::eof() ||
            !std::all_of(route.begin(), route.end(), [](char mode) { return mode == 1; })) {
            std::fprintf(stderr, "Capture requires a %d-byte all-full route.\n", anchors * steps);
            return 2;
        }
    }
    std::ifstream input(argv[2], std::ios::binary);
    std::string corpus((std::istreambuf_iterator<char>(input)), {});
    if (corpus.empty()) {
        std::fprintf(stderr, "Corpus is empty or unreadable.\n");
        return 2;
    }

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.split_mode = LLAMA_SPLIT_MODE_NONE;
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        return 2;
    }
    auto * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(corpus.size() + 16);
    int count = llama_tokenize(vocab, corpus.data(), static_cast<int>(corpus.size()), tokens.data(),
                               static_cast<int>(tokens.size()), false, false);
    if (count <= depth + steps) {
        std::fprintf(stderr, "Corpus must tokenize to at least %d tokens.\n", depth + steps + 1);
        return 2;
    }
    std::fprintf(stderr, "source_tokens=%d prefill=%d decode=%d\n", count, depth, steps);
    auto cp = llama_context_default_params();
    cp.n_ctx = depth + steps;
    cp.n_batch = cp.n_ubatch = batch_size;
    cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 8;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    Capture capture;
    capture.all_tensors = capture_all;
    if (capture_gdn) {
        const std::string prefix = argv[capture_all ? 4 : 3];
        for (int i = 0; i < anchors; ++i) {
            auto path = prefix + (capture_all ? "-gdn-out-" : "") + std::to_string(2 + 4 * i);
            capture.files[i].open(path, std::ios::binary);
            if (!capture.files[i]) {
                std::fprintf(stderr, "Unable to write %s\n", path.c_str());
                return 2;
            }
            if (capture_all) {
                auto layer = std::to_string(3 + 4 * i);
                capture.queries[i].open(prefix + "-q-" + layer, std::ios::binary);
                capture.keys[i].open(prefix + "-k-" + layer, std::ios::binary);
                capture.values[i].open(prefix + "-v-" + layer, std::ios::binary);
                if (!capture.queries[i] || !capture.keys[i] || !capture.values[i]) {
                    std::fprintf(stderr, "Unable to open attention capture files for layer %s\n", layer.c_str());
                    return 2;
                }
                if (capture_history) {
                    auto history_path = prefix + "-history-q-" + layer;
                    capture.history[i].open(history_path, std::ios::binary);
                    if (!capture.history[i]) {
                        std::fprintf(stderr, "Unable to write %s\n", history_path.c_str());
                        return 2;
                    }
                }
            }
        }
        cp.cb_eval = observe;
        cp.cb_eval_user_data = &capture;
    }
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        return 2;
    }

    auto batch = llama_batch_init(batch_size, 0, 1);
    auto decode = [&](int start, int n) {
        batch.n_tokens = n;
        for (int i = 0; i < n; ++i) {
            batch.token[i] = tokens[start + i];
            batch.pos[i] = start + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i == n - 1;
        }
        return llama_decode(ctx, batch) == 0;
    };
    for (int pos = 0; pos < depth; pos += batch_size) {
        capture.prefill_history = capture_history && pos == depth - batch_size;
        if (!decode(pos, batch_size) || capture.failed) {
            return 3;
        }
    }
    llama_synchronize(ctx);
    capture.prefill_history = false;
    if (capture_history) {
        for (int i = 0; i < anchors; ++i) {
            capture.history[i].close();
            if (!capture.history[i] || !capture.saved_history[i]) {
                std::fprintf(stderr, "Missing prefill query history for layer %d.\n", 3 + 4 * i);
                return 4;
            }
        }
    }
    capture.enabled = capture_gdn;
    std::puts("position,target,argmax,nll");
    const int vocabulary_size = llama_vocab_n_tokens(vocab);
    const int reused_before_decode = llama_perf_context(ctx).n_reused;
    const auto decode_start = std::chrono::steady_clock::now();
    for (int pos = depth; pos < depth + steps; ++pos) {
        capture.final_step = pos == depth + steps - 1;
        if (!decode(pos, 1) || capture.failed) {
            return 3;
        }
        const auto * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) {
            return 4;
        }
        const auto * peak = std::max_element(logits, logits + vocabulary_size);
        double normalizer = 0.0;
        for (int i = 0; i < vocabulary_size; ++i) {
            normalizer += std::exp(double(logits[i]) - *peak);
        }
        int target = tokens[pos + 1];
        double loss = std::log(normalizer) + *peak - logits[target];
        if (!std::isfinite(loss)) {
            return 4;
        }
        std::printf("%d,%d,%td,%.9f\n", pos, target, peak - logits, loss);
    }
    const double decode_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_start).count();
    std::fprintf(stderr, "replay_decode_seconds=%.6f replay_tokens_per_second=%.3f\n", decode_seconds, steps / decode_seconds);
    std::fprintf(stderr, "replay_graphs_reused=%d\n", llama_perf_context(ctx).n_reused - reused_before_decode);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    if (capture_gdn) {
        for (int i = 0; i < anchors; ++i) {
            capture.files[i].close();
            if (!capture.files[i] || capture.counts[i] != steps) {
                std::fprintf(stderr, "Expected %d GDN samples for layer %d, captured %d.\n",
                             steps, 2 + 4 * i, capture.counts[i]);
                return 4;
            }
            if (capture_all) {
                capture.queries[i].close();
                capture.keys[i].close();
                capture.values[i].close();
                if (!capture.queries[i] || !capture.keys[i] || !capture.values[i] ||
                    capture.query_counts[i] != steps || !capture.saved_keys[i] || !capture.saved_values[i]) {
                    std::fprintf(stderr, "Missing attention capture for layer %d. Use the patched runtime with an all-full route.\n",
                                 3 + 4 * i);
                    return 4;
                }
            }
        }
    }
}
