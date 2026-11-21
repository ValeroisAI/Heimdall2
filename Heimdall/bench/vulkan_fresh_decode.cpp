#include "llama.h"

#include <chrono>
#include <cstdio>
#include <initializer_list>

// The retrieval index lives in the context, so build it from a fresh prefill.
// stdout is CSV. Review stderr to confirm the GPU and offload placement.
int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s model.gguf\n", argv[0]);
        return 2;
    }

    using Clock = std::chrono::steady_clock;
    constexpr int batch_size = 512;
    constexpr int warmup = 16;
    constexpr int measured = 64;

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.split_mode = LLAMA_SPLIT_MODE_NONE;
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        llama_backend_free();
        return 1;
    }

    bool ok = true;
    std::puts("depth,warmup_tokens,prefill_tps,decode_tps");
    for (const int depth : {32768, 65536}) {
        auto cp = llama_context_default_params();
        cp.n_ctx = depth + warmup + measured;
        cp.n_batch = cp.n_ubatch = batch_size;
        cp.n_threads = cp.n_threads_batch = 8;
        cp.n_seq_max = 1;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        auto * ctx = llama_init_from_model(model, cp);
        if (!ctx) {
            ok = false;
            break;
        }

        const int vocab_size = llama_vocab_n_tokens(llama_model_get_vocab(model));
        auto batch = llama_batch_init(batch_size, 0, 1);
        auto decode = [&](int start, int count) {
            for (int i = 0; i < count; ++i) {
                const int pos = start + i;
                batch.token[i] = pos * 37 % vocab_size;
                batch.pos[i] = pos;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = i == count - 1;
            }
            batch.n_tokens = count;
            return llama_decode(ctx, batch) == 0;
        };

        const auto prefill_start = Clock::now();
        for (int pos = 0; pos < depth && ok; pos += batch_size) {
            ok = decode(pos, batch_size);
        }
        llama_synchronize(ctx);
        const auto prefill_seconds = std::chrono::duration<double>(Clock::now() - prefill_start).count();

        for (int pos = depth; pos < depth + warmup && ok; ++pos) {
            ok = decode(pos, 1) && llama_get_logits_ith(ctx, -1) != nullptr;
        }
        const auto decode_start = Clock::now();
        for (int pos = depth + warmup; pos < depth + warmup + measured && ok; ++pos) {
            ok = decode(pos, 1) && llama_get_logits_ith(ctx, -1) != nullptr;
        }
        const auto decode_seconds = std::chrono::duration<double>(Clock::now() - decode_start).count();

        llama_batch_free(batch);
        llama_free(ctx);
        if (!ok) {
            break;
        }
        std::printf("%d,%d,%.3f,%.3f\n", depth, warmup, depth / prefill_seconds, measured / decode_seconds);
        std::fflush(stdout);
    }

    llama_model_free(model);
    llama_backend_free();
    return ok ? 0 : 1;
}
