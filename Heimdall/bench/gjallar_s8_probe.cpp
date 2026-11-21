#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <iterator>
#include <numeric>
#include <random>
#include <vector>

namespace {

constexpr int head_dim = 256;
constexpr int groups = 2;
constexpr int heads_per_group = 4;
constexpr int captured_blocks = 2039;
constexpr int stored_blocks = 2052;
constexpr int s8_values = 2 * head_dim;
constexpr int s8_words = 144;
constexpr int c4_words = 36;
constexpr int block_tokens = 64;
constexpr int capture_depth = 32768;
constexpr int capture_steps = 256;
constexpr int sink_tokens = 4;
constexpr int window_tokens = 512;
constexpr int c4_blocks = (capture_depth - window_tokens + 1 - sink_tokens) / block_tokens;
constexpr int shortlist_size = 64;
constexpr int warmups = 5;
constexpr int repeats = 50;

void check(VkResult result, const char * operation) {
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "%s failed (%d)\n", operation, static_cast<int>(result));
        std::exit(1);
    }
}

struct Mapping {
    VkBuffer buffer;
    VkDeviceMemory memory;
    void * data;
};

Mapping allocate_buffer(VkPhysicalDevice physical, VkDevice device, VkDeviceSize bytes,
                        VkBufferUsageFlags usage, VkMemoryPropertyFlags required,
                        VkMemoryPropertyFlags excluded) {
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = bytes;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    Mapping result{};
    check(vkCreateBuffer(device, &info, nullptr, &result.buffer), "Create storage buffer");

    VkMemoryRequirements requirements{};
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetBufferMemoryRequirements(device, result.buffer, &requirements);
    vkGetPhysicalDeviceMemoryProperties(physical, &properties);
    uint32_t type = 0;
    for (; type < properties.memoryTypeCount; ++type) {
        auto flags = properties.memoryTypes[type].propertyFlags;
        if ((requirements.memoryTypeBits & (1u << type)) && (flags & required) == required &&
            !(flags & excluded)) {
            break;
        }
    }
    if (type == properties.memoryTypeCount) {
        std::fprintf(stderr, "Vulkan buffer memory with the requested flags is unavailable.\n");
        std::exit(1);
    }
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = type;
    check(vkAllocateMemory(device, &allocation, nullptr, &result.memory), "Allocate buffer memory");
    check(vkBindBufferMemory(device, result.buffer, result.memory, 0), "Bind buffer");
    if (required & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        check(vkMapMemory(device, result.memory, 0, VK_WHOLE_SIZE, 0, &result.data), "Map buffer");
    }
    return result;
}

constexpr auto coherent_host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

Mapping map_system_buffer(VkPhysicalDevice physical, VkDevice device,
                          VkDeviceSize bytes, VkMemoryPropertyFlags extra_flags = 0) {
    return allocate_buffer(physical, device, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           coherent_host | extra_flags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

void release(VkDevice device, const Mapping & mapping) {
    if (mapping.data) {
        vkUnmapMemory(device, mapping.memory);
    }
    vkDestroyBuffer(device, mapping.buffer, nullptr);
    vkFreeMemory(device, mapping.memory, nullptr);
}

template<typename T>
bool read_capture(const char * path, std::vector<T> & data) {
    std::ifstream input(path, std::ios::binary);
    auto bytes = static_cast<std::streamsize>(data.size() * sizeof(T));
    return input.read(reinterpret_cast<char *>(data.data()), bytes) &&
           input.peek() == std::ifstream::traits_type::eof();
}

bool read_floats(const char * path, std::vector<float> & data) {
    return read_capture(path, data) &&
           std::all_of(data.begin(), data.end(), [](float x) { return std::isfinite(x); });
}

void pack_row(uint32_t * row, const float * maxima, const float * minima) {
    for (int group = 0; group < s8_values / 32; ++group) {
        const float * values = group < head_dim / 32
            ? maxima + group * 32 : minima + (group * 32 - head_dim);
        float peak = 0.0f;
        for (int i = 0; i < 32; ++i) {
            peak = std::max(peak, std::abs(values[i]));
        }
        const float ideal_scale = peak / 127.0f;
        const float scale = ideal_scale > 0.0f ? ideal_scale : 1.0f;
        std::memcpy(row + 128 + group, &scale, sizeof(scale));
        for (int word = 0; word < 8; ++word) {
            uint32_t packed = 0;
            for (int i = 0; i < 4; ++i) {
                auto rounded = std::lround(values[4 * word + i] / scale);
                auto code = static_cast<int8_t>(std::clamp(rounded, -127L, 127L));
                packed |= static_cast<uint32_t>(static_cast<uint8_t>(code)) << (8 * i);
            }
            row[group * 8 + word] = packed;
        }
    }
}

uint16_t to_half(float value) {
    return static_cast<uint16_t>(_mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(value), _MM_FROUND_TO_NEAREST_INT)));
}

float from_half(uint16_t bits) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits)));
}

void pack_c4_row(uint32_t * row, const float * centroid) {
    std::fill_n(row, c4_words, 0u);
    for (int group = 0; group < head_dim / 32; ++group) {
        const float * values = centroid + group * 32;
        float peak = 0.0f;
        for (int i = 0; i < 32; ++i) {
            peak = std::max(peak, std::abs(values[i]));
        }
        const float scale = std::max(peak, 1e-12f) / 7.0f;
        const uint16_t bits = to_half(scale);
        row[32 + group / 2] |= static_cast<uint32_t>(bits) << (16 * (group % 2));
        for (int i = 0; i < 32; ++i) {
            int code = static_cast<int>(std::nearbyint(values[i] / scale));
            row[group * 4 + i / 8] |= (static_cast<uint32_t>(code) & 15u) << (4 * (i % 8));
        }
    }
}

std::array<float, heads_per_group> cpu_head_scores(const uint32_t * row, const float * query, bool c4) {
    std::array<float, heads_per_group> scores{};
    const int values = c4 ? head_dim : s8_values;
    for (int i = 0; i < values; ++i) {
        float scale;
        int code;
        if (c4) {
            const auto bits = static_cast<uint16_t>(row[32 + i / 64] >> (16 * ((i / 32) % 2)));
            scale = from_half(bits);
            code = (row[i / 8] >> (4 * (i % 8))) & 15u;
            if (code >= 8) {
                code -= 16;
            }
        } else {
            std::memcpy(&scale, row + 128 + i / 32, sizeof(scale));
            code = (row[i / 4] >> (8 * (i % 4))) & 255u;
            if (code >= 128) {
                code -= 256;
            }
        }
        for (int head = 0; head < heads_per_group; ++head) {
            scores[head] += code * scale * query[i * heads_per_group + head];
        }
    }
    return scores;
}

float cpu_score(const uint32_t * row, const float * query, bool c4) {
    auto scores = cpu_head_scores(row, query, c4);
    return *std::max_element(scores.begin(), scores.end());
}

std::vector<uint32_t> read_shader(const char * path) {
    std::ifstream shader_file(path, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(shader_file)), {});
    if (bytes.empty() || bytes.size() % sizeof(uint32_t) != 0) {
        std::fprintf(stderr, "Could not read SPIR-V module %s\n", path);
        std::exit(2);
    }
    std::vector<uint32_t> words(bytes.size() / sizeof(uint32_t));
    std::memcpy(words.data(), bytes.data(), bytes.size());
    if (words.front() != 0x07230203u) {
        std::fprintf(stderr, "Invalid SPIR-V magic in %s\n", path);
        std::exit(2);
    }
    return words;
}

void write_descriptors(VkDevice device, VkDescriptorSet set, const std::array<VkDescriptorBufferInfo, 3> & buffers) {
    std::array<VkWriteDescriptorSet, 3> writes{};
    for (uint32_t i = 0; i < writes.size(); ++i) {
        writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffers[i];
    }
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

struct ScanContext {
    VkPhysicalDevice physical;
    VkDevice device;
    VkQueue queue;
    VkCommandBuffer command;
    VkFence fence;
    VkPipelineLayout layout;
    VkPipeline scoring, reducer;
    VkDescriptorSet score_set, reduce_set;
};

void time_device_scan(const ScanContext & gpu, const Mapping & staging, const Mapping & query, uint32_t rows) {
    const auto index_bytes = rows * groups * c4_words * sizeof(uint32_t);
    const auto query_bytes = groups * head_dim * heads_per_group * sizeof(float);
    const bool mass = gpu.reducer != VK_NULL_HANDLE;
    const auto output_count = groups * (mass ? heads_per_group : rows);
    const auto output_bytes = output_count * sizeof(float);
    const auto scratch_bytes = rows * groups * heads_per_group * sizeof(float);
    auto device_buffer = [&](VkDeviceSize bytes, VkBufferUsageFlags usage) {
        return allocate_buffer(gpu.physical, gpu.device, bytes, usage | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    };
    auto index = device_buffer(index_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto output = device_buffer(output_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    Mapping scratch{};
    if (mass) {
        scratch = device_buffer(scratch_bytes, 0);
    }
    auto readback = allocate_buffer(gpu.physical, gpu.device, output_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    coherent_host | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDescriptorBufferInfo query_info{query.buffer, 0, query_bytes};
    write_descriptors(gpu.device, gpu.score_set,
                      {{{index.buffer, 0, index_bytes}, query_info,
                        {mass ? scratch.buffer : output.buffer, 0, mass ? scratch_bytes : output_bytes}}});
    if (mass) {
        write_descriptors(gpu.device, gpu.reduce_set,
                          {{{scratch.buffer, 0, scratch_bytes}, query_info, {output.buffer, 0, output_bytes}}});
    }
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &gpu.command;
    auto begin = [&] {
        check(vkResetCommandBuffer(gpu.command, 0), "Reset scan commands");
        VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(gpu.command, &info), "Begin scan commands");
    };
    auto run = [&] {
        check(vkResetFences(gpu.device, 1, &gpu.fence), "Reset scan fence");
        check(vkQueueSubmit(gpu.queue, 1, &submit, gpu.fence), "Submit scan");
        check(vkWaitForFences(gpu.device, 1, &gpu.fence, VK_TRUE, UINT64_MAX), "Wait for scan");
    };
    auto barrier = [&](VkPipelineStageFlags source, VkPipelineStageFlags dest, VkAccessFlags write, VkAccessFlags read) {
        VkMemoryBarrier memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        memory.srcAccessMask = write;
        memory.dstAccessMask = read;
        vkCmdPipelineBarrier(gpu.command, source, dest, 0, 1, &memory, 0, nullptr, 0, nullptr);
    };
    auto dispatch = [&](VkPipeline pipeline, VkDescriptorSet set, uint32_t count) {
        vkCmdBindPipeline(gpu.command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(gpu.command, VK_PIPELINE_BIND_POINT_COMPUTE, gpu.layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(gpu.command, gpu.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rows), &rows);
        vkCmdDispatch(gpu.command, count, 1, 1);
    };
    begin();
    VkBufferCopy upload{0, 0, index_bytes};
    vkCmdCopyBuffer(gpu.command, staging.buffer, index.buffer, 1, &upload);
    barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    check(vkEndCommandBuffer(gpu.command), "End index upload");
    run();

    begin();
    dispatch(gpu.scoring, gpu.score_set, groups * rows);
    if (mass) {
        barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        dispatch(gpu.reducer, gpu.reduce_set, groups);
    }
    check(vkEndCommandBuffer(gpu.command), "End device scan");
    std::array<double, repeats> times{};
    using Clock = std::chrono::steady_clock;
    for (int sample = -warmups; sample < repeats; ++sample) {
        auto before = Clock::now();
        run();
        if (sample >= 0) {
            times[sample] = std::chrono::duration<double, std::micro>(Clock::now() - before).count();
        }
    }
    begin();
    barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    VkBufferCopy download{0, 0, output_bytes};
    vkCmdCopyBuffer(gpu.command, output.buffer, readback.buffer, 1, &download);
    barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
    check(vkEndCommandBuffer(gpu.command), "End scan download");
    run();

    auto * packed = static_cast<const uint32_t *>(staging.data);
    auto * coefficients = static_cast<const float *>(query.data);
    auto * results = static_cast<const float *>(readback.data);
    std::vector<float> expected(output_count);
    for (int group = 0; group < groups; ++group) {
        const float * q = coefficients + group * head_dim * heads_per_group;
        std::vector<std::array<float, heads_per_group>> head_scores(rows);
        for (uint32_t row = 0; row < rows; ++row) {
            auto * desc = packed + (group * rows + row) * c4_words;
            if (mass) {
                head_scores[row] = cpu_head_scores(desc, q, true);
            } else {
                expected[group * rows + row] = cpu_score(desc, q, true);
            }
        }
        if (mass) {
            for (int head = 0; head < heads_per_group; ++head) {
                double peak = -INFINITY;
                for (const auto & scores : head_scores) {
                    peak = std::max(peak, scores[head] / 16.0);
                }
                double sum = 0;
                for (const auto & scores : head_scores) {
                    sum += std::exp(scores[head] / 16.0 - peak);
                }
                expected[group * heads_per_group + head] = static_cast<float>(peak + std::log(sum * block_tokens));
            }
        }
    }
    float max_error = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(results[i]) || !std::isfinite(expected[i])) {
            std::fprintf(stderr, "Non-finite scan result at %zu\n", i);
            std::exit(1);
        }
        max_error = std::max(max_error, std::abs(expected[i] - results[i]));
    }
    if (max_error > (mass ? 0.001f : 0.05f)) {
        std::fprintf(stderr, "Device scan differs from CPU reference by %f\n", max_error);
        std::exit(1);
    }
    std::sort(times.begin(), times.end());
    std::printf("%u,%zu,%.3f,%.3f,%.6f\n", rows, index_bytes,
                (times[repeats / 2 - 1] + times[repeats / 2]) / 2, times[(95 * repeats + 99) / 100 - 1], max_error);
    release(gpu.device, readback);
    if (mass) {
        release(gpu.device, scratch);
    }
    release(gpu.device, output);
    release(gpu.device, index);
}

void time_key_upload(VkPhysicalDevice physical, VkDevice device, VkQueue queue, VkCommandBuffer command,
                     VkFence fence, const std::vector<uint16_t> & keys,
                     const std::array<std::array<uint32_t, shortlist_size>, groups> & shortlist) {
    constexpr VkDeviceSize block_bytes = block_tokens * head_dim * sizeof(uint16_t);
    auto host = allocate_buffer(physical, device, groups * c4_blocks * block_bytes,
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT, coherent_host, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    auto gpu = allocate_buffer(physical, device, groups * shortlist_size * block_bytes,
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    auto * archive = static_cast<uint16_t *>(host.data);
    for (int group = 0; group < groups; ++group) {
        for (int token = 0; token < c4_blocks * block_tokens; ++token) {
            const auto * key = keys.data() + ((sink_tokens + token) * groups + group) * head_dim;
            std::copy_n(key, head_dim, archive + (group * c4_blocks * block_tokens + token) * head_dim);
        }
    }

    std::array<VkBufferCopy, groups * shortlist_size> copies{};
    for (int group = 0; group < groups; ++group) {
        // Keep the selected regions in archive order for the DMA submission.
        auto ordered = shortlist[group];
        std::sort(ordered.begin(), ordered.end());
        for (int i = 0; i < shortlist_size; ++i) {
            int offset = group * shortlist_size + i;
            copies[offset] = {(group * c4_blocks + ordered[i]) * block_bytes, offset * block_bytes, block_bytes};
        }
    }

    std::fputs("rerank_groups,rerank_key_bytes,regions,upload_median_us,upload_p95_us\n", stderr);
    for (uint32_t group_count : {1u, static_cast<uint32_t>(groups)}) {
        const uint32_t region_count = group_count * shortlist_size;
        check(vkResetCommandBuffer(command, 0), "Reset key upload commands");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(command, &begin), "Begin key upload");
        vkCmdCopyBuffer(command, host.buffer, gpu.buffer, region_count, copies.data());
        check(vkEndCommandBuffer(command), "End key upload");

        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        std::array<double, repeats> times{};
        for (int sample = -warmups; sample < repeats; ++sample) {
            check(vkResetFences(device, 1, &fence), "Reset key upload fence");
            auto start = std::chrono::steady_clock::now();
            check(vkQueueSubmit(queue, 1, &submit, fence), "Submit key upload");
            check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "Wait for key upload");
            if (sample >= 0) {
                times[sample] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
            }
        }
        std::sort(times.begin(), times.end());
        std::fprintf(stderr, "%u,%llu,%u,%.3f,%.3f\n", group_count,
                     static_cast<unsigned long long>(region_count * block_bytes), region_count,
                     (times[repeats / 2 - 1] + times[repeats / 2]) / 2, times[(95 * repeats + 99) / 100 - 1]);
    }
    release(device, gpu);
    release(device, host);
}

} // namespace

int main(int argc, char ** argv) {
    const bool mass = argc > 1 && std::strcmp(argv[1], "--c4-mass") == 0;
    const bool device_scan = mass || (argc > 1 && std::strcmp(argv[1], "--c4-device") == 0);
    const bool c4 = device_scan || (argc > 1 && std::strcmp(argv[1], "--c4") == 0);
    const char * program = argv[0];
    if (c4) {
        --argc;
        ++argv;
    }
    const int capture_arg = mass ? 3 : 2;
    const bool captured = argc == capture_arg + 2;
    if (argc != capture_arg && !captured) {
        std::fprintf(stderr, "Usage: %s [--c4 | --c4-device] shader.spv [capture capture-queries.f32]\n"
                             "       %s --c4-mass per-head.spv reduce.spv [capture capture-queries.f32]\n"
                             "C4 capture: cal-{active,archive}-k-L (fp16, 33024 tokens); "
                             "S8 capture: layer-ksum-L (fp32).\n", program, program);
        return 2;
    }
    auto shader_words = read_shader(argv[1]);
    auto reducer_words = mass ? read_shader(argv[2]) : std::vector<uint32_t>{};

    const int descriptor_values = c4 ? head_dim : s8_values;
    const int descriptor_words = c4 ? c4_words : s8_words;
    const int input_blocks = c4 ? c4_blocks : captured_blocks;
    std::vector<float> bounds(c4 ? 0 : 2 * groups * stored_blocks * head_dim);
    std::vector<uint16_t> keys(c4 ? (capture_depth + capture_steps) * groups * head_dim : 0);
    std::vector<float> query_samples((c4 ? capture_steps : 64) * groups * heads_per_group * head_dim);
    if (captured) {
        bool valid = c4 ? read_capture(argv[capture_arg], keys) &&
                          std::all_of(keys.begin(), keys.end(), [](uint16_t h) { return (h & 0x7c00u) != 0x7c00u; })
                        : read_floats(argv[capture_arg], bounds);
        if (!valid || !read_floats(argv[capture_arg + 1], query_samples)) {
            std::fprintf(stderr, "Capture size or values do not match the Qwen3.5-0.8B probe geometry.\n");
            return 2;
        }
    } else {
        std::mt19937 rng(0);
        std::uniform_real_distribution<float> distribution(0.0f, 3.0f);
        auto midpoint = bounds.begin() + bounds.size() / 2;
        std::generate(bounds.begin(), midpoint, [&] { return distribution(rng); });
        std::generate(midpoint, bounds.end(), [&] { return -distribution(rng); });
        std::generate(query_samples.begin(), query_samples.end(), [&] { return distribution(rng) - 1.5f; });
        std::generate(keys.begin(), keys.end(), [&] { return to_half(distribution(rng) - 1.5f); });
    }

    std::vector<uint32_t> centroid_rows(c4 ? groups * c4_blocks * c4_words : 0);
    for (int group = 0; c4 && group < groups; ++group) {
        for (int block = 0; block < c4_blocks; ++block) {
            std::array<float, head_dim> centroid{};
            for (int token = 0; token < block_tokens; ++token) {
                int position = sink_tokens + block * block_tokens + token;
                const uint16_t * key = keys.data() + (position * groups + group) * head_dim;
                for (int d = 0; d < head_dim; ++d) {
                    centroid[d] += from_half(key[d]) / block_tokens;
                }
            }
            pack_c4_row(centroid_rows.data() + (group * c4_blocks + block) * c4_words, centroid.data());
        }
    }

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &app;
    VkInstance instance;
    check(vkCreateInstance(&instance_info, nullptr, &instance), "Create instance");
    uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(instance, &count, nullptr), "Count physical devices");
    std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(instance, &count, devices.data()), "List physical devices");
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};
    for (auto candidate : devices) {
        vkGetPhysicalDeviceProperties(candidate, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            physical = candidate;
            break;
        }
    }
    if (!physical) {
        std::fprintf(stderr, "No discrete Vulkan GPU found.\n");
        return 2;
    }
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
    auto compute = std::find_if(families.begin(), families.end(), [](const auto & family) {
        return (family.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
    });
    if (compute == families.end()) {
        std::fprintf(stderr, "No Vulkan compute queue found.\n");
        return 2;
    }
    uint32_t family = static_cast<uint32_t>(compute - families.begin());
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    VkDevice device;
    check(vkCreateDevice(physical, &device_info, nullptr, &device), "Create device");
    VkQueue queue;
    vkGetDeviceQueue(device, family, 0, &queue);

    VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    module_info.codeSize = shader_words.size() * sizeof(uint32_t);
    module_info.pCode = shader_words.data();
    VkShaderModule module;
    check(vkCreateShaderModule(device, &module_info, nullptr, &module), "Create shader module");

    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
    for (uint32_t i = 0; i < bindings.size(); ++i) {
        bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    }
    VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set_info.bindingCount = static_cast<uint32_t>(bindings.size());
    set_info.pBindings = bindings.data();
    VkDescriptorSetLayout set_layout;
    check(vkCreateDescriptorSetLayout(device, &set_info, nullptr, &set_layout), "Create descriptor layout");
    VkPushConstantRange push_range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)};
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &set_layout;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push_range;
    VkPipelineLayout pipeline_layout;
    check(vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout), "Create pipeline layout");

    VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.layout = pipeline_layout;
    pipeline_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = module;
    pipeline_info.stage.pName = "main";
    VkPipeline pipeline;
    check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline), "Create compute pipeline");
    VkShaderModule reducer_module = VK_NULL_HANDLE;
    VkPipeline reducer = VK_NULL_HANDLE;
    if (mass) {
        module_info.codeSize = reducer_words.size() * sizeof(uint32_t);
        module_info.pCode = reducer_words.data();
        check(vkCreateShaderModule(device, &module_info, nullptr, &reducer_module), "Create reduction module");
        pipeline_info.stage.module = reducer_module;
        check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &reducer),
              "Create reduction pipeline");
    }
    const uint32_t set_count = mass ? 2 : 1;
    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * set_count};
    VkDescriptorPoolCreateInfo descriptor_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    descriptor_info.maxSets = set_count;
    descriptor_info.poolSizeCount = 1;
    descriptor_info.pPoolSizes = &pool_size;
    VkDescriptorPool descriptor_pool;
    check(vkCreateDescriptorPool(device, &descriptor_info, nullptr, &descriptor_pool), "Create descriptor pool");
    VkDescriptorSetAllocateInfo descriptor_alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    descriptor_alloc.descriptorPool = descriptor_pool;
    std::array<VkDescriptorSetLayout, 2> set_layouts{set_layout, set_layout};
    descriptor_alloc.descriptorSetCount = set_count;
    descriptor_alloc.pSetLayouts = set_layouts.data();
    std::array<VkDescriptorSet, 2> sets{};
    check(vkAllocateDescriptorSets(device, &descriptor_alloc, sets.data()), "Allocate descriptors");
    auto descriptors = sets[0];

    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = family;
    VkCommandPool pool;
    check(vkCreateCommandPool(device, &pool_info, nullptr, &pool), "Create command pool");
    VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_info.commandPool = pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    VkCommandBuffer command;
    check(vkAllocateCommandBuffers(device, &command_info, &command), "Allocate command buffer");
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    check(vkCreateFence(device, &fence_info, nullptr, &fence), "Create fence");

    std::fprintf(stderr, "%s (%s input)\n", props.deviceName, captured ? "captured" : "synthetic");
    std::puts(device_scan ? "rows_per_group,index_bytes,submit_median_us,submit_p95_us,max_delta"
                         : "rows_per_group,index_bytes,score_median_us,top64_median_us,top64_p95_us,max_score_delta");
    using Clock = std::chrono::steady_clock;
    std::array<std::array<uint32_t, shortlist_size>, groups> captured_shortlist{};
    for (uint32_t rows : {static_cast<uint32_t>(input_blocks), 16384u}) {
        const auto index_bytes = rows * groups * descriptor_words * sizeof(uint32_t);
        const auto query_bytes = groups * descriptor_values * heads_per_group * sizeof(float);
        const auto result_bytes = rows * groups * sizeof(float);
        auto index = device_scan ? allocate_buffer(physical, device, index_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                  coherent_host, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                                 : map_system_buffer(physical, device, index_bytes);
        auto query = map_system_buffer(physical, device, query_bytes);
        auto * packed = static_cast<uint32_t *>(index.data);
        auto * coefficients = static_cast<float *>(query.data);
        for (int group = 0; group < groups; ++group) {
            for (uint32_t row = 0; row < rows; ++row) {
                auto source = row % input_blocks;
                auto * output = packed + (group * rows + row) * descriptor_words;
                if (c4) {
                    std::copy_n(centroid_rows.data() + (group * c4_blocks + source) * c4_words, c4_words, output);
                    continue;
                }
                const auto * maxima = bounds.data() + (group * stored_blocks + source) * head_dim;
                const auto * minima = maxima + groups * stored_blocks * head_dim;
                pack_row(output, maxima, minima);
            }
            for (int d = 0; d < descriptor_values; ++d) {
                for (int head = 0; head < heads_per_group; ++head) {
                    float x = query_samples[(group * heads_per_group + head) * head_dim + d % head_dim];
                    coefficients[(group * descriptor_values + d) * heads_per_group + head] =
                        c4 ? x : (d < head_dim ? std::max(x, 0.0f) : std::min(x, 0.0f));
                }
            }
        }
        if (device_scan) {
            time_device_scan({physical, device, queue, command, fence, pipeline_layout, pipeline, reducer,
                              sets[0], sets[1]}, index, query, rows);
            release(device, index);
            release(device, query);
            continue;
        }
        auto results = map_system_buffer(physical, device, result_bytes, VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        const auto * scores = static_cast<const float *>(results.data);
        write_descriptors(device, descriptors,
                          {{{index.buffer, 0, index_bytes}, {query.buffer, 0, query_bytes}, {results.buffer, 0, result_bytes}}});
        check(vkResetCommandBuffer(command, 0), "Reset command buffer");
        VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(command, &begin_info), "Begin commands");
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &descriptors, 0, nullptr);
        vkCmdPushConstants(command, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rows), &rows);
        vkCmdDispatch(command, groups * rows, 1, 1);
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                             0, 1, &barrier, 0, nullptr, 0, nullptr);
        check(vkEndCommandBuffer(command), "End commands");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        std::vector<uint32_t> candidates(rows);
        std::array<double, repeats> score_times{}, shortlist_times{};
        uint64_t checksum = 0;
        for (int sample = -warmups; sample < repeats; ++sample) {
            check(vkResetFences(device, 1, &fence), "Reset fence");
            auto start = Clock::now();
            check(vkQueueSubmit(queue, 1, &submit, fence), "Submit scoring");
            check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "Wait for scoring");
            auto scored = Clock::now();
            for (int group = 0; group < groups; ++group) {
                std::iota(candidates.begin(), candidates.end(), 0);
                const float * group_scores = scores + group * rows;
                std::partial_sort(candidates.begin(), candidates.begin() + shortlist_size, candidates.end(),
                                  [&](uint32_t a, uint32_t b) {
                                      if (group_scores[a] == group_scores[b]) {
                                          return a < b;
                                      }
                                      return group_scores[a] > group_scores[b];
                                  });
                if (sample == -1 && c4 && rows == c4_blocks) {
                    std::copy_n(candidates.data(), shortlist_size, captured_shortlist[group].data());
                }
                checksum += std::accumulate(candidates.begin(), candidates.begin() + shortlist_size, uint64_t{0});
            }
            auto selected = Clock::now();
            if (sample >= 0) {
                score_times[sample] = std::chrono::duration<double, std::micro>(scored - start).count();
                shortlist_times[sample] = std::chrono::duration<double, std::micro>(selected - start).count();
            }
        }
        float max_error = 0.0f;
        for (uint32_t row = 0; row < groups * rows; ++row) {
            float expected = cpu_score(packed + row * descriptor_words,
                                       coefficients + (row / rows) * descriptor_values * heads_per_group, c4);
            if (!std::isfinite(expected) || !std::isfinite(scores[row])) {
                std::fprintf(stderr, "Non-finite score at row %u\n", row);
                return 1;
            }
            max_error = std::max(max_error, std::abs(expected - scores[row]));
        }
        if (max_error > 0.05f) {
            std::fprintf(stderr, "GPU scores differ from the packed CPU reference by %.6f\n", max_error);
            return 1;
        }
        std::sort(score_times.begin(), score_times.end());
        std::sort(shortlist_times.begin(), shortlist_times.end());
        auto median = [](const auto & times) { return (times[repeats / 2 - 1] + times[repeats / 2]) / 2; };
        std::printf("%u,%zu,%.3f,%.3f,%.3f,%.6f\n", rows, index_bytes, median(score_times),
                    median(shortlist_times), shortlist_times[(95 * repeats + 99) / 100 - 1], max_error);
        std::fprintf(stderr, "Top-%d checksum for %u rows: %llu\n", shortlist_size, rows,
                     static_cast<unsigned long long>(checksum));
        release(device, index);
        release(device, query);
        release(device, results);
    }
    if (c4 && !device_scan) {
        time_key_upload(physical, device, queue, command, fence, keys, captured_shortlist);
    }
    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, pool, nullptr);
    vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
    if (mass) {
        vkDestroyPipeline(device, reducer, nullptr);
        vkDestroyShaderModule(device, reducer_module, nullptr);
    }
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
    vkDestroyShaderModule(device, module, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
}
