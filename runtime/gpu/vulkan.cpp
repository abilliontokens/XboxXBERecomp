// The title's Direct3D 8 calls drawn with Vulkan 1.2: the native Linux backend behind
// gpu::Backend, matching d3d12.cpp's observable behaviour (same surfaces, read-back rules,
// clear mask, gamma at present, internal-resolution scaling) on the Vulkan API instead.
// The whole file is the non-Windows CreateBackend; d3d12.cpp keeps the Windows one.
#if !defined(_WIN32)

#include "gpu/vulkan.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if __has_include(<imgui.h>)
#include <imgui.h>
#if __has_include(<backends/imgui_impl_vulkan.h>)
#include <backends/imgui_impl_vulkan.h>
#define XBE_VK_IMGUI 1
#endif
#if __has_include(<backends/imgui_impl_sdl3.h>)
#include <backends/imgui_impl_sdl3.h>
#define XBE_VK_IMGUI_SDL3 1
#endif
#endif

#include "app/app.h"
#include "app/platform.h"
#include "core/diag.h"
#include "core/log.h"
#include "gpu/backend.h"
#include "gpu/shaders.h"
#include "gpu/spirv.h"
#include "gpu/texture.h"
#include "gpu/vertex.h"
#include "kernel/kernel.h"
#include "kernel/memory.h"

namespace gpu {
namespace {

constexpr int kFrames = 3;
constexpr uint64_t kStaleFrames = 8;
constexpr uint64_t kUploadSize = 64ull << 20;
constexpr uint64_t kUploadReserve = 8ull << 20;
// Descriptor sets one recording segment can allocate: one per draw, blit or depth
// copy (a household frame is ~2200). EnsureSetRoom submits the segment before it
// runs out, so no draw or blit is ever dropped for want of a set.
constexpr uint32_t kSetsPerSegment = 8192;
constexpr VkFormat kBgra = VK_FORMAT_B8G8R8A8_UNORM;

std::mutex g_handover_lock;
VkInstance g_instance = VK_NULL_HANDLE;
VkSurfaceKHR g_surface = VK_NULL_HANDLE;
std::vector<std::string> g_extension_storage;
std::vector<const char*> g_extensions;
VulkanHandles g_handles;
// ---- perf (mirrors d3d12.cpp's Perf/LogPerf; logged once a second under diag::Perf()) ----
using VkPerfClock = std::chrono::steady_clock;
double VkPerfMs(VkPerfClock::time_point t) { return std::chrono::duration<double, std::milli>(VkPerfClock::now() - t).count(); }
struct VkPerf {
    double draw = 0, present = 0, clear = 0, copy = 0;
    uint64_t draws = 0, skipped = 0, nopipeline = 0, nopipeShader = 0, nopipePso = 0, nopipeSet = 0, staleAlloc = 0, verts = 0, raw = 0;
    uint64_t upload = 0, overflow = 0, texups = 0, texupBytes = 0, newtex = 0;
    uint64_t targets = 0, rtups = 0, syncs = 0, shaders = 0, psos = 0, clears = 0, copies = 0;
};
VkPerf g_vkPerf;
struct VkPerfScope {
    double& acc;
    VkPerfClock::time_point t = VkPerfClock::now();
    ~VkPerfScope() { acc += VkPerfMs(t); }
};
// ---- XBE_VK_DEBUG: validation layer + debug-utils messenger (mirrors XBE_D3D12_DEBUG) ----
bool VkDebugEnabled() { static const bool on = std::getenv("XBE_VK_DEBUG") != nullptr; return on; }
VkDebugUtilsMessengerEXT g_vkMessenger = VK_NULL_HANDLE;
VKAPI_ATTR VkBool32 VKAPI_CALL VkDebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                               VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    if (!data || !data->pMessage) return VK_FALSE;
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) LOGE("vk-validation: %s", data->pMessage);
    else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) LOGW("vk-validation: %s", data->pMessage);
    else LOGI("vk-validation: %s", data->pMessage);
    return VK_FALSE;
}
void DestroyVkMessenger() {
    if (g_vkMessenger && g_instance) {
        auto fn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(g_instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (fn) fn(g_instance, g_vkMessenger, nullptr);
    }
    g_vkMessenger = VK_NULL_HANDLE;
}
// The window's real drawable size in pixels. On Wayland the WSI reports
// currentExtent as {UINT32_MAX, UINT32_MAX} ("the client chooses"), and the
// swapchain extent IS the window size once presented - so the swapchain must
// follow the window, never app::Output() (the settings/desktop size).
// platform::ClientSize() is SDL_GetWindowSizeInPixels cached on the window
// thread by main_sdl's resize/pixel-size event handling; {0,0} before the
// window exists, where app::Output() is the only fallback.
VkExtent2D DrawableExtent() {
    app::platform::Size s = app::platform::ClientSize();
    if (s.width && s.height) return {s.width, s.height};
    auto out = app::Output();
    return {out.width, out.height};
}



float BitsF(uint32_t v) { float f; std::memcpy(&f, &v, 4); return f; }
void ColorF(uint32_t argb, float* o) {
    o[0] = ((argb >> 16) & 255) / 255.0f; o[1] = ((argb >> 8) & 255) / 255.0f;
    o[2] = (argb & 255) / 255.0f; o[3] = (argb >> 24) / 255.0f;
}
// DXGI_FORMAT values Assembled::elements carries (harness/compat/dxgiformat.h)
enum : uint8_t { kDxgiRgba32f = 2, kDxgiRgb32f = 6, kDxgiRgba16Snorm = 13, kDxgiRgba16Sint = 14, kDxgiRg32f = 16,
    kDxgiRgba8 = 28, kDxgiRg16Snorm = 37, kDxgiRg16Sint = 38, kDxgiR32f = 41, kDxgiR32u = 42,
    kDxgiRg8 = 49, kDxgiR16Snorm = 58, kDxgiR16Sint = 59, kDxgiR8 = 61, kDxgiBgra8 = 87 };
VkFormat VkFormatForDxgi(uint8_t f) {
    switch (f) {
        case kDxgiRgba32f: return VK_FORMAT_R32G32B32A32_SFLOAT;
        case kDxgiRgb32f: return VK_FORMAT_R32G32B32_SFLOAT;
        case kDxgiRg32f: return VK_FORMAT_R32G32_SFLOAT;
        case kDxgiR32f: return VK_FORMAT_R32_SFLOAT;
        case kDxgiR32u: return VK_FORMAT_R32_UINT;
        case kDxgiBgra8: return VK_FORMAT_B8G8R8A8_UNORM;
        case kDxgiRgba8: return VK_FORMAT_R8G8B8A8_UNORM;
        case kDxgiRgba16Snorm: return VK_FORMAT_R16G16B16A16_SNORM;
        case kDxgiRgba16Sint: return VK_FORMAT_R16G16B16A16_SINT;
        case kDxgiRg16Snorm: return VK_FORMAT_R16G16_SNORM;
        case kDxgiRg16Sint: return VK_FORMAT_R16G16_SINT;
        case kDxgiR16Snorm: return VK_FORMAT_R16_SNORM;
        case kDxgiR16Sint: return VK_FORMAT_R16_SINT;
        case kDxgiRg8: return VK_FORMAT_R8G8_UNORM;
        case kDxgiR8: return VK_FORMAT_R8_UNORM;
        default: return VK_FORMAT_R32G32B32A32_SFLOAT;
    }
}
VkBlendFactor BlendFactor(uint32_t v, bool alpha) {
    switch (v) {
        case 0: return VK_BLEND_FACTOR_ZERO;
        case 1: return VK_BLEND_FACTOR_ONE;
        case 0x300: return alpha ? VK_BLEND_FACTOR_SRC_ALPHA : VK_BLEND_FACTOR_SRC_COLOR;
        case 0x301: return alpha ? VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA : VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case 0x302: return VK_BLEND_FACTOR_SRC_ALPHA;
        case 0x303: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        case 0x304: return VK_BLEND_FACTOR_DST_ALPHA;
        case 0x305: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        case 0x306: return alpha ? VK_BLEND_FACTOR_DST_ALPHA : VK_BLEND_FACTOR_DST_COLOR;
        case 0x307: return alpha ? VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA : VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        case 0x308: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
        case 0x8001: case 0x8003: return VK_BLEND_FACTOR_CONSTANT_COLOR;
        case 0x8002: case 0x8004: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
        default: return VK_BLEND_FACTOR_ONE;
    }
}
VkBlendOp BlendOpVk(uint32_t v) {
    switch (v) {
        case 0x800A: return VK_BLEND_OP_SUBTRACT;
        case 0x800B: return VK_BLEND_OP_REVERSE_SUBTRACT;
        case 0x8007: return VK_BLEND_OP_MIN;
        case 0x8008: return VK_BLEND_OP_MAX;
        default: return VK_BLEND_OP_ADD;
    }
}
VkCompareOp CmpOp(uint32_t v) { return VkCompareOp(v >= 0x200 && v <= 0x207 ? v - 0x200 : 7); }
uint32_t CmpIndex(uint32_t v) { return v >= 0x200 && v <= 0x207 ? v - 0x200 : 7; }
VkStencilOp StencilOpVk(uint32_t v) {
    switch (v) {
        case 0: return VK_STENCIL_OP_ZERO;
        case 0x1E01: return VK_STENCIL_OP_REPLACE;
        case 0x1E02: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
        case 0x1E03: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
        case 0x150A: return VK_STENCIL_OP_INVERT;
        case 0x8507: return VK_STENCIL_OP_INCREMENT_AND_WRAP;
        case 0x8508: return VK_STENCIL_OP_DECREMENT_AND_WRAP;
        default: return VK_STENCIL_OP_KEEP;
    }
}
VkSamplerAddressMode AddressMode(uint32_t v) {
    switch (v) {
        case 2: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        case 3: case 5: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        case 4: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        default: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    }
}
VkFormat HostFormatVk(Host h) {
    switch (h) {
        case Host::BC1: return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
        case Host::BC2: return VK_FORMAT_BC2_UNORM_BLOCK;
        case Host::BC3: return VK_FORMAT_BC3_UNORM_BLOCK;
        case Host::R32F: return VK_FORMAT_R32_SFLOAT;
        case Host::R16Unorm: return VK_FORMAT_R16_UNORM;
        default: return VK_FORMAT_B8G8R8A8_UNORM;
    }
}
uint8_t ShadowDepthKind(uint32_t format) { return format == 0x2C || format == 0x30 ? 2 : 1; }
uint32_t GuestPitch(uint32_t width) { return (width * 4 + 63) & ~63u; }
bool ReadBackFits(uint32_t data, uint32_t gw, uint32_t gh, uint32_t size) {
    uint64_t bytes = uint64_t(GuestPitch(gw)) * gh;
    if (!size || bytes <= size) return true;
    static std::mutex m; static std::unordered_set<uint32_t> refused; bool first;
    { std::lock_guard lock(m); first = refused.insert(data).second; }
    if (first) LOG_FIRST(32, "vulkan: read-back of %08X refused, its %ux%u target needs %llu bytes, the surface has %u",
                         data, gw, gh, (unsigned long long)bytes, size);
    return false;
}
// Low physical memory holds kernel variables, never a surface: no read-back lands below 0x80100000.
bool LowPhysical(uint32_t data) { return data >= mem::kContiguousBase && data < mem::kContiguousBase + mem::kKernelDataSize; }
uint64_t MemSig(uint32_t data, uint32_t total) {
    const uint8_t* p = static_cast<const uint8_t*>(kernel::HostPtr(data));
    uint64_t sig = Hash(&total, sizeof(total));
    if (total <= 65536) return Hash(p, total, sig);
    for (uint32_t i = 0; i < 64; i++) sig = Hash(p + uint64_t(total - 64) * i / 63, 64, sig);
    return sig;
}
struct ImageRes {
    VkImage image = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE, srvView = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED; VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    uint32_t w = 0, h = 0, layers = 1;
    VkImageView SampleView() const { return srvView ? srvView : view; }
    bool Valid() const { return image != VK_NULL_HANDLE; }
};
struct Target {
    ImageRes res, fb; bool depth = false;
    uint32_t gw = 0, gh = 0, w = 0, h = 0; float dx = 0;
    uint32_t data = 0, bytes = 0; uint64_t epoch = 0, sig = 0;
    bool drawn = false; std::vector<uint8_t> read;
};
struct Tex {
    ImageRes res; TextureDesc d; VkFormat fmt = VK_FORMAT_UNDEFINED;
    uint32_t levels = 1; uint64_t sig = 0, frame = 0, palette = 0; uint8_t flags = 0;
};
bool ChangedTarget(const Target& t) {
    if (t.drawn || t.read.empty()) return false;
    uint32_t row = t.gw * 4, pitch = GuestPitch(t.gw);
    const uint8_t* p = static_cast<const uint8_t*>(kernel::HostPtr(t.data));
    for (uint32_t y = 0; y < t.gh; y++) if (std::memcmp(p + size_t(y) * pitch, &t.read[size_t(y) * row], row)) return true;
    return false;
}
bool LatchCpuWrite(Target& t) {
    uint64_t e = DirtyEpoch(t.data);
    if (e <= t.epoch) return false;
    t.epoch = e;
    uint64_t sig = MemSig(t.data, t.bytes);
    if (sig == t.sig && !ChangedTarget(t)) return false;
    t.sig = sig; return true;
}
uint64_t TargetMapKey(uint32_t data, bool depth) { return uint64_t(data) << 1 | (depth ? 1 : 0); }
struct DrawTextures {
    TexInfo info[4]; VkImageView view[4] = {}; TextureDesc desc[4]; bool hasDesc[4] = {};
    bool rt[4] = {}; float k[4] = {1,1,1,1}; float wide[4] = {};
};
struct DrawShaders {
    bool prog = false, quad = false, fixed_transform = false; FixedKey fk;
    uint64_t vkey = 0, vh = 0, pkey = 0, ph = 0;
    VkShaderModule vs = VK_NULL_HANDLE, ps = VK_NULL_HANDLE;
};
struct GuestExtent { uint32_t w, h; float dx = 0; };
struct PsoKey {
    uint64_t vs, ps; uint32_t regs, topo, strip, has_rt, has_ds; uint32_t elements[16];
    uint32_t blend, src, dst, op, wmask, zen, zwrite, zfunc, sten, sfunc, smask, swmask, sfail, szfail, spass, cull, ccw, fill;
    int32_t zbias; float zslope; uint32_t rtfmt; uint32_t sref;
};
uint32_t SimpleState(const State& s, uint32_t method) { return s.simple[method >> 2]; }
const char* kBlitHlsl = R"(
cbuffer B : register(b1) { float4 uvrect; float4 opts; };
Texture2D img : register(t0); Texture2D lut : register(t1);
SamplerState ls : register(s0); SamplerState ps : register(s1);
struct O { float4 p : SV_Position; float2 uv : TEXCOORD0; };
O vs(uint id : SV_VertexID) { O o; float2 t = float2((id << 1) & 2, id & 2);
    o.p = float4(t * float2(2, -2) + float2(-1, 1), 0, 1); o.uv = uvrect.xy + t * uvrect.zw; return o; }
float4 psmain(O i) : SV_Target { float4 c = 0; int n = max((int)opts.z, 1); float2 px = fwidth(i.uv);
    for (int y = 0; y < n; y++) for (int x = 0; x < n; x++) c += img.SampleLevel(ls, i.uv + px * ((float2(x, y) + 0.5) / n - 0.5), 0);
    c /= n * n; c.rgb *= opts.w;
    if (opts.x > 0) c.rgb = float3(lut.Sample(ps, float2(c.r, 0.5)).r, lut.Sample(ps, float2(c.g, 0.5)).g, lut.Sample(ps, float2(c.b, 0.5)).b);
    return float4(c.rgb, opts.y > 0 ? c.a : 1); }
)";
const char* kDepthHlsl = R"(
Texture2D<float> img : register(t0); SamplerState ps : register(s1);
struct O { float4 p : SV_Position; float2 uv : TEXCOORD0; };
O vs(uint id : SV_VertexID) { O o; o.uv = float2((id << 1) & 2, id & 2); o.p = float4(o.uv * float2(2, -2) + float2(-1, 1), 0, 1); return o; }
float psmain(O i) : SV_Depth { return img.SampleLevel(ps, i.uv, 0); }
)";

class VulkanBackend final : public Backend {
public:
    ~VulkanBackend() override { Destroy(); }
    bool Init(uint32_t width, uint32_t height) override;
    void Clear(const State& s, uint32_t count, const uint8_t* rects, uint32_t flags, uint32_t color, float z, uint32_t stencil) override;
    void Draw(const State& s, uint32_t prim, uint32_t start, uint32_t count, const uint8_t* indices) override;
    void Copy(const Surface& src, const Surface& dst, uint32_t count, uint32_t src_rects, uint32_t dst_points) override;
    void Present(const Surface& front) override;
    void Sync(uint32_t data, uint32_t size) override;
    void SyncLate(uint32_t data, uint32_t size) override;
    bool DeliverLate(uint32_t data, uint32_t size) override;
    void RecordLate(uint32_t data) override;
    void Flush() override;
    void SetGamma(const uint8_t* ramp) override;
    bool WidensFrame() const override { return app::HostAspect() > 0.0f; }
private:
    struct Frame {
        VkCommandBuffer cmd = VK_NULL_HANDLE; VkFence fence = VK_NULL_HANDLE; bool fencePending = false;
        VkBuffer upload = VK_NULL_HANDLE; VkDeviceMemory uploadMem = VK_NULL_HANDLE; uint8_t* map = nullptr;
        // D6: uploadCoherent=false when the ring fell back to HOST_VISIBLE only; its mapped
        // range (and any non-coherent overflow allocation in uploadFlushMems) is flushed in
        // Submit before the queue sees it.
        bool uploadCoherent = true;
        std::vector<VkDeviceMemory> uploadFlushMems;
        uint64_t used = 0; VkDescriptorPool pool = VK_NULL_HANDLE; uint32_t sets = 0; uint64_t gen = 0;
        VkSemaphore imageAvailable = VK_NULL_HANDLE, renderFinished = VK_NULL_HANDLE;
        std::vector<std::function<void()>> garbage;
    };
    struct Alloc { VkBuffer buffer = VK_NULL_HANDLE; uint64_t off = 0; uint8_t* cpu = nullptr; };
    struct Late {
        ImageRes down; VkBuffer rb[kFrames] = {}; VkDeviceMemory rbm[kFrames] = {};
        // D2/D3: fence[i] is the submission number RecordLate predicts for the copy in rb[i],
        // submitFence[i] the VkFence that submission signals (Cur().fence, resolved on the
        // render thread while recording), rbCoherent[i] whether rbm[i] needs an invalidate
        // before a CPU read. A slot is finished only when its number has actually been
        // submitted (submitCounter_ reached it) and that fence signalled; a recorded but
        // unsubmitted copy is not finished.
        uint64_t fence[kFrames] = {}, frame[kFrames] = {}, delivered = 0;
        VkFence submitFence[kFrames] = {};
        bool rbCoherent[kFrames] = {true, true, true};
        uint32_t gw = 0, gh = 0, next = 0;
    };
    bool CreateDeviceAndSwap();
    bool CreateSwapchain();
    void DestroySwapchain();
    void ResizeSwapIfNeeded(bool force = false);
    uint32_t FindMemory(uint32_t typeBits, VkMemoryPropertyFlags props) const;
    // D6: host-visible memory for uploads/read-backs: coherent if there is any, else
    // HOST_VISIBLE only (coherent=false; caller flushes/invalidates explicitly).
    // UINT32_MAX when neither exists.
    uint32_t FindHostMemory(uint32_t typeBits, bool& coherent) const;
    // D7: a segment that never reached the queue leaves its RecordLate copies
    // recorded-but-never-submitted; drop their stamps so they are never delivered.
    void DiscardRecordedLate();
    ImageRes CreateImage(uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                         uint32_t layers = 1, VkImageType type = VK_IMAGE_TYPE_2D, uint32_t mips = 1);
    VkImageView MakeView(VkImage image, VkFormat fmt, VkImageViewType type, VkImageAspectFlags aspect, uint32_t levels, uint32_t layers);
    void DestroyImage(ImageRes& r);
    void DeferDestroyImage(ImageRes r);
    VkRenderPass RenderPassFor(VkFormat color, VkFormat depth, bool hasColor, bool hasDepth);
    Frame& Cur() { return frames_[frame_ % kFrames]; }
    void EnsureRecording();
    void CheckAllocGeneration(const char* what);
    void EnsureSetRoom();
    void Submit(bool wait, VkSemaphore waitSem = VK_NULL_HANDLE, VkSemaphore signalSem = VK_NULL_HANDLE);
    void WaitFrame(Frame& f);
    void RunGarbage(Frame& f);
    Alloc UploadAlloc(uint64_t size, uint64_t align = 256);
    void Transition(ImageRes& r, VkImageLayout want);
    void TransitionRaw(VkImage image, VkImageLayout& cur, VkImageLayout want, VkImageAspectFlags aspect);
    VkCommandBuffer Cmd() { EnsureRecording(); return Cur().cmd; }
    uint32_t WidenedGuestWidth() const;
    bool IsFrameTarget(const Target& t) const { return t.gw == guestW_ && t.gh == guestH_; }
    GuestExtent BoundExtent(const Surface& s) const;
    struct TargetSlot { uint32_t data; uint64_t key; uint32_t gw, gh; std::unordered_map<uint64_t, Target>::iterator it; bool sized; };
    TargetSlot FindTargetSlot(const Surface& s, bool depth, uint64_t alt);
    uint64_t SizedTargetKey(uint32_t data, bool depth, uint32_t gw, uint32_t gh) const;
    Target* FindTarget(const Surface& s, bool depth);
    Target* EnsureTarget(const Surface& s, bool depth);
    bool TargetStale(uint32_t data) const;
    void ReleaseTarget(std::unordered_map<uint64_t, Target>::iterator it);
    bool BindTargets(const State& s);
    void UploadTarget(Target& t, const Surface& s);
    ImageRes CreatePixelImage(const std::vector<uint8_t>& px, uint32_t pitch, uint32_t w, uint32_t h);
    VkShaderModule CompileModule(const std::string& src, const char* entry, bool fragment, uint64_t& hash);
    void EnsureVertexShader(const State& s, const Assembled& a, DrawShaders& sh);
    void EnsurePixelShader(const PixelKey& pk, DrawShaders& sh);
    VkPipeline EnsurePipeline(const PsoKey& k, VkShaderModule vs, VkShaderModule ps);
    VkPipeline EnsureBlitPipeline(VkFormat rtv, bool over, bool depthWrite = false, bool swap = false);
    VkSampler GetSampler(const VkSamplerCreateInfo& ci);
    VkSampler StageSamplerFor(const State& s, int i, const DrawTextures& stages, bool clampCoords);
    bool BindStageTextures(const State& s, DrawTextures& stages);
    VkImageView BindTexture(const State& s, int stage, DrawTextures& stages);
    VkImageView BindTargetTexture(const State& s, int stage, const TextureDesc& d, Target& t, bool box, DrawTextures& stages);
    VkImageView BindGuestTexture(const State& s, int stage, const TextureDesc& d, const FormatInfo& fi, DrawTextures& stages);
    bool CreateTexImage(Tex& tx, const TextureDesc& d, const FormatInfo& fi);
    void UploadTexImage(Tex& tx, const TextureDesc& d, const FormatInfo& fi, const uint8_t* p, const uint32_t* colours);
    ImageRes Feedback(Target& t);
    void FillConstants(const State& s, const DrawShaders& sh, const DrawTextures& stages, GuestExtent size, DrawConstants& c);
    VkDescriptorSet AllocDrawSet(VkBuffer vcBuf, uint64_t vcOff, VkBuffer dcBuf, uint64_t dcOff,
                                 const DrawTextures& stages, const std::array<VkSampler, 4>& samplers);
    void SetViewportScissor(const State& s, GuestExtent size, uint32_t targetW, uint32_t targetH);
    void DrawGeometry(const Assembled& a);
    void LogDraw(const State& s, uint32_t prim, uint32_t count, const Assembled& a, const DrawShaders& sh, const DrawTextures& stages,
                 const DrawConstants& c, GuestExtent size);
    void BeginTargetPass(Target* rt, Target* ds, VkFramebuffer& fbOut);
    void FlushPass();
    void PrepareBlitSource(ImageRes& src, bool gamma);
    void BlitInto(Target& dst, ImageRes& src, const float uv[4], bool gamma, bool alpha, uint32_t taps, float mul, bool over);
    void BlitToImage(VkFormat rtvFormat, uint32_t w, uint32_t h, ImageRes& src, const float uv[4],
                     bool gamma, bool alpha, uint32_t taps, float mul, bool over, const std::array<float, 4>* viewportRect,
                     bool swap = false);
    ImageRes ShrinkImage(ImageRes& src, uint32_t w, uint32_t h, uint32_t taps, float mul, ImageRes* reuse = nullptr);
    bool ReadBackImage(ImageRes& src, uint32_t w, uint32_t h, std::vector<uint8_t>& out);
    ImageRes PrepareCopySource(Target& t, ImageRes* reuseDown);
    bool DepthCopyDraw(const State& s);
    void UploadGammaRamp();
    void BlitFrame(Target& t, uint32_t w, uint32_t h);
    void RenderUi(VkCommandBuffer cmd);
    void ApplySettings();
    void ReleaseOldTextures();
    void CountFps();
    void LogPerf();
    void PeriodicCapture(Target* front);
    void SavePipelineCache();
    void CapSaveColor(Target& t, const char* name);
    void CapSaveColorAs(Target& t, const std::filesystem::path& file);
    void CapStep(const Surface& rt);
    void Destroy();

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t qfamily_ = 0;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat swapFormat_ = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D swapExtent_{};
    std::vector<VkImage> swapImages_;
    std::vector<VkImageView> swapViews_;
    std::vector<VkFramebuffer> swapFbs_;
    VkRenderPass swapPass_ = VK_NULL_HANDLE;
    bool swapVsync_ = true;
    bool swapStale_ = false;  // acquire/present said SUBOPTIMAL: recreate at the next present boundary
    ImageRes offscreen_;
    VkFramebuffer offscreenFb_ = VK_NULL_HANDLE;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    Frame frames_[kFrames];
    uint64_t frame_ = 0, presents_ = 0;
    bool recording_ = false; uint64_t segmentGen_ = 0;  // bumped every time EnsureRecording opens a segment
    // The open render pass. BeginTargetPass leaves it open and reuses it for the next draw on the same
    // attachments; FlushPass ends it, and every command that may not be recorded inside a render pass
    // calls FlushPass first. Beginning one per draw - with the framebuffer each needs - cost 8 ms a
    // frame of the arena's 13,000 draws.
    bool passOpen_ = false; Target* passRt_ = nullptr; Target* passDs_ = nullptr; VkFramebuffer passFb_ = VK_NULL_HANDLE;
    // The constants and descriptor set the last draw of this segment used. The arena submits thousands
    // of draws of about five vertices each, and long runs of them belong to one object, so they repeat
    // the same 3 KB constant block, the same per draw constants and the same textures. Re-uploading and
    // re-allocating a set for every one cost 10 ms a frame. EnsureRecording clears these: an allocation
    // is only valid in the segment that records its use.
    std::array<std::array<float, 4>, 192> vscLast_{}; Alloc vscAlloc_{}; bool vscValid_ = false;
    DrawConstants dcLast_{}; Alloc dcAlloc_{}; bool dcValid_ = false;
    std::array<VkImageView, 4> viewLast_{}; std::array<VkSampler, 4> samplerLast_{};
    VkDescriptorSet setLast_ = VK_NULL_HANDLE; bool setValid_ = false;
    // What is already bound in the open command buffer. Vulkan keeps a binding until something
    // replaces it, and consecutive draws of one object repeat all of it, so re-recording each bind
    // per draw is pure cost. InvalidateBinds forgets it: a reset command buffer holds no bindings,
    // and the blit and depth-copy paths bind their own pipeline, set, viewport and scissor.
    VkPipeline boundPipeline_ = VK_NULL_HANDLE;
    VkDescriptorSet boundSet_ = VK_NULL_HANDLE;
    float boundBlend_[4] = {-1, -1, -1, -1};
    VkViewport boundViewport_{}; VkRect2D boundScissor_{}; bool boundVpValid_ = false, boundScValid_ = false;
    // The geometry the last draw uploaded. The arena's draws are a few vertices each and runs of them
    // repeat one buffer, so re-uploading and re-binding it per draw cost 5 ms a frame.
    struct VbCache { const void* src = nullptr; uint32_t bytes = 0; Alloc alloc{}; };
    std::array<VbCache, 16> vbLast_{}; uint32_t vbCount_ = 0; bool vbValid_ = false;
    void InvalidateBinds() {
        boundPipeline_ = VK_NULL_HANDLE; boundSet_ = VK_NULL_HANDLE;
        boundBlend_[0] = boundBlend_[1] = boundBlend_[2] = boundBlend_[3] = -1;
        boundVpValid_ = boundScValid_ = false; vbValid_ = false;
    }
    // D2: submissions successfully queued, published by Submit (render thread) after
    // vkQueueSubmit and read by DeliverLate (title thread). This atomic counter, plus
    // the VkFence each Late slot resolved at record time, replaces the old submitFences_
    // unordered_map that DeliverLate find()ed while Submit inserted/erased it: there is
    // no shared mutable container left to race, and Submit pays one atomic store per
    // segment, nothing per draw.
    std::atomic<uint64_t> submitCounter_{0};
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineCache pipelineCache_ = VK_NULL_HANDLE;
    std::filesystem::path pipelineCacheFile_;
    VkDescriptorPool imguiPool_ = VK_NULL_HANDLE;
    VkFormat depthFormat_ = VK_FORMAT_D24_UNORM_S8_UINT;
    bool anisotropy_ = false, fillNonSolid_ = false, bc_ = false;
    std::unordered_map<uint64_t, VkShaderModule> modules_;
    // D4: hashes whose HLSL failed to compile this run (kept out of modules_, which
    // never holds a null module); a vkCreateShaderModule failure is not put here, so
    // the next bind retries it instead of caching the failure.
    std::unordered_set<uint64_t> failedModules_;
    std::unordered_map<uint64_t, uint64_t> shaderSrc_;
    std::unordered_map<uint64_t, VkPipeline> pipelines_, blitPipelines_;
    std::unordered_map<uint64_t, VkRenderPass> passes_;
    std::unordered_map<uint64_t, VkSampler> samplers_;
    std::unordered_map<uint64_t, Target> targets_;
    std::unordered_map<uint32_t, uint64_t> targetUsed_;
    std::unordered_map<uint64_t, Tex> textures_;
    std::unordered_map<uint32_t, Late> late_;
    // Recursive because Submit (which can run nested inside RecordLate's
    // PrepareCopySource flush) may need it on its D7 failure path while RecordLate
    // already holds it.
    std::recursive_mutex lateLock_;
    std::atomic<uint64_t> presentsSeen_{0};
    Target* boundRt_ = nullptr;
    Target* boundDs_ = nullptr;
    Target* drawRt_ = nullptr;
    Target* drawDs_ = nullptr;
    uint32_t guestW_ = 640, guestH_ = 480, bbW_ = 0, bbH_ = 0;
    float scale_ = 1.0f, widen_ = 1.0f, hostAspect_ = 0.0f;
    ImageRes lut_;
    uint8_t ramp_[768] = {};
    bool rampDirty_ = false, gamma_ = false;
    ImageRes nullImage_;
    VkSampler nullSampler_ = VK_NULL_HANDLE;
    Assembled asm_;
    FILE* cap_ = nullptr;
    std::filesystem::path capDir_;
    uint32_t capI_ = 0, capRt_ = 0;
    double fps_ = 0;
    std::chrono::steady_clock::time_point fpsT_ = std::chrono::steady_clock::now();
    uint64_t fpsN_ = 0;
    VkPerfClock::time_point perfT_ = VkPerfClock::now();
    uint64_t perfFrames_ = 0;
    uint64_t presentLogged_ = 0;
    bool destroyed_ = false;
};
}  // namespace

// The handover functions of vulkan.h: external, in gpu::, over the anonymous state above.
void SetVulkanInstanceExtensions(const char* const* extensions, uint32_t count) {
    std::lock_guard lock(g_handover_lock);
    g_extension_storage.clear(); g_extensions.clear();
    for (uint32_t i = 0; i < count; i++) g_extension_storage.emplace_back(extensions[i]);
    for (auto& e : g_extension_storage) g_extensions.push_back(e.c_str());
}
VkInstance VulkanInstance() {
    std::lock_guard lock(g_handover_lock);
    if (g_instance) return g_instance;
    std::vector<const char*> exts = g_extensions;
    auto add = [&](const char* name) {
        if (std::find_if(exts.begin(), exts.end(), [&](const char* e) { return !std::strcmp(e, name); }) == exts.end()) exts.push_back(name);
    };
    add(VK_KHR_SURFACE_EXTENSION_NAME);
    uint32_t count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> avail(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, avail.data());
    for (auto& a : avail) if (!std::strcmp(a.extensionName, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)) add(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    std::vector<const char*> layers;
    if (VkDebugEnabled()) {
        uint32_t lcount = 0; vkEnumerateInstanceLayerProperties(&lcount, nullptr);
        std::vector<VkLayerProperties> lavail(lcount); vkEnumerateInstanceLayerProperties(&lcount, lavail.data());
        bool haveLayer = false;
        for (auto& l : lavail) if (!std::strcmp(l.layerName, "VK_LAYER_KHRONOS_validation")) haveLayer = true;
        if (haveLayer) layers.push_back("VK_LAYER_KHRONOS_validation");
        else LOGW("vulkan: XBE_VK_DEBUG set but VK_LAYER_KHRONOS_validation is not available");
        bool haveDebug = false;
        for (auto& a : avail) if (!std::strcmp(a.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) haveDebug = true;
        if (haveDebug) add(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        else LOGW("vulkan: XBE_VK_DEBUG set but VK_EXT_debug_utils is not available");
    }
    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "xbe-recomp"; appInfo.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &appInfo; ci.enabledExtensionCount = uint32_t(exts.size()); ci.ppEnabledExtensionNames = exts.data();
    ci.enabledLayerCount = uint32_t(layers.size()); ci.ppEnabledLayerNames = layers.data();
    VkDebugUtilsMessengerCreateInfoEXT msgCi{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    if (VkDebugEnabled() && !layers.empty()) {
        msgCi.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        msgCi.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        msgCi.pfnUserCallback = VkDebugCallback;
        ci.pNext = &msgCi;
    }
    if (vkCreateInstance(&ci, nullptr, &g_instance) != VK_SUCCESS) { LOGE("vulkan: vkCreateInstance failed"); g_instance = VK_NULL_HANDLE; return g_instance; }
    if (VkDebugEnabled() && !layers.empty() && g_instance) {
        auto fn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(g_instance, "vkCreateDebugUtilsMessengerEXT"));
        if (fn && fn(g_instance, &msgCi, nullptr, &g_vkMessenger) != VK_SUCCESS) g_vkMessenger = VK_NULL_HANDLE;
        LOGI("vulkan: XBE_VK_DEBUG validation layer + debug utils enabled (messenger %p)", (void*)g_vkMessenger);
    }
    return g_instance;
}
void SetVulkanSurface(VkInstance instance, VkSurfaceKHR surface) {
    std::lock_guard lock(g_handover_lock);
    g_instance = instance; g_surface = surface;
    g_handles.instance = instance; g_handles.surface = surface;
}
VkSurfaceKHR VulkanSurface() { std::lock_guard lock(g_handover_lock); return g_surface; }
const VulkanHandles& GetVulkanHandles() { return g_handles; }

namespace {
uint32_t VulkanBackend::FindMemory(uint32_t typeBits, VkMemoryPropertyFlags props) const {
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(physical_, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) return i;
    // D6: no match is UINT32_MAX, never type 0: type 0 may not be in typeBits at all, and
    // every caller now bails on the sentinel instead of binding/mapping through it.
    return UINT32_MAX;
}
uint32_t VulkanBackend::FindHostMemory(uint32_t typeBits, bool& coherent) const {
    coherent = true;
    uint32_t t = FindMemory(typeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (t != UINT32_MAX) return t;
    // D6: fall back to HOST_VISIBLE alone, not to an arbitrary type; the caller then
    // flushes before GPU reads and invalidates before CPU reads.
    coherent = false;
    return FindMemory(typeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
}
VkImageView VulkanBackend::MakeView(VkImage image, VkFormat fmt, VkImageViewType type, VkImageAspectFlags aspect, uint32_t levels, uint32_t layers) {
    VkImageViewCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    ci.image = image; ci.viewType = type; ci.format = fmt;
    ci.subresourceRange.aspectMask = aspect; ci.subresourceRange.levelCount = levels; ci.subresourceRange.layerCount = layers;
    VkImageView view = VK_NULL_HANDLE; vkCreateImageView(device_, &ci, nullptr, &view); return view;
}
ImageRes VulkanBackend::CreateImage(uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                                    uint32_t layers, VkImageType type, uint32_t mips) {
    ImageRes r; r.format = fmt; r.w = std::max(w, 1u); r.h = std::max(h, 1u); r.layers = layers;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = type; ci.format = fmt; ci.extent = {r.w, r.h, type == VK_IMAGE_TYPE_3D ? layers : 1};
    ci.mipLevels = mips; ci.arrayLayers = type == VK_IMAGE_TYPE_3D ? 1 : layers; ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL; ci.usage = usage; ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (layers == 6 && type == VK_IMAGE_TYPE_2D) ci.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    if (vkCreateImage(device_, &ci, nullptr, &r.image) != VK_SUCCESS) return r;
    VkMemoryRequirements mr; vkGetImageMemoryRequirements(device_, r.image, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size; ai.memoryTypeIndex = FindMemory(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    // D6: bail cleanly (destroying the partial image) when no type fits, allocation
    // fails, or the bind fails, instead of handing out an image bound to nothing.
    if (ai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(device_, &ai, nullptr, &r.mem) != VK_SUCCESS) {
        vkDestroyImage(device_, r.image, nullptr); r.image = VK_NULL_HANDLE; return r;
    }
    if (vkBindImageMemory(device_, r.image, r.mem, 0) != VK_SUCCESS) { DestroyImage(r); return r; }
    VkImageViewType vt = type == VK_IMAGE_TYPE_3D ? VK_IMAGE_VIEW_TYPE_3D : layers == 6 ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D;
    r.view = MakeView(r.image, fmt, vt, aspect, mips, type == VK_IMAGE_TYPE_3D ? 1 : layers);
    if (!r.view) { DestroyImage(r); return r; }
    if (aspect & VK_IMAGE_ASPECT_DEPTH_BIT) {
        r.srvView = MakeView(r.image, fmt, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT, 1, 1);
        if (!r.srvView) { DestroyImage(r); return r; }
    }
    return r;
}
void VulkanBackend::DestroyImage(ImageRes& r) {
    if (!device_) return;
    if (r.srvView) vkDestroyImageView(device_, r.srvView, nullptr);
    if (r.view) vkDestroyImageView(device_, r.view, nullptr);
    if (r.image) vkDestroyImage(device_, r.image, nullptr);
    if (r.mem) vkFreeMemory(device_, r.mem, nullptr);
    r = ImageRes{};
}
void VulkanBackend::DeferDestroyImage(ImageRes r) { if (r.image) Cur().garbage.push_back([this, r]() mutable { DestroyImage(r); }); }
void VulkanBackend::TransitionRaw(VkImage image, VkImageLayout& cur, VkImageLayout want, VkImageAspectFlags aspect) {
    if (cur == want) return;
    FlushPass();  // a barrier may not be recorded inside a render pass
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = cur; b.newLayout = want; b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image; b.subresourceRange.aspectMask = aspect;
    b.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS; b.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
    auto stageAccess = [](VkImageLayout l, VkAccessFlags& a, VkPipelineStageFlags& s) {
        switch (l) {
            case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: a = VK_ACCESS_TRANSFER_WRITE_BIT; s = VK_PIPELINE_STAGE_TRANSFER_BIT; break;
            case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL: a = VK_ACCESS_TRANSFER_READ_BIT; s = VK_PIPELINE_STAGE_TRANSFER_BIT; break;
            case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL: a = VK_ACCESS_SHADER_READ_BIT; s = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT; break;
            case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL: a = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT; s = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; break;
            case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL: a = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT; s = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT; break;
            case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR: a = 0; s = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT; break;
            default: a = 0; s = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT; break;
        }
    };
    VkAccessFlags sa, da; VkPipelineStageFlags ss, ds; stageAccess(cur, sa, ss); stageAccess(want, da, ds);
    b.srcAccessMask = sa; b.dstAccessMask = da;
    vkCmdPipelineBarrier(Cmd(), ss, ds, 0, 0, nullptr, 0, nullptr, 1, &b);
    cur = want;
}
void VulkanBackend::Transition(ImageRes& r, VkImageLayout want) {
    bool depth = r.format == VK_FORMAT_D24_UNORM_S8_UINT || r.format == VK_FORMAT_D32_SFLOAT_S8_UINT || r.format == VK_FORMAT_D32_SFLOAT || r.format == VK_FORMAT_D16_UNORM;
    VkImageAspectFlags aspect = depth ? VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT | (r.format == VK_FORMAT_D24_UNORM_S8_UINT || r.format == VK_FORMAT_D32_SFLOAT_S8_UINT ? VK_IMAGE_ASPECT_STENCIL_BIT : 0)) : VkImageAspectFlags(VK_IMAGE_ASPECT_COLOR_BIT);
    TransitionRaw(r.image, r.layout, want, aspect);
}
VkRenderPass VulkanBackend::RenderPassFor(VkFormat color, VkFormat depth, bool hasColor, bool hasDepth) {
    uint64_t key = (uint64_t(hasColor) << 63) | (uint64_t(hasDepth) << 62) | (uint64_t(color) << 20) | uint64_t(depth);
    auto it = passes_.find(key); if (it != passes_.end()) return it->second;
    VkAttachmentDescription at[2]{}; VkAttachmentReference refs[2]{}; uint32_t n = 0;
    const VkAttachmentReference* colorRef = nullptr; const VkAttachmentReference* depthRef = nullptr;
    if (hasColor) {
        at[n].format = color; at[n].samples = VK_SAMPLE_COUNT_1_BIT; at[n].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        at[n].storeOp = VK_ATTACHMENT_STORE_OP_STORE; at[n].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        at[n].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; refs[n] = {n, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        colorRef = &refs[n]; n++;
    }
    if (hasDepth) {
        at[n].format = depth; at[n].samples = VK_SAMPLE_COUNT_1_BIT; at[n].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        at[n].storeOp = VK_ATTACHMENT_STORE_OP_STORE; at[n].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        at[n].stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE; at[n].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        at[n].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL; refs[n] = {n, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        depthRef = &refs[n]; n++;
    }
    VkSubpassDescription sub{}; sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = hasColor ? 1 : 0; sub.pColorAttachments = colorRef; sub.pDepthStencilAttachment = depthRef;
    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL; dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstStageMask = dep.srcStageMask;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    ci.attachmentCount = n; ci.pAttachments = at; ci.subpassCount = 1; ci.pSubpasses = &sub; ci.dependencyCount = 1; ci.pDependencies = &dep;
    VkRenderPass rp = VK_NULL_HANDLE; vkCreateRenderPass(device_, &ci, nullptr, &rp); passes_[key] = rp; return rp;
}
void VulkanBackend::EnsureRecording() {
    if (recording_) return;
    Frame& f = Cur();
    if (f.fencePending) WaitFrame(f);
    vkResetCommandBuffer(f.cmd, 0); vkResetDescriptorPool(device_, f.pool, 0); f.used = 0; f.sets = 0; f.gen = ++segmentGen_; f.uploadFlushMems.clear();
    passOpen_ = false;  // the reset command buffer holds no pass
    vscValid_ = dcValid_ = setValid_ = false;  // and none of the last segment's allocations
    InvalidateBinds();                         // nor any of its bindings
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(f.cmd, &bi); recording_ = true;
}
// An allocation is only valid in the segment that will record its use: it must be
// handed out while that segment is open, stamped with its generation. Anything
// handed out before EnsureRecording opens the segment is reset away by it - the
// upload ring offset is reused and the descriptor set freed - which is the stale
// case this counts. Both allocators call EnsureRecording first, so this stays 0;
// it fires if that ordering is ever broken again.
void VulkanBackend::CheckAllocGeneration(const char* what) {
    Frame& f = Cur();
    if (recording_ && f.gen == segmentGen_) return;
    g_vkPerf.staleAlloc++;
    LOG_FIRST(16, "vulkan: stale %s allocation: no open segment (frame gen %llu, segment gen %llu, present %llu)", what, (unsigned long long)f.gen, (unsigned long long)segmentGen_, (unsigned long long)presents_);
}
// Outside any render pass, before work that allocates a descriptor set (and before
// its uploads, since a flush restarts the upload ring as well): a segment that has
// used its pool is submitted here, because inside a pass it cannot be.
void VulkanBackend::EnsureSetRoom() { if (recording_ && Cur().sets >= kSetsPerSegment) Flush(); }
void VulkanBackend::RunGarbage(Frame& f) { for (auto& fn : f.garbage) fn(); f.garbage.clear(); }
void VulkanBackend::WaitFrame(Frame& f) {
    if (f.fencePending) { vkWaitForFences(device_, 1, &f.fence, VK_TRUE, UINT64_MAX); f.fencePending = false; }
    RunGarbage(f);
}
void VulkanBackend::DiscardRecordedLate() {
    // D7/D3: called only when a segment never reached the queue. Every Late slot
    // stamped ahead of submitCounter_ was recorded into that segment and will never
    // run, so clear its stamp: DeliverLate must not deliver its buffer, and
    // RecordLate may reuse the slot (the GPU never wrote it).
    std::lock_guard lock(lateLock_);
    uint64_t submitted = submitCounter_.load(std::memory_order_acquire);
    for (auto& [data, l] : late_) {
        (void)data;
        for (int i = 0; i < kFrames; i++)
            if (l.fence[i] > submitted) { l.fence[i] = 0; l.submitFence[i] = VK_NULL_HANDLE; }
    }
}
void VulkanBackend::Submit(bool wait, VkSemaphore waitSem, VkSemaphore signalSem) {
    if (!recording_) { if (wait && device_) vkQueueWaitIdle(queue_); return; }
    FlushPass();  // a segment may not end inside a render pass
    Frame& f = Cur();
    // D7: every result is checked. On any failure the fence is left not pending and
    // submitCounter_ does not advance, so the next WaitFrame cannot block forever
    // on a fence that was reset but never submitted; the failure is logged loudly
    // here, once, at its cause (the backend interface returns void).
    VkResult endR = vkEndCommandBuffer(f.cmd); recording_ = false;
    if (endR != VK_SUCCESS) {
        LOGE("vulkan: vkEndCommandBuffer failed %d; segment dropped, fence not marked pending", int(endR));
        DiscardRecordedLate();
        return;
    }
    VkResult resetR = vkResetFences(device_, 1, &f.fence);
    if (resetR != VK_SUCCESS) {
        LOGE("vulkan: vkResetFences failed %d; segment not submitted, fence not marked pending", int(resetR));
        DiscardRecordedLate();
        return;
    }
    // D6: non-coherent upload memory must be flushed before the queue reads it.
    if (!f.uploadCoherent && f.uploadMem) {
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = f.uploadMem; range.offset = 0; range.size = VK_WHOLE_SIZE;
        vkFlushMappedMemoryRanges(device_, 1, &range);
    }
    for (VkDeviceMemory mem : f.uploadFlushMems) {
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = mem; range.offset = 0; range.size = VK_WHOLE_SIZE;
        vkFlushMappedMemoryRanges(device_, 1, &range);
    }
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &f.cmd;
    if (waitSem) { si.waitSemaphoreCount = 1; si.pWaitSemaphores = &waitSem; si.pWaitDstStageMask = &stage; }
    if (signalSem) { si.signalSemaphoreCount = 1; si.pSignalSemaphores = &signalSem; }
    VkResult submitR = vkQueueSubmit(queue_, 1, &si, f.fence);
    if (submitR != VK_SUCCESS) {
        LOGE("vulkan: vkQueueSubmit failed %d; fence left unsignalled and not marked pending, submission not counted", int(submitR));
        DiscardRecordedLate();
        return;
    }
    f.fencePending = true;
    // D2: publish the submission only after vkQueueSubmit succeeded; DeliverLate
    // (title thread) reads this atomic and the per-slot VkFence, never a shared map.
    submitCounter_.fetch_add(1, std::memory_order_release);
    if (wait) { WaitFrame(f); f.used = 0; }
}
void VulkanBackend::Flush() { Submit(true); }
VulkanBackend::Alloc VulkanBackend::UploadAlloc(uint64_t size, uint64_t align) {
    // The ring belongs to the open segment: allocating after a submit but before the
    // next segment starts would hand out space that segment's reset then reuses.
    EnsureRecording(); CheckAllocGeneration("upload");
    Frame& f = Cur();
    uint64_t o = (f.used + align - 1) & ~(align - 1);
    if (o + size <= kUploadSize) { f.used = o + size; g_vkPerf.upload += size; return {f.upload, o, f.map + o}; }
    g_vkPerf.overflow++;
    VkBuffer buf = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE;
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = std::max<uint64_t>(size, 1);
    ci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    if (vkCreateBuffer(device_, &ci, nullptr, &buf) != VK_SUCCESS) return {};
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(device_, buf, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    // D6: no suitable host memory, a failed allocate/bind, or a failed map hands out
    // nothing (callers already drop the draw on a null cpu) instead of arithmetic on null.
    bool coherent = true;
    ai.allocationSize = mr.size; ai.memoryTypeIndex = FindHostMemory(mr.memoryTypeBits, coherent);
    if (ai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(device_, &ai, nullptr, &mem) != VK_SUCCESS) { vkDestroyBuffer(device_, buf, nullptr); return {}; }
    if (vkBindBufferMemory(device_, buf, mem, 0) != VK_SUCCESS) { vkDestroyBuffer(device_, buf, nullptr); vkFreeMemory(device_, mem, nullptr); return {}; }
    void* p = nullptr;
    if (vkMapMemory(device_, mem, 0, VK_WHOLE_SIZE, 0, &p) != VK_SUCCESS || !p) { vkDestroyBuffer(device_, buf, nullptr); vkFreeMemory(device_, mem, nullptr); return {}; }
    if (!coherent) f.uploadFlushMems.push_back(mem);  // D6: flushed in Submit
    f.garbage.push_back([this, buf, mem] { vkUnmapMemory(device_, mem); vkDestroyBuffer(device_, buf, nullptr); vkFreeMemory(device_, mem, nullptr); });
    g_vkPerf.upload += size;
    return {buf, 0, static_cast<uint8_t*>(p)};
}
// ---- Init ----
bool VulkanBackend::Init(uint32_t width, uint32_t height) {
    guestW_ = width; guestH_ = height;
    { std::lock_guard lock(g_handover_lock); instance_ = g_instance ? g_instance : VulkanInstance(); surface_ = g_surface; }
    if (!instance_) return false;
    if (!CreateDeviceAndSwap()) return false;
    {
        VkDescriptorSetLayoutBinding b[10]{};
        for (int i = 0; i < 10; i++) { b[i].binding = uint32_t(i); b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT; }
        b[0].descriptorType = b[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        for (int i = 2; i < 6; i++) b[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        for (int i = 6; i < 10; i++) b[i].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.bindingCount = 10; ci.pBindings = b;
        vkCreateDescriptorSetLayout(device_, &ci, nullptr, &setLayout_);
        VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        li.setLayoutCount = 1; li.pSetLayouts = &setLayout_;
        vkCreatePipelineLayout(device_, &li, nullptr, &pipelineLayout_);
    }
    {
        VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; ci.queueFamilyIndex = qfamily_;
        vkCreateCommandPool(device_, &ci, nullptr, &pool_);
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        for (auto& f : frames_) {
            VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            ai.commandPool = pool_; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
            vkAllocateCommandBuffers(device_, &ai, &f.cmd);
            VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            vkCreateFence(device_, &fi, nullptr, &f.fence);
            vkCreateSemaphore(device_, &si, nullptr, &f.imageAvailable);
            vkCreateSemaphore(device_, &si, nullptr, &f.renderFinished);
            // kSetsPerSegment sets of the one layout (2 uniform + 4 combined + 4
            // sampler descriptors each), so the set count is the only limit
            VkDescriptorPoolSize sizes[3] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2 * kSetsPerSegment}, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4 * kSetsPerSegment}, {VK_DESCRIPTOR_TYPE_SAMPLER, 4 * kSetsPerSegment}};
            VkDescriptorPoolCreateInfo di{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            di.maxSets = kSetsPerSegment; di.poolSizeCount = 3; di.pPoolSizes = sizes;
            vkCreateDescriptorPool(device_, &di, nullptr, &f.pool);
            VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bi.size = kUploadSize;
            bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
            // D6: a ring that cannot be created, allocated, bound or mapped fails Init
            // outright; otherwise UploadAlloc would hand out f.map + o on a null map and
            // the first draw's memcpy would fault. Partial handles are destroyed and
            // nulled so Destroy does not touch them again.
            if (vkCreateBuffer(device_, &bi, nullptr, &f.upload) != VK_SUCCESS) {
                LOGE("vulkan: upload ring buffer creation failed; Init fails (D6)");
                return false;
            }
            VkMemoryRequirements mr; vkGetBufferMemoryRequirements(device_, f.upload, &mr);
            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            mai.allocationSize = mr.size; mai.memoryTypeIndex = FindHostMemory(mr.memoryTypeBits, f.uploadCoherent);
            if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(device_, &mai, nullptr, &f.uploadMem) != VK_SUCCESS) {
                LOGE("vulkan: upload ring allocation failed (no host-visible memory type or out of memory); Init fails (D6)");
                vkDestroyBuffer(device_, f.upload, nullptr); f.upload = VK_NULL_HANDLE;
                return false;
            }
            if (vkBindBufferMemory(device_, f.upload, f.uploadMem, 0) != VK_SUCCESS ||
                vkMapMemory(device_, f.uploadMem, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void**>(&f.map)) != VK_SUCCESS || !f.map) {
                LOGE("vulkan: upload ring bind/map failed; Init fails (D6)");
                f.map = nullptr;
                vkDestroyBuffer(device_, f.upload, nullptr); f.upload = VK_NULL_HANDLE;
                vkFreeMemory(device_, f.uploadMem, nullptr); f.uploadMem = VK_NULL_HANDLE;
                return false;
            }
        }
    }
    nullImage_ = CreateImage(1, 1, kBgra, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    {
        Transition(nullImage_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}; VkClearColorValue black{};
        vkCmdClearColorImage(Cmd(), nullImage_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
        Transition(nullImage_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO}; si.magFilter = si.minFilter = VK_FILTER_NEAREST;
        vkCreateSampler(device_, &si, nullptr, &nullSampler_);
    }
    lut_ = CreateImage(256, 1, kBgra, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    for (int i = 0; i < 256; i++) ramp_[i] = ramp_[256 + i] = ramp_[512 + i] = uint8_t(i);
    rampDirty_ = true;
    scale_ = app::RenderScale(guestH_); widen_ = app::AspectWiden(); hostAspect_ = app::HostAspect();
    {
        VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(physical_, &props);
        char key[192];
        std::snprintf(key, sizeof(key), "%04x-%04x-drv%u.%u.%u-%02x%02x%02x%02x", props.vendorID, props.deviceID,
                      VK_VERSION_MAJOR(props.driverVersion), VK_VERSION_MINOR(props.driverVersion), VK_VERSION_PATCH(props.driverVersion),
                      props.pipelineCacheUUID[0], props.pipelineCacheUUID[1], props.pipelineCacheUUID[2], props.pipelineCacheUUID[3]);
        std::filesystem::path cacheDir = app::ExeDir() / "shadercache";
        SetSpirvCacheDir(cacheDir, key);
        pipelineCacheFile_ = cacheDir / key / "pipeline.bin";
        std::vector<uint8_t> blob;
        if (std::ifstream f(pipelineCacheFile_, std::ios::binary | std::ios::ate); f) {
            std::streamsize n = f.tellg();
            if (n > 0) { blob.resize(size_t(n)); f.seekg(0); f.read(reinterpret_cast<char*>(blob.data()), n); }
        }
        VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        ci.initialDataSize = blob.size(); ci.pInitialData = blob.data();
        if (vkCreatePipelineCache(device_, &ci, nullptr, &pipelineCache_) != VK_SUCCESS) pipelineCache_ = VK_NULL_HANDLE;
        LOGR("Vulkan adapter: %s (vendor %04x device %04x, driver %u)", props.deviceName, props.vendorID, props.deviceID, props.driverVersion);
        VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1024}, {VK_DESCRIPTOR_TYPE_SAMPLER, 1024}};
        VkDescriptorPoolCreateInfo di{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        di.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT; di.maxSets = 1024; di.poolSizeCount = 2; di.pPoolSizes = ps;
        vkCreateDescriptorPool(device_, &di, nullptr, &imguiPool_);
        g_handles.instance = instance_; g_handles.physicalDevice = physical_; g_handles.device = device_;
        g_handles.graphicsQueue = queue_; g_handles.graphicsQueueFamily = qfamily_;
        g_handles.descriptorPool = imguiPool_; g_handles.pipelineCache = pipelineCache_;
        g_handles.renderPass = swapPass_; g_handles.dynamicRendering = false; g_handles.colorFormat = swapFormat_;
        g_handles.minImageCount = std::max<uint32_t>(uint32_t(swapImages_.size()), 2);
        g_handles.imageCount = uint32_t(swapImages_.size()); g_handles.msaaSamples = VK_SAMPLE_COUNT_1_BIT; g_handles.surface = surface_;
    }
    LOGI("vulkan: %ux%u guest, %ux%u window, render scale %.3f, HLSL->SPIR-V by DXC (libdxcompiler) with glslang fallback", width, height, bbW_, bbH_, scale_);
    return true;
}
bool VulkanBackend::CreateDeviceAndSwap() {
    uint32_t count = 0; vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count); vkEnumeratePhysicalDevices(instance_, &count, devices.data());
    int bestScore = -1; uint32_t bestFamily = 0;
    for (auto dev : devices) {
        VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(dev, &props);
        uint32_t qcount = 0; vkGetPhysicalDeviceQueueFamilyProperties(dev, &qcount, nullptr);
        std::vector<VkQueueFamilyProperties> qfs(qcount); vkGetPhysicalDeviceQueueFamilyProperties(dev, &qcount, qfs.data());
        for (uint32_t i = 0; i < qcount; i++) {
            if (!(qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
            if (surface_) { VkBool32 present = VK_FALSE; vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface_, &present); if (!present) continue; }
            int score = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 30 : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 20 : 10;
            VkPhysicalDeviceFeatures feats; vkGetPhysicalDeviceFeatures(dev, &feats);
            if (feats.textureCompressionBC) score += 5;
            if (score > bestScore) { bestScore = score; physical_ = dev; bestFamily = i; }
            break;
        }
    }
    if (!physical_) { LOGE("vulkan: no physical device with a graphics queue"); return false; }
    qfamily_ = bestFamily;
    VkPhysicalDeviceFeatures feats; vkGetPhysicalDeviceFeatures(physical_, &feats);
    anisotropy_ = feats.samplerAnisotropy; fillNonSolid_ = feats.fillModeNonSolid; bc_ = feats.textureCompressionBC;
    std::vector<const char*> exts;
    if (surface_) exts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = qfamily_; qi.queueCount = 1; qi.pQueuePriorities = &prio;
    VkPhysicalDeviceFeatures want{}; want.samplerAnisotropy = anisotropy_; want.fillModeNonSolid = fillNonSolid_; want.textureCompressionBC = bc_; want.depthClamp = feats.depthClamp;
    VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    ci.queueCreateInfoCount = 1; ci.pQueueCreateInfos = &qi; ci.enabledExtensionCount = uint32_t(exts.size()); ci.ppEnabledExtensionNames = exts.data(); ci.pEnabledFeatures = &want;
    if (vkCreateDevice(physical_, &ci, nullptr, &device_) != VK_SUCCESS) { LOGE("vulkan: vkCreateDevice failed"); return false; }
    vkGetDeviceQueue(device_, qfamily_, 0, &queue_);
    for (VkFormat f : {VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D16_UNORM}) {
        VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(physical_, f, &fp);
        if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) { depthFormat_ = f; break; }
    }
    if (surface_) return CreateSwapchain();
    bbW_ = guestW_; bbH_ = guestH_; swapExtent_ = {bbW_, bbH_}; swapFormat_ = kBgra;
    swapPass_ = RenderPassFor(swapFormat_, VK_FORMAT_UNDEFINED, true, false);
    // headless: the swap pass above loads; the offscreen present still works, nothing is shown
    offscreen_ = CreateImage(bbW_, bbH_, swapFormat_, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fi.renderPass = swapPass_; fi.attachmentCount = 1; fi.pAttachments = &offscreen_.view; fi.width = bbW_; fi.height = bbH_; fi.layers = 1;
    vkCreateFramebuffer(device_, &fi, nullptr, &offscreenFb_);
    return true;
}
bool VulkanBackend::CreateSwapchain() {
    VkSurfaceCapabilitiesKHR caps;
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_, surface_, &caps) != VK_SUCCESS) return false;
    uint32_t fcount = 0; vkGetPhysicalDeviceSurfaceFormatsKHR(physical_, surface_, &fcount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(fcount); vkGetPhysicalDeviceSurfaceFormatsKHR(physical_, surface_, &fcount, formats.data());
    VkSurfaceFormatKHR chosen = formats.empty() ? VkSurfaceFormatKHR{kBgra, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR} : formats[0];
    for (auto& f : formats) if (f.format == kBgra && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) chosen = f;
    swapFormat_ = chosen.format; swapExtent_ = caps.currentExtent;
    if (swapExtent_.width == UINT32_MAX) {
        // Wayland: currentExtent is undefined and the swapchain extent becomes
        // the window size, so it must be the window's drawable size in pixels.
        VkExtent2D drawable = DrawableExtent();
        swapExtent_ = {std::clamp(drawable.width, caps.minImageExtent.width, caps.maxImageExtent.width),
                       std::clamp(drawable.height, caps.minImageExtent.height, caps.maxImageExtent.height)};
    }
    swapVsync_ = app::Get().vsync;
    uint32_t pcount = 0; vkGetPhysicalDeviceSurfacePresentModesKHR(physical_, surface_, &pcount, nullptr);
    std::vector<VkPresentModeKHR> modes(pcount); vkGetPhysicalDeviceSurfacePresentModesKHR(physical_, surface_, &pcount, modes.data());
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
    if (!swapVsync_) { for (auto m : modes) if (m == VK_PRESENT_MODE_MAILBOX_KHR) mode = m; for (auto m : modes) if (m == VK_PRESENT_MODE_IMMEDIATE_KHR) mode = m; }
    uint32_t imageCount = std::max<uint32_t>(kFrames, caps.minImageCount);
    if (caps.maxImageCount) imageCount = std::min(imageCount, caps.maxImageCount);
    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface = surface_; ci.minImageCount = imageCount; ci.imageFormat = chosen.format; ci.imageColorSpace = chosen.colorSpace;
    ci.imageExtent = swapExtent_; ci.imageArrayLayers = 1; ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE; ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; ci.presentMode = mode; ci.clipped = VK_TRUE;
    if (vkCreateSwapchainKHR(device_, &ci, nullptr, &swapchain_) != VK_SUCCESS) { LOGE("vulkan: no swapchain"); return false; }
    uint32_t n = 0; vkGetSwapchainImagesKHR(device_, swapchain_, &n, nullptr);
    swapImages_.resize(n); vkGetSwapchainImagesKHR(device_, swapchain_, &n, swapImages_.data());
    if (!swapPass_) {
        VkAttachmentDescription at{}; at.format = swapFormat_; at.samples = VK_SAMPLE_COUNT_1_BIT;
        at.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; at.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        at.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; at.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sub{}; sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS; sub.colorAttachmentCount = 1; sub.pColorAttachments = &ref;
        // External dependencies make the UNDEFINED->COLOR_ATTACHMENT->PRESENT transitions
        // synchronised with the acquire semaphore (waited on at COLOR_ATTACHMENT_OUTPUT)
        // and with the present engine's read. Without them the layout transition is
        // unsynchronised and the presented image can be read while it is written.
        VkSubpassDependency deps[2]{};
        deps[0].srcSubpass = VK_SUBPASS_EXTERNAL; deps[0].dstSubpass = 0;
        deps[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        deps[0].srcAccessMask = 0; deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        deps[1].srcSubpass = 0; deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; deps[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; deps[1].dstAccessMask = 0;
        VkRenderPassCreateInfo ri{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        ri.attachmentCount = 1; ri.pAttachments = &at; ri.subpassCount = 1; ri.pSubpasses = &sub;
        ri.dependencyCount = 2; ri.pDependencies = deps;
        vkCreateRenderPass(device_, &ri, nullptr, &swapPass_);
    }
    swapViews_.assign(n, VK_NULL_HANDLE); swapFbs_.assign(n, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < n; i++) {
        swapViews_[i] = MakeView(swapImages_[i], swapFormat_, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, 1, 1);
        VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fi.renderPass = swapPass_; fi.attachmentCount = 1; fi.pAttachments = &swapViews_[i]; fi.width = swapExtent_.width; fi.height = swapExtent_.height; fi.layers = 1;
        vkCreateFramebuffer(device_, &fi, nullptr, &swapFbs_[i]);
    }
    bbW_ = swapExtent_.width; bbH_ = swapExtent_.height;
    { app::platform::Size drawable = app::platform::ClientSize();
      LOGI("vulkan: swapchain %ux%u (window drawable %ux%u, surface currentExtent %ux%u)", bbW_, bbH_, drawable.width, drawable.height, caps.currentExtent.width, caps.currentExtent.height); }
    return true;
}
void VulkanBackend::DestroySwapchain() {
    if (!device_) return;
    for (auto fb : swapFbs_) if (fb) vkDestroyFramebuffer(device_, fb, nullptr);
    for (auto v : swapViews_) if (v) vkDestroyImageView(device_, v, nullptr);
    swapFbs_.clear(); swapViews_.clear(); swapImages_.clear();
    if (swapchain_) vkDestroySwapchainKHR(device_, swapchain_, nullptr);
    swapchain_ = VK_NULL_HANDLE;
}
void VulkanBackend::ResizeSwapIfNeeded(bool force) {
    if (!surface_ || !swapchain_) return;
    VkSurfaceCapabilitiesKHR caps;
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_, surface_, &caps) != VK_SUCCESS) return;
    bool vsync = app::Get().vsync; VkExtent2D want = caps.currentExtent;
    if (want.width == UINT32_MAX) want = DrawableExtent();  // Wayland: the window size is the authority, every time
    if (!want.width || !want.height) return;  // no window yet, or minimised to zero
    want = {std::clamp(want.width, caps.minImageExtent.width, caps.maxImageExtent.width),
            std::clamp(want.height, caps.minImageExtent.height, caps.maxImageExtent.height)};
    if (!force && want.width == swapExtent_.width && want.height == swapExtent_.height && vsync == swapVsync_) return;
    Flush(); vkDeviceWaitIdle(device_); DestroySwapchain(); CreateSwapchain();
    g_handles.renderPass = swapPass_; g_handles.colorFormat = swapFormat_;
    g_handles.imageCount = uint32_t(swapImages_.size()); g_handles.minImageCount = std::max<uint32_t>(uint32_t(swapImages_.size()), 2);
}
// ---- targets ----
// Hor+ ultrawide widening: at app::HostAspect() (wider than 16:9, ultrawide on)
// the frame target is built wider than the guest's buffer by that aspect at the
// same height, so the guest's projection (overridden to the same aspect in
// src/game/hooks.cpp) shows more world at the sides at an unchanged vertical
// field of view. The gained columns sit half on each side (Target::dx); the
// XDK layer centres both the 3-D fold and the 2-D screen matrix by the same dx
// (sdk/runtime/xdk/d3d8.cpp FrameCentreShift), and BlitFrame presents the
// widened target whole. Off (aspect 0) every number here is the console's.
uint32_t VulkanBackend::WidenedGuestWidth() const {
    float aspect = app::HostAspect();
    if (aspect <= 0.0f) return guestW_;
    return std::max(guestW_, uint32_t(std::lround(float(guestH_) * aspect)));
}
GuestExtent VulkanBackend::BoundExtent(const Surface& s) const {
    GuestExtent e{}; SurfaceSize(s, e.w, e.h);
    const Target* t = boundRt_ ? boundRt_ : boundDs_;
    if (t && t->dx != 0 && t->gw == e.w && t->gh == e.h) { e.dx = t->dx; e.w = t->gw + uint32_t(std::lround(t->dx * 2)); }
    return e;
}
uint64_t VulkanBackend::SizedTargetKey(uint32_t data, bool depth, uint32_t gw, uint32_t gh) const {
    uint64_t key = TargetMapKey(data, depth);
    if (!(depth && (gw != guestW_ || gh != guestH_))) return key;
    return key | uint64_t(gw) << 34 | uint64_t(gh) << 48;
}
VulkanBackend::TargetSlot VulkanBackend::FindTargetSlot(const Surface& s, bool depth, uint64_t alt) {
    TargetSlot slot{}; slot.data = mem::VirtualFromPhysical(s.data); SurfaceSize(s, slot.gw, slot.gh);
    slot.key = SizedTargetKey(slot.data, depth, slot.gw, slot.gh) | alt;
    slot.it = targets_.find(slot.key);
    slot.sized = slot.it != targets_.end() && slot.it->second.gw == slot.gw && slot.it->second.gh == slot.gh;
    return slot;
}
Target* VulkanBackend::FindTarget(const Surface& s, bool depth) {
    if (!s.header) return nullptr;
    TargetSlot slot = FindTargetSlot(s, depth, 0);
    if (!slot.sized) return nullptr;
    targetUsed_[slot.data] = frame_; return &slot.it->second;
}
void VulkanBackend::ReleaseTarget(std::unordered_map<uint64_t, Target>::iterator it) {
    if (boundRt_ == &it->second || boundDs_ == &it->second) boundRt_ = boundDs_ = nullptr;
    DeferDestroyImage(it->second.res); DeferDestroyImage(it->second.fb); targets_.erase(it);
}
Target* VulkanBackend::EnsureTarget(const Surface& s, bool depth) {
    if (!s.header) return nullptr;
    TargetSlot slot = FindTargetSlot(s, depth, 0);
    targetUsed_[slot.data] = frame_;
    if (slot.sized) return &slot.it->second;
    if (slot.it != targets_.end()) ReleaseTarget(slot.it);
    Target t; t.depth = depth; t.gw = slot.gw; t.gh = slot.gh;
    uint32_t gw = IsFrameTarget(t) ? WidenedGuestWidth() : t.gw;
    t.dx = (gw - t.gw) / 2.0f;
    t.w = std::max(1u, uint32_t(std::lround(gw * scale_))); t.h = std::max(1u, uint32_t(std::lround(t.gh * scale_)));
    if (depth) t.res = CreateImage(t.w, t.h, depthFormat_, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT));
    else t.res = CreateImage(t.w, t.h, kBgra, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    if (!t.res.Valid()) return nullptr;
    t.data = slot.data; t.bytes = depth ? slot.gw * slot.gh * 4 : GuestPitch(slot.gw) * slot.gh;
    if (depth) {
        Transition(t.res, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkImageSubresourceRange range{VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT), 0, 1, 0, 1}; VkClearDepthStencilValue v{1.0f, 0};
        vkCmdClearDepthStencilImage(Cmd(), t.res.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &v, 1, &range);
        Transition(t.res, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    } else {
        Transition(t.res, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}; VkClearColorValue black{};
        vkCmdClearColorImage(Cmd(), t.res.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
        Transition(t.res, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    }
    t.epoch = DirtyEpoch(slot.data); t.sig = MemSig(slot.data, t.bytes);
    g_vkPerf.targets++;
    auto& ref = targets_[slot.key] = std::move(t); return &ref;
}
bool VulkanBackend::TargetStale(uint32_t data) const {
    auto it = targetUsed_.find(data); uint64_t used = it == targetUsed_.end() ? 0 : it->second;
    return frame_ - used > kStaleFrames;
}
bool VulkanBackend::BindTargets(const State& s) {
    Target* rt = EnsureTarget(s.rt, false);
    Target* ds = s.rs[kRsZEnable] || s.rs[kRsStencilEnable] ? EnsureTarget(s.zs, true) : nullptr;
    if (rt && ds && (rt->w != ds->w || rt->h != ds->h)) ds = nullptr;
    if (!rt && !ds) return false;
    if (rt) { if (LatchCpuWrite(*rt)) UploadTarget(*rt, s.rt); Transition(rt->res, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL); rt->drawn = true; }
    if (ds) { Transition(ds->res, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL); ds->drawn = true; }
    boundRt_ = rt; boundDs_ = ds; return true;
}
// ---- shaders / pipelines ----
VkShaderModule VulkanBackend::CompileModule(const std::string& src, const char* entry, bool fragment, uint64_t& hash) {
    uint64_t seed = Hash(entry, std::strlen(entry), fragment ? 0x7073ull : 0x7673ull);
    hash = Hash(src.data(), src.size(), seed);
    auto it = modules_.find(hash); if (it != modules_.end()) return it->second;
    if (failedModules_.count(hash)) return VK_NULL_HANDLE;  // D4: known HLSL compile failure this run
    std::vector<uint32_t> words; std::string error; VkShaderModule mod = VK_NULL_HANDLE;
    if (CompileHlslToSpirv(src, entry, fragment, words, &error) && !words.empty()) {
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = words.size() * 4; ci.pCode = words.data();
        // D4: a failed create (e.g. SPIR-V from a truncated cache file that passed the
        // old magic-only check) is logged with the shader hash, the on-disk cache entry
        // is discarded so the next run recompiles it, and the null module is never
        // cached: the next bind retries instead of skipping every draw forever.
        VkResult r = vkCreateShaderModule(device_, &ci, nullptr, &mod);
        if (r != VK_SUCCESS) {
            LOGE("vulkan: vkCreateShaderModule failed %d for %s shader %016llx; discarding its shader-cache entry (D4)", int(r), fragment ? "ps" : "vs", (unsigned long long)hash);
            DiscardCachedSpirv(src, entry, fragment);
            return VK_NULL_HANDLE;
        }
    } else {
        LOGE("vulkan: %s shader compile failed: %s", fragment ? "ps" : "vs", error.c_str()); LOGE("%s", src.c_str());
        failedModules_.insert(hash);
        return VK_NULL_HANDLE;
    }
    if (mod) g_vkPerf.shaders++;
    modules_[hash] = mod; return mod;
}
void VulkanBackend::EnsureVertexShader(const State& s, const Assembled& a, DrawShaders& sh) {
    auto it = shaderSrc_.find(sh.vkey);
    if (it != shaderSrc_.end()) {
        // D4: modules_ never holds null; a missing entry means a device-side create
        // failure, which falls through and retries instead of reading a cached null.
        auto mit = modules_.find(it->second);
        if (mit != modules_.end()) { sh.vs = mit->second; sh.vh = it->second; return; }
        if (failedModules_.count(it->second)) { sh.vs = VK_NULL_HANDLE; sh.vh = it->second; return; }
    }
    std::string src = sh.prog ? VertexProgramHlsl(s.vs->program, a.regs, a.decode) : FixedVertexHlsl(a.regs, a.pretransformed, a.decode, sh.fk);
    if (sh.quad) src = QuadHlsl(std::move(src));
    sh.vs = CompileModule(src, "main", false, sh.vh); shaderSrc_[sh.vkey] = sh.vh;
}
void VulkanBackend::EnsurePixelShader(const PixelKey& pk, DrawShaders& sh) {
    constexpr uint64_t kSalt = 0x9E3779B97F4A7C15ull;
    auto it = shaderSrc_.find(sh.pkey ^ kSalt);
    if (it != shaderSrc_.end()) {
        auto mit = modules_.find(it->second);
        if (mit != modules_.end()) { sh.ps = mit->second; sh.ph = it->second; return; }
        if (failedModules_.count(it->second)) { sh.ps = VK_NULL_HANDLE; sh.ph = it->second; return; }
    }
    sh.ps = CompileModule(PixelHlsl(pk), "main", true, sh.ph); shaderSrc_[sh.pkey ^ kSalt] = sh.ph;
}
VkSampler VulkanBackend::GetSampler(const VkSamplerCreateInfo& ci) {
    uint64_t key = Hash(&ci, sizeof(ci));
    auto it = samplers_.find(key); if (it != samplers_.end()) return it->second;
    VkSampler s = VK_NULL_HANDLE; vkCreateSampler(device_, &ci, nullptr, &s); samplers_[key] = s; return s;
}
VkPipeline VulkanBackend::EnsurePipeline(const PsoKey& k, VkShaderModule vs, VkShaderModule ps) {
    uint64_t h = Hash(&k, sizeof(k));
    auto it = pipelines_.find(h); if (it != pipelines_.end()) return it->second;
    VkVertexInputAttributeDescription attrs[16]; VkVertexInputBindingDescription binds[16];
    uint32_t na = 0, nb = 0; bool used[16] = {};
    for (uint32_t r = 0; r < 16; r++) {
        if (!((k.regs >> r) & 1)) continue;
        uint32_t e = k.elements[r];
        // SPIR-V location is NOT the guest register number: DXC/glslang number the
        // shader's ATTR<r> inputs sequentially (0,1,2,...) in declaration order
        // (VsInputs declares them in ascending register order, as this loop walks).
        // Using r here left every non-contiguous register set (e.g. regs 0,3,9)
        // reading undefined (0) attributes: positions survived (register 0 is
        // first) but texcoords were all zero, so textured draws sampled texel
        // (0,0) and the frame was the texture's corner colour, changing per frame.
        attrs[na].location = na; attrs[na].binding = (e >> 8) & 0xFF; attrs[na].format = VkFormatForDxgi(uint8_t(e & 0xFF)); attrs[na].offset = e >> 16; na++;
        used[attrs[na - 1].binding] = true;
    }
    // strides live in the draw's buffers, not the D3D12 pipeline key: read them from asm_ while
    // this draw is assembled (EnsurePipeline is called mid-Draw). Two draws sharing a key but
    // not strides would share the first pipeline - the noted approximation of this backend.
    for (uint32_t b = 0; b < asm_.buffer_count && nb < 16; b++) if (used[b]) { binds[nb].binding = b; binds[nb].stride = asm_.buffers[b].stride; binds[nb].inputRate = VK_VERTEX_INPUT_RATE_VERTEX; nb++; }
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = nb; vi.pVertexBindingDescriptions = binds; vi.vertexAttributeDescriptionCount = na; vi.pVertexAttributeDescriptions = attrs;
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = k.topo == 0 ? VK_PRIMITIVE_TOPOLOGY_POINT_LIST : k.topo == 1 ? (k.strip ? VK_PRIMITIVE_TOPOLOGY_LINE_STRIP : VK_PRIMITIVE_TOPOLOGY_LINE_LIST) : (k.strip ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO}; vp.viewportCount = 1; vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = (k.fill == 0x1B01 && fillNonSolid_) ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
    rs.cullMode = k.cull ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
    rs.frontFace = k.ccw ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE; rs.lineWidth = 1.0f;
    rs.depthClampEnable = VK_FALSE;
    if (k.zbias || k.zslope != 0) { rs.depthBiasEnable = VK_TRUE; rs.depthBiasConstantFactor = float(k.zbias) / 16777215.0f; rs.depthBiasSlopeFactor = k.zslope; }
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO}; ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    dss.depthTestEnable = k.zen; dss.depthWriteEnable = k.zwrite; dss.depthCompareOp = CmpOp(k.zfunc); dss.stencilTestEnable = k.sten;
    VkStencilOpState sop{}; sop.failOp = StencilOpVk(k.sfail); sop.depthFailOp = StencilOpVk(k.szfail); sop.passOp = StencilOpVk(k.spass);
    sop.compareOp = CmpOp(k.sfunc); sop.compareMask = k.smask; sop.writeMask = k.swmask; sop.reference = k.sref;
    dss.front = dss.back = sop;
    VkPipelineColorBlendAttachmentState cb{};
    cb.blendEnable = k.blend; cb.srcColorBlendFactor = BlendFactor(k.src, false); cb.dstColorBlendFactor = BlendFactor(k.dst, false);
    cb.colorBlendOp = BlendOpVk(k.op); cb.srcAlphaBlendFactor = BlendFactor(k.src, true); cb.dstAlphaBlendFactor = BlendFactor(k.dst, true);
    cb.alphaBlendOp = BlendOpVk(k.op); cb.colorWriteMask = VkColorComponentFlags(k.wmask & 0xF);
    VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cbs.attachmentCount = k.has_rt ? 1 : 0; cbs.pAttachments = &cb;
    cbs.blendConstants[0] = cbs.blendConstants[1] = cbs.blendConstants[2] = cbs.blendConstants[3] = 0;
    VkDynamicState dyn[3] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_BLEND_CONSTANTS};
    VkPipelineDynamicStateCreateInfo dyns{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO}; dyns.dynamicStateCount = 3; dyns.pDynamicStates = dyn;
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT; stages[0].module = vs; stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = ps; stages[1].pName = "main";
    VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    ci.stageCount = 2; ci.pStages = stages; ci.pVertexInputState = &vi; ci.pInputAssemblyState = &ia; ci.pViewportState = &vp;
    ci.pRasterizationState = &rs; ci.pMultisampleState = &ms; ci.pDepthStencilState = &dss; ci.pColorBlendState = &cbs;
    ci.pDynamicState = &dyns; ci.layout = pipelineLayout_;
    ci.renderPass = RenderPassFor(VkFormat(k.rtfmt), depthFormat_, k.has_rt != 0, k.has_ds != 0); ci.subpass = 0;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkResult pr = vkCreateGraphicsPipelines(device_, pipelineCache_, 1, &ci, nullptr, &pipe);
    if (pr != VK_SUCCESS) LOGE("vulkan: pipeline creation failed: VkResult %d vs %llx ps %llx", int(pr), (unsigned long long)k.vs, (unsigned long long)k.ps);
    if (pipe) g_vkPerf.psos++;
    pipelines_[h] = pipe; return pipe;
}
VkPipeline VulkanBackend::EnsureBlitPipeline(VkFormat rtv, bool over, bool depthWrite, bool swap) {
    uint64_t key = (uint64_t(rtv) << 3) | (over ? 1 : 0) | (depthWrite ? 2 : 0) | (swap ? 4 : 0);
    auto it = blitPipelines_.find(key); if (it != blitPipelines_.end()) return it->second;
    uint64_t h; VkShaderModule vs = CompileModule(depthWrite ? kDepthHlsl : kBlitHlsl, "vs", false, h);
    VkShaderModule ps = CompileModule(depthWrite ? kDepthHlsl : kBlitHlsl, "psmain", true, h);
    VkPipeline pipe = VK_NULL_HANDLE;
    if (vs && ps) {
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO}; ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO}; vp.viewportCount = 1; vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO}; ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        if (depthWrite) { dss.depthTestEnable = VK_TRUE; dss.depthWriteEnable = VK_TRUE; dss.depthCompareOp = VK_COMPARE_OP_ALWAYS; }
        VkPipelineColorBlendAttachmentState cb{}; cb.colorWriteMask = 0xF;
        if (over) { cb.blendEnable = VK_TRUE; cb.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; cb.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA; cb.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; cb.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE; }
        VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO}; cbs.attachmentCount = depthWrite ? 0 : 1; cbs.pAttachments = &cb;
        VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dyns{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO}; dyns.dynamicStateCount = 2; dyns.pDynamicStates = dyn;
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT; stages[0].module = vs; stages[0].pName = "vs";
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = ps; stages[1].pName = "psmain";
        VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        ci.stageCount = 2; ci.pStages = stages; ci.pVertexInputState = &vi; ci.pInputAssemblyState = &ia; ci.pViewportState = &vp;
        ci.pRasterizationState = &rs; ci.pMultisampleState = &ms; ci.pDepthStencilState = &dss; ci.pColorBlendState = &cbs;
        ci.pDynamicState = &dyns; ci.layout = pipelineLayout_;
        ci.renderPass = depthWrite ? RenderPassFor(VK_FORMAT_UNDEFINED, depthFormat_, false, true)
                                   : swap ? swapPass_ : RenderPassFor(rtv, VK_FORMAT_UNDEFINED, true, false);
        ci.subpass = 0;
        vkCreateGraphicsPipelines(device_, pipelineCache_, 1, &ci, nullptr, &pipe);
    }
    if (pipe) g_vkPerf.psos++;
    blitPipelines_[key] = pipe; return pipe;
}
// ---- textures ----
TexKind TextureKindOf(const TextureDesc& d) { if (d.cube) return kTexCube; if (d.depth > 1) return kTex3D; return kTex2D; }
bool SameTexture(const Tex& tx, const TextureDesc& d) {
    return tx.res.Valid() && tx.d.format == d.format && tx.d.width == d.width && tx.d.height == d.height && tx.d.depth == d.depth && tx.d.levels == d.levels && tx.d.cube == d.cube && tx.d.pitch == d.pitch;
}
bool VulkanBackend::CreateTexImage(Tex& tx, const TextureDesc& d, const FormatInfo& fi) {
    if (tx.res.Valid()) DeferDestroyImage(tx.res);
    tx = Tex{}; tx.d = d; tx.fmt = HostFormatVk(fi.host);
    uint32_t w = d.width, h = d.height; tx.levels = d.levels;
    if (fi.compressed && ((w & 3) || (h & 3))) { w = (w + 3) & ~3u; h = (h + 3) & ~3u; tx.levels = 1; }
    VkImageType type = d.depth > 1 ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
    uint32_t layers = d.cube ? 6 : (type == VK_IMAGE_TYPE_3D ? d.depth : 1);
    tx.res = CreateImage(w, h, tx.fmt, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT, layers, type, tx.levels);
    if (tx.res.Valid()) g_vkPerf.newtex++;
    return tx.res.Valid();
}
void VulkanBackend::UploadTexImage(Tex& tx, const TextureDesc& d, const FormatInfo& fi, const uint8_t* p, const uint32_t* colours) {
    Transition(tx.res, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    uint32_t faces = d.cube ? 6 : 1; std::vector<uint8_t> px;
    for (uint32_t f = 0; f < faces; f++) for (uint32_t l = 0; l < tx.levels; l++) {
        uint32_t pitch = 0; DecodeLevel(d, l, f, p, colours, px, pitch);
        if (px.empty()) continue;
        g_vkPerf.texups++; g_vkPerf.texupBytes += px.size();
        uint32_t lw = std::max(1u, d.width >> l), lh = std::max(1u, d.height >> l);
        if (fi.compressed) { lw = (lw + 3) & ~3u; lh = (lh + 3) & ~3u; }
        Alloc a = UploadAlloc(px.size(), 16); if (!a.cpu) return;
        std::memcpy(a.cpu, px.data(), px.size());
        VkBufferImageCopy region{}; region.bufferOffset = a.off;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; region.imageSubresource.mipLevel = l;
        region.imageSubresource.baseArrayLayer = d.depth > 1 ? 0 : f; region.imageSubresource.layerCount = 1;
        region.imageExtent = {lw, lh, d.depth > 1 ? std::max(1u, d.depth >> l) : 1};
        vkCmdCopyBufferToImage(Cmd(), a.buffer, tx.res.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    }
    Transition(tx.res, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}
struct StagePalette { uint32_t colours[256] = {}; uint32_t va = 0; uint64_t hash = 0; };
void ReadStagePalette(const State& s, int stage, const TextureDesc& d, StagePalette& palette) {
    if (d.format != 0x0B || !s.palettes[size_t(stage)]) return;
    uint32_t header = s.palettes[size_t(stage)];
    palette.va = mem::VirtualFromPhysical(kernel::R32(header + 4));
    uint32_t sizeCode = kernel::R32(header) >> 30;
    size_t bytes = (std::size(palette.colours) >> sizeCode) * sizeof(uint32_t);
    std::memcpy(palette.colours, kernel::HostPtr(palette.va), bytes);
    palette.hash = Hash(palette.colours, sizeof(palette.colours));
}
VkImageView VulkanBackend::BindGuestTexture(const State& s, int stage, const TextureDesc& d, const FormatInfo& fi, DrawTextures& stages) {
    TexInfo& info = stages.info[stage];
    StagePalette palette; ReadStagePalette(s, stage, d, palette);
    const uint64_t key = uint64_t(palette.va) << 32 | d.data;
    Tex& tx = textures_[key];
    uint32_t total = TotalSize(d); bool same = SameTexture(tx, d);
    if (!same && !CreateTexImage(tx, d, fi)) { textures_.erase(key); return VK_NULL_HANDLE; }
    if (tx.frame != presents_ + 1 || !same || tx.palette != palette.hash) {
        tx.frame = presents_ + 1; tx.palette = palette.hash;
        const uint8_t* p = static_cast<const uint8_t*>(kernel::HostPtr(d.data));
        uint64_t sig = MemSig(d.data, total) ^ DirtyEpoch(d.data) ^ palette.hash;
        if (sig != tx.sig || !same) { tx.sig = sig; UploadTexImage(tx, d, fi, p, palette.va ? palette.colours : nullptr); }
    }
    if (fi.depth) info.shadow = ShadowDepthKind(d.format);
    return tx.res.SampleView();
}
ImageRes VulkanBackend::Feedback(Target& t) {
    if (!t.fb.Valid()) {
        t.fb = CreateImage(t.res.w, t.res.h, t.res.format, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                           t.depth ? VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT) : VkImageAspectFlags(VK_IMAGE_ASPECT_COLOR_BIT), 1);
        if (!t.fb.Valid()) return t.fb;
    }
    VkImageLayout was = t.res.layout;
    Transition(t.res, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL); Transition(t.fb, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkImageCopy region{};
    region.srcSubresource.aspectMask = region.dstSubresource.aspectMask = t.depth ? VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT) : VkImageAspectFlags(VK_IMAGE_ASPECT_COLOR_BIT);
    region.srcSubresource.layerCount = region.dstSubresource.layerCount = 1; region.extent = {t.res.w, t.res.h, 1};
    vkCmdCopyImage(Cmd(), t.res.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, t.fb.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    Transition(t.fb, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL); Transition(t.res, was);
    return t.fb;
}
VkImageView VulkanBackend::BindTargetTexture(const State& s, int stage, const TextureDesc& d, Target& t, bool box, DrawTextures& stages) {
    TexInfo& info = stages.info[stage];
    if (&t == drawDs_) { LOG_FIRST(8, "vulkan: stage %d samples the depth buffer it draws with, reads nothing", stage); return VK_NULL_HANDLE; }
    bool feedback = &t == drawRt_; stages.rt[stage] = true;
    if (!t.depth && LatchCpuWrite(t)) { Surface su; su.header = s.textures[size_t(stage)]; UploadTarget(t, su); }
    info.kind = kTex2D;
    Target* rt = FindTarget(s.rt, false);
    float k = float(t.w) / (t.gw * stages.k[stage]);
    if (!t.depth && box && k > 1.01f && rt && rt != &t && (t.gw > rt->gw || t.gh > rt->gh)) { stages.wide[stage] = k; info.flags |= kTexWide; }
    if (feedback) { ImageRes fb = Feedback(t); return fb.Valid() ? fb.SampleView() : VK_NULL_HANDLE; }
    Transition(t.res, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    if (t.depth) info.shadow = ShadowDepthKind(d.format);
    info.kind = kTex2D; return t.res.SampleView();
}
VkImageView VulkanBackend::BindTexture(const State& s, int stage, DrawTextures& stages) {
    TexInfo& info = stages.info[stage]; info = {};
    stages.rt[stage] = false; stages.k[stage] = 1; stages.wide[stage] = 0; stages.hasDesc[stage] = false;
    TextureDesc d; uint32_t header = s.textures[size_t(stage)];
    if (!header) return VK_NULL_HANDLE;
    if (!Describe(header, d)) { LOG_FIRST(8, "vulkan: stage %d texture %08X is no texture, reads nothing", stage, header); return VK_NULL_HANDLE; }
    stages.desc[stage] = d; stages.hasDesc[stage] = true;
    const FormatInfo& fi = Format(d.format);
    info.kind = TextureKindOf(d); info.rect = d.linear;
    auto tit = targets_.find(SizedTargetKey(d.data, fi.depth, d.width, d.height));
    if (tit != targets_.end() && !d.cube && d.depth == 1 && !fi.compressed) return BindTargetTexture(s, stage, d, tit->second, true, stages);
    return BindGuestTexture(s, stage, d, fi, stages);
}
bool VulkanBackend::BindStageTextures(const State& s, DrawTextures& stages) {
    drawRt_ = boundRt_; drawDs_ = boundDs_;
    for (int i = 0; i < 4; i++) stages.view[i] = BindTexture(s, i, stages);
    return BindTargets(s);
}
VkSampler VulkanBackend::StageSamplerFor(const State& s, int i, const DrawTextures& stages, bool clampCoords) {
    const auto& t = s.tss[size_t(i)];
    auto lin = [](uint32_t f) { return f >= 2; };
    VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    ci.magFilter = lin(t[kTssMagFilter]) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    ci.minFilter = lin(t[kTssMinFilter]) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    ci.mipmapMode = t[kTssMipFilter] == 2 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    ci.addressModeU = AddressMode(clampCoords && t[kTssAddressU] == 1 ? 3 : t[kTssAddressU]);
    ci.addressModeV = AddressMode(clampCoords && t[kTssAddressV] == 1 ? 3 : t[kTssAddressV]);
    ci.addressModeW = AddressMode(t[kTssAddressW]);
    ci.mipLodBias = BitsF(t[kTssMipLodBias]); ci.minLod = float(t[kTssMaxMipLevel]);
    ci.maxLod = t[kTssMipFilter] ? VK_LOD_CLAMP_NONE : ci.minLod;
    float bc[4]; ColorF(t[kTssBorderColor], bc);
    ci.borderColor = (bc[0] || bc[1] || bc[2] || bc[3]) ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    bool anisoWanted = t[kTssMinFilter] == 3 || t[kTssMagFilter] == 3;
    uint32_t an = anisoWanted ? std::max<uint32_t>(t[kTssMaxAnisotropy], 1) : 1;
    if (!anisoWanted && app::Get().aniso && lin(t[kTssMinFilter]) && t[kTssMipFilter] && !stages.info[i].shadow) an = app::Get().aniso;
    if (an > 1 && anisotropy_) { ci.anisotropyEnable = VK_TRUE; ci.maxAnisotropy = std::min<uint32_t>(an, 16); }
    if (stages.info[i].shadow) {
        static const uint8_t swap[8] = {0, 4, 2, 6, 1, 5, 3, 7};
        ci.compareEnable = VK_TRUE; ci.compareOp = VkCompareOp(swap[CmpIndex(s.rs[kRsShadowFunc])]);
        ci.magFilter = ci.minFilter = (lin(t[kTssMinFilter]) || lin(t[kTssMagFilter])) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        ci.anisotropyEnable = VK_FALSE;
    }
    return GetSampler(ci);
}
// ---- constants ----
void FillFog(const State& s, bool prog, DrawConstants& c) {
    if (!s.rs[kRsFogEnable]) return;
    if (prog && s.rs[kRsFogTableMode] == 0) { c.fog[0] = 1; c.fog[1] = 0; c.fog[2] = 0; c.fog[3] = 3.0f; return; }
    c.fog[0] = BitsF(s.rs[kRsFogStart]); c.fog[1] = BitsF(s.rs[kRsFogEnd]); c.fog[2] = BitsF(s.rs[kRsFogDensity]); c.fog[3] = float(s.rs[kRsFogTableMode]);
}
void MatMulC(const float* x, const float* y, float* o) {
    for (int r = 0; r < 4; r++) for (int col = 0; col < 4; col++) o[r * 4 + col] = x[r * 4] * y[col] + x[r * 4 + 1] * y[4 + col] + x[r * 4 + 2] * y[8 + col] + x[r * 4 + 3] * y[12 + col];
}
void TransposeC(const float* m, float* o) { for (int r = 0; r < 4; r++) for (int col = 0; col < 4; col++) o[col * 4 + r] = m[r * 4 + col]; }
void VulkanBackend::FillConstants(const State& s, const DrawShaders& sh, const DrawTextures& stages, GuestExtent size, DrawConstants& c) {
    // a vertex program's screen space positions are the render target's pixels, SuperSampleScale of them to each of
    // the frame's on a supersampled back buffer; fixed function and pretransformed positions stay in the frame's
    c.surf[0] = float(size.w) * (sh.prog ? s.super_x : 1.0f); c.surf[1] = float(size.h) * (sh.prog ? s.super_y : 1.0f); c.surf[2] = 1.0f / s.zscale;
    ColorF(s.rs[kRsFogColor], c.fog_color);
    c.misc[0] = float(SimpleState(s, kMAlphaRef) & 0xFF);
    c.misc[1] = SimpleState(s, kMAlphaTestEnable) ? float(CmpIndex(SimpleState(s, kMAlphaFunc))) : 7.0f;
    c.misc[2] = float(CmpIndex(s.rs[kRsShadowFunc])); c.misc[3] = s.rs[kRsSpecularEnable] ? 1.0f : 0.0f;
    FillFog(s, sh.prog, c);
    for (int i = 0; i < 8; i++) { ColorF(SimpleState(s, kMPsConstant0 + i * 4), c.pc[i * 2]); ColorF(SimpleState(s, kMPsConstant1 + i * 4), c.pc[i * 2 + 1]); }
    ColorF(SimpleState(s, kMPsFinalConstant0), c.pc[16]); ColorF(SimpleState(s, kMPsFinalConstant1), c.pc[17]);
    ColorF(s.rs[kRsTextureFactor], c.tfactor);
    for (int i = 0; i < 4; i++) {
        if (stages.hasDesc[i]) {
            const TextureDesc& d = stages.desc[i];
            c.texsize[i][0] = float(d.width); c.texsize[i][1] = float(d.height);
            if (stages.rt[i]) {
                float k = stages.k[i];
                c.texsize[i][2] = -0.5f / (float(d.width) * k); c.texsize[i][3] = -0.5f / (float(d.height) * k);
                if (stages.wide[i] > 0) { float n = std::ceil(3.0f * stages.wide[i] * 0.5f); c.spread[i][0] = 3.0f / (n * float(d.width) * k); c.spread[i][1] = 3.0f / (n * float(d.height) * k); c.spread[i][2] = n; }
            }
        }
        const auto& t = s.tss[size_t(i)];
        c.bump[i][0] = BitsF(t[kTssBumpEnvMat00]); c.bump[i][1] = BitsF(t[kTssBumpEnvMat01]); c.bump[i][2] = BitsF(t[kTssBumpEnvMat10]); c.bump[i][3] = BitsF(t[kTssBumpEnvMat11]);
        c.bumpl[i][0] = BitsF(t[kTssBumpEnvLScale]); c.bumpl[i][1] = BitsF(t[kTssBumpEnvLOffset]);
    }
    if (sh.fixed_transform) {
        constexpr size_t kTsView = 2, kTsProj = 3, kTsTex0 = 16, kTsWorld = 256;
        float wv[16], wvp[16], full[16];
        MatMulC(s.transforms[kTsWorld].data(), s.transforms[kTsView].data(), wv);
        MatMulC(wv, s.transforms[kTsProj].data(), wvp);
        const auto& v = s.viewport;
        float sx = float(v.w + size.dx * 2) / size.w, sy = float(v.h) / size.h;
        float ox = (2.0f * (v.x + size.dx) + v.w) / size.w - 1, oy = 1 - (2.0f * v.y + v.h) / size.h;
        float vpm[16] = {sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, v.max_z - v.min_z, 0, ox, oy, v.min_z, 1};
        MatMulC(wvp, vpm, full); TransposeC(full, c.wvp); TransposeC(wv, c.wv);
        float det = wv[0] * (wv[5] * wv[10] - wv[6] * wv[9]) - wv[1] * (wv[4] * wv[10] - wv[6] * wv[8]) + wv[2] * (wv[4] * wv[9] - wv[5] * wv[8]);
        float id = det != 0 ? 1.0f / det : 0; float nrm[16] = {};
        for (int r = 0; r < 3; r++) for (int col = 0; col < 3; col++) { int r1 = (r + 1) % 3, r2 = (r + 2) % 3, c1 = (col + 1) % 3, c2 = (col + 2) % 3; nrm[r * 4 + col] = (wv[r1 * 4 + c1] * wv[r2 * 4 + c2] - wv[r1 * 4 + c2] * wv[r2 * 4 + c1]) * id; }
        TransposeC(nrm, c.wvn);
        for (int t = 0; t < 4; t++) TransposeC(s.transforms[kTsTex0 + size_t(t)].data(), c.tm[t]);
        std::memcpy(c.mat, s.material.data(), sizeof(float) * 16); c.mat[4][0] = s.material[16];
        ColorF(s.rs[kRsAmbient], c.amb);
        const float* vm = s.transforms[kTsView].data();
        for (int i = 0; i < 8; i++) {
            if (!((sh.fk.types >> (i * 2)) & 3)) continue;
            float f[26]; std::memcpy(f, s.lights[size_t(i)].data(), sizeof(f)); auto& o = c.lt[i];
            for (int j = 0; j < 4; j++) { o[0][j] = f[1 + j]; o[1][j] = f[5 + j]; o[2][j] = f[9 + j]; }
            for (int j = 0; j < 3; j++) { o[3][j] = f[13] * vm[j] + f[14] * vm[4 + j] + f[15] * vm[8 + j] + vm[12 + j]; o[4][j] = f[16] * vm[j] + f[17] * vm[4 + j] + f[18] * vm[8 + j]; }
            o[3][3] = f[19]; o[4][3] = f[20]; o[5][0] = f[21]; o[5][1] = f[22]; o[5][2] = f[23];
            o[6][0] = std::cos(f[24] * 0.5f); o[6][1] = std::cos(f[25] * 0.5f);
        }
    }
}
bool TexcoordRectOf(const Assembled& a, int reg, float lo[2], float hi[2]) {
    if (!a.vertices.empty() || !a.vertex_count || a.vertex_count > 4096 || !(a.regs >> reg & 1)) return false;
    const VertexElement& e = a.elements[size_t(reg)];
    if (e.format != kDxgiRg32f && e.format != kDxgiRgba32f) return false;
    const VertexBuffer& vb = a.buffers[e.buffer];
    if (!vb.src || !vb.stride || uint64_t(a.vertex_count) * vb.stride > vb.bytes) return false;
    const uint32_t inBuffer = vb.bytes / vb.stride;
    lo[0] = lo[1] = 3.4028235e38f; hi[0] = hi[1] = -3.4028235e38f;
    auto read = [&](uint32_t v) { float uv[2]; std::memcpy(uv, vb.src + size_t(v) * vb.stride + e.offset, sizeof(uv));
        if (!std::isfinite(uv[0]) || !std::isfinite(uv[1])) return false;
        lo[0] = std::min(lo[0], uv[0]); lo[1] = std::min(lo[1], uv[1]); hi[0] = std::max(hi[0], uv[0]); hi[1] = std::max(hi[1], uv[1]); return true; };
    if (!a.indices.empty()) { if (a.indices.size() > 12288) return false; for (uint32_t idx : a.indices) if (idx >= inBuffer || !read(idx)) return false; }
    else for (uint32_t v = 0; v < a.vertex_count; v++) if (!read(v)) return false;
    return hi[0] > lo[0] && hi[1] > lo[1];
}
void FillUvRect(const State& s, uint32_t prim, uint32_t count, const Assembled& a, const DrawShaders& sh, const DrawTextures& stages, DrawConstants& c) {
    static const bool kOff = std::getenv("XBE_NO_UVRECT") != nullptr;
    if (kOff || prim != 8 || count != 4 || a.vertex_count != 4) return;
    int candidate = -1;
    if (!sh.prog) candidate = -2;
    else { for (int r = 1; r < 16; r++) { if (!(a.regs >> r & 1)) continue; const VertexElement& e = a.elements[size_t(r)]; if (e.format != kDxgiRg32f && e.format != kDxgiRgba32f) continue; if (candidate >= 0) return; candidate = r; } if (candidate < 0) return; }
    for (int i = 0; i < 4; i++) {
        if (!stages.view[i] || stages.rt[i] || stages.info[i].kind != kTex2D || stages.info[i].shadow || (stages.info[i].flags & kTexWide) || !s.textures[size_t(i)] || !stages.hasDesc[i]) continue;
        const TextureDesc& d = stages.desc[i];
        if (!d.width || !d.height || d.levels != 1) continue;
        if (!sh.prog && ((sh.fk.tex[i] >> 4) & 7)) continue;
        const int reg = candidate >= 0 ? candidate : 9 + int(sh.fk.tex[i] & 3);
        float lo[2], hi[2]; if (!TexcoordRectOf(a, reg, lo, hi)) continue;
        const float w = float(d.width), hgt = float(d.height); float l, t, r, b;
        if (stages.info[i].rect) { if (lo[0] < -0.5f || lo[1] < -0.5f || hi[0] > w + 0.5f || hi[1] > hgt + 0.5f) continue; l = (lo[0] + 0.5f) / w; r = (hi[0] - 0.5f) / w; t = (lo[1] + 0.5f) / hgt; b = (hi[1] - 0.5f) / hgt; }
        else { if (lo[0] < -1e-4f || lo[1] < -1e-4f || hi[0] > 1.0001f || hi[1] > 1.0001f) continue; l = lo[0] + 0.5f / w; r = hi[0] - 0.5f / w; t = lo[1] + 0.5f / hgt; b = hi[1] - 0.5f / hgt; }
        if (r <= l) { l = r = (l + r) * 0.5f; r += 1e-7f; } if (b <= t) { t = b = (t + b) * 0.5f; b += 1e-7f; }
        c.uvrect[i][0] = l; c.uvrect[i][1] = t; c.uvrect[i][2] = r; c.uvrect[i][3] = b;
    }
}
bool UnitTexcoordsOf(const Assembled& a) {
    if (!a.vertices.empty() || !a.vertex_count || a.vertex_count > 4096) return false;
    bool any = false;
    for (int r = 0; r < 16; r++) {
        const VertexElement& e = a.elements[size_t(r)];
        if (!(a.regs >> r & 1) || e.format != kDxgiRg32f) continue;
        const VertexBuffer& vb = a.buffers[e.buffer];
        if (!vb.src || !vb.stride || uint64_t(a.vertex_count) * vb.stride > vb.bytes) return false;
        for (uint32_t v = 0; v < a.vertex_count; v++) { float uv[2]; std::memcpy(uv, vb.src + size_t(v) * vb.stride + e.offset, sizeof(uv)); if (!(uv[0] >= -1e-4f && uv[0] <= 1.0001f && uv[1] >= -1e-4f && uv[1] <= 1.0001f)) return false; }
        any = true;
    }
    return any;
}
VkDescriptorSet VulkanBackend::AllocDrawSet(VkBuffer vcBuf, uint64_t vcOff, VkBuffer dcBuf, uint64_t dcOff, const DrawTextures& stages, const std::array<VkSampler, 4>& samplers) {
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    EnsureRecording(); CheckAllocGeneration("descriptor set");  // as UploadAlloc: the set must come from the open segment's pool
    ai.descriptorPool = Cur().pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &setLayout_;
    VkResult ar = vkAllocateDescriptorSets(device_, &ai, &set);
    if (ar != VK_SUCCESS) { LOG_FIRST(16, "vulkan: vkAllocateDescriptorSets failed: VkResult %d (%u sets in this segment)", int(ar), Cur().sets); return VK_NULL_HANDLE; }
    Cur().sets++;
    VkDescriptorBufferInfo bi[2] = {{vcBuf, vcOff, sizeof(float) * 4 * 192}, {dcBuf, dcOff, sizeof(DrawConstants)}};
    VkDescriptorImageInfo ii[8]{}; VkWriteDescriptorSet w[10]{};
    for (int i = 0; i < 10; i++) { w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = set; w[i].dstBinding = uint32_t(i); w[i].descriptorCount = 1; }
    w[0].descriptorType = w[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[0].pBufferInfo = &bi[0]; w[1].pBufferInfo = &bi[1];
    for (int i = 0; i < 4; i++) {
        ii[i].sampler = samplers[i] ? samplers[i] : nullSampler_; ii[i].imageView = stages.view[i] ? stages.view[i] : nullImage_.SampleView(); ii[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        w[2 + i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[2 + i].pImageInfo = &ii[i];
        ii[4 + i].sampler = ii[i].sampler; w[6 + i].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; w[6 + i].pImageInfo = &ii[4 + i];
    }
    vkUpdateDescriptorSets(device_, 10, w, 0, nullptr); return set;
}
void VulkanBackend::BeginTargetPass(Target* rt, Target* ds, VkFramebuffer& fbOut) {
    if (passOpen_ && passRt_ == rt && passDs_ == ds) { fbOut = passFb_; return; }  // already drawing there
    FlushPass();
    VkRenderPass rp = RenderPassFor(rt ? kBgra : VK_FORMAT_UNDEFINED, depthFormat_, rt != nullptr, ds != nullptr);
    if (rt) Transition(rt->res, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    if (ds) Transition(ds->res, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    VkImageView views[2]; uint32_t n = 0;
    if (rt) views[n++] = rt->res.view;
    if (ds) views[n++] = ds->res.view;
    uint32_t w = rt ? rt->res.w : ds->res.w, h = rt ? rt->res.h : ds->res.h;
    VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fi.renderPass = rp; fi.attachmentCount = n; fi.pAttachments = views; fi.width = w; fi.height = h; fi.layers = 1;
    fbOut = VK_NULL_HANDLE; vkCreateFramebuffer(device_, &fi, nullptr, &fbOut);
    Cur().garbage.push_back([this, fb = fbOut] { if (fb) vkDestroyFramebuffer(device_, fb, nullptr); });
    VkRenderPassBeginInfo bi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    bi.renderPass = rp; bi.framebuffer = fbOut; bi.renderArea = {{0, 0}, {w, h}};
    vkCmdBeginRenderPass(Cmd(), &bi, VK_SUBPASS_CONTENTS_INLINE);
    passOpen_ = true; passRt_ = rt; passDs_ = ds; passFb_ = fbOut;
}
void VulkanBackend::FlushPass() {
    if (!passOpen_) return;
    passOpen_ = false;
    vkCmdEndRenderPass(Cmd());
}
void VulkanBackend::SetViewportScissor(const State& s, GuestExtent size, uint32_t targetW, uint32_t targetH) {
    float W = float(targetW), H = float(targetH);
    VkViewport vp{0, H, W, -H, 0, 1};  // negative height: shaders emit D3D clip space
    if (!boundVpValid_ || std::memcmp(&boundViewport_, &vp, sizeof(vp)) != 0) { vkCmdSetViewport(Cmd(), 0, 1, &vp); boundViewport_ = vp; boundVpValid_ = true; }
    float sc = W / size.w;
    int32_t x0 = int32_t((s.viewport.x + size.dx) * sc), y0 = int32_t(s.viewport.y * sc);
    int32_t x1 = int32_t((s.viewport.x + s.viewport.w + size.dx) * sc), y1 = int32_t((s.viewport.y + s.viewport.h) * sc);
    if (s.scissor_count) { x0 = std::max(x0, int32_t((s.scissors[0] + size.dx) * sc)); y0 = std::max(y0, int32_t(s.scissors[1] * sc)); x1 = std::min(x1, int32_t((s.scissors[2] + size.dx) * sc)); y1 = std::min(y1, int32_t(s.scissors[3] * sc)); }
    if (size.dx != 0) { if (x0 <= int32_t(size.dx * sc)) x0 = 0; if (x1 >= int32_t((float(size.w) - size.dx) * sc)) x1 = int32_t(W); }
    int32_t cx0 = std::clamp<int32_t>(x0, 0, int32_t(W)), cy0 = std::clamp<int32_t>(y0, 0, int32_t(H));
    int32_t cx1 = std::clamp<int32_t>(x1, 0, int32_t(W)), cy1 = std::clamp<int32_t>(y1, 0, int32_t(H));
    VkRect2D rect{{cx0, cy0}, {uint32_t(std::max(0, cx1 - cx0)), uint32_t(std::max(0, cy1 - cy0))}};
    if (!boundScValid_ || std::memcmp(&boundScissor_, &rect, sizeof(rect)) != 0) { vkCmdSetScissor(Cmd(), 0, 1, &rect); boundScissor_ = rect; boundScValid_ = true; }
}
void VulkanBackend::DrawGeometry(const Assembled& a) {
    // Runs of draws share one buffer: when every binding is the same bytes from the same source as the
    // last draw's, its upload is still live in this segment and still bound, so neither is repeated.
    bool sameVb = vbValid_ && vbCount_ == a.buffer_count;
    for (uint32_t i = 0; sameVb && i < a.buffer_count; i++)
        sameVb = vbLast_[i].src == a.buffers[i].src && vbLast_[i].bytes == a.buffers[i].bytes &&
                 std::memcmp(vbLast_[i].alloc.cpu, a.buffers[i].src, a.buffers[i].bytes) == 0;
    if (!sameVb) {
        VkBuffer bufs[16]; VkDeviceSize offs[16]{};
        for (uint32_t i = 0; i < a.buffer_count; i++) {
            Alloc vb = UploadAlloc(a.buffers[i].bytes, 16); if (!vb.cpu) { vbValid_ = false; return; }
            std::memcpy(vb.cpu, a.buffers[i].src, a.buffers[i].bytes); bufs[i] = vb.buffer; offs[i] = vb.off;
            vbLast_[i] = {a.buffers[i].src, a.buffers[i].bytes, vb};
        }
        if (a.buffer_count) vkCmdBindVertexBuffers(Cmd(), 0, a.buffer_count, bufs, offs);
        vbCount_ = a.buffer_count; vbValid_ = true;
    }
    if (a.index_count) {
        uint32_t ibytes = a.index_count * (a.index16 ? 2 : 4);
        Alloc ib = UploadAlloc(ibytes, 4); if (!ib.cpu) return;
        std::memcpy(ib.cpu, a.index_src, ibytes);
        vkCmdBindIndexBuffer(Cmd(), ib.buffer, ib.off, a.index16 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(Cmd(), a.index_count, 1, 0, a.base_vertex, 0);
    } else vkCmdDraw(Cmd(), a.draw_count, 1, 0, 0);
}
DrawShaders ClassifyShaders(const State& s, const Assembled& a, uint32_t count) {
    DrawShaders sh; sh.prog = (s.vs_handle & 1) && s.vs && !s.vs->program.empty();
    sh.quad = count <= 6 && !s.rs[kRsZEnable]; sh.fixed_transform = !sh.prog && !a.pretransformed;
    if (!sh.prog) sh.fk = MakeFixedKey(s); return sh;
}
uint64_t VertexKeyOf(const State& s, const Assembled& a, const DrawShaders& sh) {
    uint64_t seed = sh.prog ? s.vs->hash ^ a.regs : Hash(&s.vs_handle, 4, a.regs | 0x10000ull);
    uint64_t vkey = Hash(&a.decode, 8, seed);
    if (!sh.prog && a.pretransformed) vkey ^= 0x5A5Aull;
    if (sh.quad) vkey ^= 0xC3C30000ull;
    if (!sh.prog) vkey = Hash(&sh.fk, sizeof(sh.fk), vkey);
    return vkey;
}
void BuildPsoKey(const State& s, const Assembled& a, uint64_t vh, uint64_t ph, Target* rt, Target* ds, PsoKey& k) {
    std::memset(&k, 0, sizeof(k)); k.vs = vh; k.ps = ph; k.regs = a.regs;
    for (int r = 0; r < 16; r++) if ((a.regs >> r) & 1) { const VertexElement& e = a.elements[size_t(r)]; k.elements[r] = e.format | uint32_t(e.buffer) << 8 | uint32_t(e.offset) << 16; }
    k.topo = uint32_t(a.topology); k.strip = a.strip; k.has_rt = rt != nullptr; k.has_ds = ds != nullptr; k.rtfmt = rt ? uint32_t(kBgra) : 0;
    k.sref = SimpleState(s, kMStencilRef) & 0xFF;
    k.blend = SimpleState(s, kMBlendEnable) != 0;
    if (k.blend) { k.src = SimpleState(s, kMSrcBlend); k.dst = SimpleState(s, kMDestBlend); k.op = SimpleState(s, kMBlendOp); }
    uint32_t cw = SimpleState(s, kMColorWriteEnable);
    k.wmask = ((cw >> 16) & 1) | ((cw >> 8) & 1) << 1 | (cw & 1) << 2 | ((cw >> 24) & 1) << 3;
    if (k.has_ds) {
        k.zen = s.rs[kRsZEnable] != 0;
        if (k.zen) { k.zwrite = SimpleState(s, kMZWriteEnable) != 0; k.zfunc = SimpleState(s, kMZFunc); }
        k.sten = s.rs[kRsStencilEnable] != 0;
        if (k.sten) { k.sfunc = SimpleState(s, kMStencilFunc); k.smask = SimpleState(s, kMStencilMask) & 0xFF; k.swmask = SimpleState(s, kMStencilWriteMask) & 0xFF; k.sfail = s.rs[kRsStencilFail]; k.szfail = SimpleState(s, kMStencilZFail); k.spass = SimpleState(s, kMStencilPass); }
        if (a.topology == Topology::Triangles && SimpleState(s, kMSolidOffsetEnable)) {
            k.zbias = int32_t(std::lround(BitsF(SimpleState(s, kMDepthOffset)) * 16777215.0f / s.zscale));
            k.zslope = BitsF(SimpleState(s, kMSlopeScale)) * (ds ? float(ds->w) / ds->gw : 1.0f);
        }
    }
    uint32_t cull = s.rs[kRsCullMode]; k.cull = cull != 0 && a.topology == Topology::Triangles; k.ccw = cull == 0x900; k.fill = s.rs[kRsFillMode];
}
// ---- XBE_VK_DEBUG_DRAW: one draw in full, every 60th present ----
// The vertex stage worked on the CPU the way the generated HLSL does it (VertexProgramHlsl's microcode
// fields, FixedVertexHlsl's pretransformed path, kVsEpilogue's pixel to clip step), so the log states
// where a draw lands and what it samples instead of leaving it to be guessed from a picture.
uint32_t VsField(const uint32_t* t, int sub, int bit, int len) { return (t[sub] >> bit) & ((1u << len) - 1); }
void VsWrite(float* dst, const float* src, uint32_t mask) { for (int i = 0; i < 4; i++) if ((mask >> (3 - i)) & 1) dst[i] = src[i]; }
float VsMagClamp(float v) { float m = std::clamp(std::fabs(v), 5.42101e-20f, 1.84467e+19f); return v < 0 ? -m : m; }
// out: by output register, 0 oPos, 3 oD0, 4 oD1, 5 oFog, 6 oPts, 7 oB0, 8 oB1, 9 to 12 oT0 to oT3
void EvalVertexProgram(const std::vector<uint32_t>& prog, uint16_t regs, const float (&v)[16][4], const std::array<std::array<float, 4>, 192>& c,
                       float (&out)[16][4], std::vector<int>& reads) {
    float R[13][4] = {};  // R0 to R11, R12 is oPos
    std::memset(out, 0, sizeof(out));
    for (int o : {3, 7, 9, 10, 11, 12}) out[o][3] = 1;
    out[6][0] = out[6][1] = out[6][2] = out[6][3] = 1;
    int a0 = 0;
    auto operand = [&](const uint32_t* t, int which, float (&o)[4]) {
        uint32_t neg, sw[4], reg, mux;
        if (which == 0) { neg = VsField(t, 1, 8, 1); sw[0] = VsField(t, 1, 6, 2); sw[1] = VsField(t, 1, 4, 2); sw[2] = VsField(t, 1, 2, 2); sw[3] = VsField(t, 1, 0, 2); reg = VsField(t, 2, 28, 4); mux = VsField(t, 2, 26, 2); }
        else if (which == 1) { neg = VsField(t, 2, 25, 1); sw[0] = VsField(t, 2, 23, 2); sw[1] = VsField(t, 2, 21, 2); sw[2] = VsField(t, 2, 19, 2); sw[3] = VsField(t, 2, 17, 2); reg = VsField(t, 2, 13, 4); mux = VsField(t, 2, 11, 2); }
        else {
            neg = VsField(t, 2, 10, 1); sw[0] = VsField(t, 2, 8, 2); sw[1] = VsField(t, 2, 6, 2); sw[2] = VsField(t, 2, 4, 2); sw[3] = VsField(t, 2, 2, 2);
            reg = VsField(t, 2, 0, 2) << 2 | VsField(t, 3, 30, 2); mux = VsField(t, 3, 28, 2);
            uint32_t ilu = VsField(t, 1, 25, 3); if (ilu >= 2 && ilu <= 6) sw[1] = sw[2] = sw[3] = sw[0];
        }
        const float one[4] = {1, 1, 1, 1}, unit[4] = {0, 0, 0, 1}; const float* base;
        if (mux == 1) base = R[std::min<uint32_t>(reg, 12)];
        else if (mux == 2) { uint32_t vi = VsField(t, 1, 9, 4); base = (regs >> vi) & 1 ? v[vi] : vi == 3 ? one : unit; }
        else { uint32_t cr = VsField(t, 1, 13, 8); int idx = VsField(t, 3, 1, 1) ? std::clamp(a0 + int(cr), 0, 191) : int(cr); base = c[size_t(idx)].data(); reads.push_back(idx); }
        for (int i = 0; i < 4; i++) o[i] = neg ? -base[sw[i]] : base[sw[i]];
    };
    for (size_t n = 0; n + 4 <= prog.size(); n += 4) {
        const uint32_t* t = &prog[n];
        uint32_t mac = VsField(t, 1, 21, 4), ilu = VsField(t, 1, 25, 3);
        if (mac || ilu) {
            float A[4], B[4], C[4], mr[4] = {}, ir[4] = {};
            operand(t, 0, A); operand(t, 1, B); operand(t, 2, C);
            auto mul = [](float x, float y) { return x == 0 || y == 0 ? 0.0f : x * y; };
            switch (mac) {
                case 1: case 13: std::memcpy(mr, A, sizeof(mr)); break;
                case 2: for (int i = 0; i < 4; i++) mr[i] = mul(A[i], B[i]); break;
                case 3: for (int i = 0; i < 4; i++) mr[i] = A[i] + C[i]; break;
                case 4: for (int i = 0; i < 4; i++) mr[i] = mul(A[i], B[i]) + C[i]; break;
                case 5: mr[0] = mr[1] = mr[2] = mr[3] = A[0] * B[0] + A[1] * B[1] + A[2] * B[2]; break;
                case 6: mr[0] = mr[1] = mr[2] = mr[3] = A[0] * B[0] + A[1] * B[1] + A[2] * B[2] + B[3]; break;
                case 7: mr[0] = mr[1] = mr[2] = mr[3] = A[0] * B[0] + A[1] * B[1] + A[2] * B[2] + A[3] * B[3]; break;
                case 8: mr[0] = 1; mr[1] = A[1] * B[1]; mr[2] = A[2]; mr[3] = B[3]; break;
                case 9: for (int i = 0; i < 4; i++) mr[i] = std::min(A[i], B[i]); break;
                case 10: for (int i = 0; i < 4; i++) mr[i] = std::max(A[i], B[i]); break;
                case 11: for (int i = 0; i < 4; i++) mr[i] = A[i] < B[i] ? 1.0f : 0.0f; break;
                case 12: for (int i = 0; i < 4; i++) mr[i] = A[i] >= B[i] ? 1.0f : 0.0f; break;
                default: break;
            }
            float x = C[0], whole = std::floor(x), mag = std::fabs(x), l2 = std::log2(mag), ex = std::floor(l2);
            switch (ilu) {
                case 1: std::memcpy(ir, C, sizeof(ir)); break;
                case 2: ir[0] = ir[1] = ir[2] = ir[3] = 1.0f / x; break;
                case 3: ir[0] = ir[1] = ir[2] = ir[3] = VsMagClamp(1.0f / x); break;
                case 4: ir[0] = ir[1] = ir[2] = ir[3] = x == 0 ? INFINITY : 1.0f / std::sqrt(mag); break;
                case 5: ir[0] = std::exp2(whole); ir[1] = x - whole; ir[2] = std::exp2(x); ir[3] = 1; break;
                case 6: if (mag == 0) { ir[0] = ir[2] = -INFINITY; ir[1] = ir[3] = 1; } else { ir[0] = ex; ir[1] = mag / std::exp2(ex); ir[2] = l2; ir[3] = 1; } break;
                case 7: { const float cap = 127.99609375f; float d = std::max(C[0], 0.0f);
                          ir[0] = 1; ir[1] = d; ir[2] = d > 0 ? std::exp2(std::clamp(C[3], -cap, cap) * std::log2(std::max(C[1], 0.0f))) : 0; ir[3] = 1; break; }
                default: break;
            }
            uint32_t r = VsField(t, 3, 20, 4), omask = VsField(t, 3, 12, 4);
            if (mac == 13) a0 = int(std::floor(mr[0] + 1e-3f));
            else if (mac) { uint32_t m = VsField(t, 3, 24, 4); if (ilu && r == 1) m = 0; if (m && r <= 12) VsWrite(R[r], mr, m); }
            if (ilu) { uint32_t m = VsField(t, 3, 16, 4), d = mac ? 1 : r; if (m && d <= 12) VsWrite(R[d], ir, m); }
            bool useIlu = VsField(t, 3, 2, 1) == 1;
            if (omask && VsField(t, 3, 11, 1) && ((useIlu && ilu) || (!useIlu && mac && mac != 13))) {
                uint32_t o = VsField(t, 3, 3, 8) & 0xF;
                VsWrite(o == 0 ? R[12] : out[o], useIlu ? ir : mr, omask);
            }
        }
        if (VsField(t, 3, 0, 1)) break;
    }
    std::memcpy(out[0], R[12], sizeof(out[0]));
}
// one register of one vertex as the shader reads it: the element's components, (0, 0, 0, 1) behind them, then
// the decode the vertex shader finishes a raw element with
void ReadVertexRegister(const Assembled& a, int r, uint32_t vtx, float (&o)[4]) {
    o[0] = o[1] = o[2] = 0; o[3] = 1;
    const VertexElement& e = a.elements[size_t(r)]; const VertexBuffer& vb = a.buffers[e.buffer];
    if (!vb.src || !vb.stride || uint64_t(vtx + 1) * vb.stride > vb.bytes) return;
    const uint8_t* p = vb.src + size_t(vtx) * vb.stride + e.offset;
    auto f32 = [&](int n) { std::memcpy(o, p, size_t(n) * 4); };
    auto i16 = [&](int n, float div) { for (int i = 0; i < n; i++) { int16_t x; std::memcpy(&x, p + i * 2, 2); o[i] = div ? std::max(x / div, -1.0f) : float(x); } };
    auto u8 = [&](int n) { for (int i = 0; i < n; i++) o[i] = p[i] / 255.0f; };
    switch (e.format) {
        case kDxgiRgba32f: f32(4); break; case kDxgiRgb32f: f32(3); break; case kDxgiRg32f: f32(2); break; case kDxgiR32f: f32(1); break;
        case kDxgiR32u: { uint32_t u; std::memcpy(&u, p, 4); o[0] = float(u); break; }
        case kDxgiRgba16Snorm: i16(4, 32767); break; case kDxgiRg16Snorm: i16(2, 32767); break; case kDxgiR16Snorm: i16(1, 32767); break;
        case kDxgiRgba16Sint: i16(4, 0); break; case kDxgiRg16Sint: i16(2, 0); break; case kDxgiR16Sint: i16(1, 0); break;
        case kDxgiRgba8: u8(4); break; case kDxgiRg8: u8(2); break; case kDxgiR8: u8(1); break;
        case kDxgiBgra8: o[0] = p[2] / 255.0f; o[1] = p[1] / 255.0f; o[2] = p[0] / 255.0f; o[3] = p[3] / 255.0f; break;
        default: break;
    }
    switch (uint32_t(a.decode >> (r * 4)) & 15) {
        case kDecodeW1: case kDecodeIntW1: o[3] = 1; break;
        case kDecodePacked: { uint32_t u; std::memcpy(&u, p, 4);
            o[0] = std::max(float(int32_t(u << 21) >> 21) / 1023.0f, -1.0f); o[1] = std::max(float(int32_t(u << 10) >> 21) / 1023.0f, -1.0f);
            o[2] = std::max(float(int32_t(u) >> 22) / 511.0f, -1.0f); o[3] = 1; break; }
        case kDecodeFloat2H: o[3] = o[2]; o[2] = 0; break;
        default: break;
    }
}
void VulkanBackend::LogDraw(const State& s, uint32_t prim, uint32_t count, const Assembled& a, const DrawShaders& sh, const DrawTextures& stages,
                            const DrawConstants& c, GuestExtent size) {
    const Target* rt = boundRt_ ? boundRt_ : boundDs_;
    const uint32_t W = rt ? rt->w : 0, H = rt ? rt->h : 0;
    LOGI("vk-draw: present %llu rt %08X prim %u count %u | %u vertices, %u indices, draw count %u, topology %u strip %d",
         (unsigned long long)presents_, s.rt.data, prim, count, a.vertex_count, a.index_count, a.draw_count, uint32_t(a.topology), int(a.strip));
    LOGI("vk-draw:  vertex shader %08X: %s, pretransformed %d, quad %d, registers %04X, decode %llX",
         s.vs_handle, sh.prog ? "program" : "fixed function (FVF)", int(a.pretransformed), int(sh.quad), a.regs, (unsigned long long)a.decode);
    if (s.vs && (s.vs_handle & 1)) {
        std::string decl; char tok[16];
        for (uint32_t d : s.vs->decl) { std::snprintf(tok, sizeof(tok), " %08X", d); decl += tok; }
        LOGI("vk-draw:  declaration%s | program %zu instructions", decl.c_str(), s.vs->program.size() / 4);
    }
    const auto& v = s.viewport;
    std::string sc; char buf[96];
    for (uint32_t i = 0; i < s.scissor_count; i++) { std::snprintf(buf, sizeof(buf), " (%u,%u)-(%u,%u)", s.scissors[i * 4], s.scissors[i * 4 + 1], s.scissors[i * 4 + 2], s.scissors[i * 4 + 3]); sc += buf; }
    LOGI("vk-draw:  viewport (%u,%u %ux%u z %.3f..%.3f) scissors %u%s | screen space offset (%.5f,%.5f) zscale %.0f | multisample antialias %u mask %08X mode %u rendertarget mode %u",
         v.x, v.y, v.w, v.h, double(v.min_z), double(v.max_z), s.scissor_count, sc.c_str(), double(s.ss_offset_x), double(s.ss_offset_y), double(s.zscale),
         s.rs[kRsMultiSampleAntiAlias], s.rs[kRsMultiSampleMask], s.rs[kRsMultiSampleMode], s.rs[kRsMultiSampleRenderTargetMode]);
    LOGI("vk-draw:  target guest %ux%u host %ux%u (render scale %.3f) | guest extent %ux%u dx %.1f supersample %.0fx%.0f | surf (%.1f, %.1f, %.3g)",
         rt ? rt->gw : 0, rt ? rt->gh : 0, W, H, double(scale_), size.w, size.h, double(size.dx), double(s.super_x), double(s.super_y),
         double(c.surf[0]), double(c.surf[1]), double(c.surf[2]));
    for (int i = 0; i < 4; i++) {
        if (!stages.hasDesc[i]) continue;
        const TextureDesc& d = stages.desc[i];
        LOGI("vk-draw:  stage %d texture %08X format %02X %s %ux%u depth %u levels %u pitch %u linear %d cube %d, %u bytes | target %d kind %u rect %u | texsize (%.0f, %.0f, %.6f, %.6f) uvrect (%.5f,%.5f)-(%.5f,%.5f)",
             i, d.data, d.format, Format(d.format).name, d.width, d.height, d.depth, d.levels, d.pitch, int(d.linear), int(d.cube), TotalSize(d),
             int(stages.rt[i]), stages.info[i].kind, stages.info[i].rect, double(c.texsize[i][0]), double(c.texsize[i][1]), double(c.texsize[i][2]), double(c.texsize[i][3]),
             double(c.uvrect[i][0]), double(c.uvrect[i][1]), double(c.uvrect[i][2]), double(c.uvrect[i][3]));
    }
    for (int r = 0; r < 16; r++) if ((a.regs >> r) & 1) {
        const VertexElement& e = a.elements[size_t(r)];
        LOGI("vk-draw:  register %d: format %u, buffer %u offset %u stride %u", r, e.format, e.buffer, e.offset, a.buffers[e.buffer].stride);
    }
    std::vector<int> reads;
    for (uint32_t vtx = 0; vtx < std::min(a.vertex_count, 4u); vtx++) {
        float in[16][4] = {}, out[16][4] = {};
        std::string regs;
        for (int r = 0; r < 16; r++) if ((a.regs >> r) & 1) {
            ReadVertexRegister(a, r, vtx, in[r]);
            std::snprintf(buf, sizeof(buf), " v%d (%.4f, %.4f, %.4f, %.4f)", r, double(in[r][0]), double(in[r][1]), double(in[r][2]), double(in[r][3])); regs += buf;
        }
        LOGI("vk-draw:  vertex %u:%s", vtx, regs.c_str());
        float p[4];
        if (sh.prog) { EvalVertexProgram(s.vs->program, a.regs, in, s.vsc, out, reads); std::memcpy(p, out[0], sizeof(p)); }
        else if (a.pretransformed) { p[0] = in[0][0]; p[1] = in[0][1]; p[2] = in[0][2] / c.surf[2]; p[3] = in[0][3] == 0 ? 1 : 1 / in[0][3]; std::memcpy(out[9], in[9], sizeof(out[9])); }
        else {
            // FillConstants' wvp already holds the viewport: clip space straight out, shown here as the pixel that is
            float cp[4]; for (int i = 0; i < 4; i++) cp[i] = in[0][0] * c.wvp[i * 4] + in[0][1] * c.wvp[i * 4 + 1] + in[0][2] * c.wvp[i * 4 + 2] + c.wvp[i * 4 + 3];
            p[0] = (cp[0] / cp[3] + 1) * 0.5f * c.surf[0]; p[1] = (1 - cp[1] / cp[3]) * 0.5f * c.surf[1]; p[2] = cp[2] / cp[3] / c.surf[2]; p[3] = cp[3];
            std::memcpy(out[9], in[9], sizeof(out[9]));
        }
        // kVsEpilogue: a pass ending half a pixel short of the far edges is pulled out to them, then pixels to clip
        float px = std::fabs(p[0] - (c.surf[0] - 0.5f)) < 1e-3f ? c.surf[0] : p[0], py = std::fabs(p[1] - (c.surf[1] - 0.5f)) < 1e-3f ? c.surf[1] : p[1];
        float nx = px * 2 / c.surf[0] - 1, ny = 1 - py * 2 / c.surf[1], nz = p[2] * c.surf[2];
        float hx = (nx + 1) * 0.5f * float(W), hy = (1 - ny) * 0.5f * float(H);
        const float* tc = out[9]; float uv[2] = {tc[0], tc[1]};
        if (sh.prog && tc[3] != 0) { uv[0] /= tc[3]; uv[1] /= tc[3]; }
        if (stages.hasDesc[0] && stages.info[0].rect) { uv[0] /= c.texsize[0][0]; uv[1] /= c.texsize[0][1]; }
        uv[0] += c.texsize[0][2]; uv[1] += c.texsize[0][3];
        LOGI("vk-draw:  vertex %u: oPos (%.4f, %.4f, %.4f, %.4f) -> clip (%.5f, %.5f, %.5f) -> host pixel (%.2f, %.2f) = guest pixel (%.3f, %.3f) | oT0 (%.4f, %.4f, %.4f, %.4f) -> uv (%.5f, %.5f)",
             vtx, double(p[0]), double(p[1]), double(p[2]), double(p[3]), double(nx), double(ny), double(nz), double(hx), double(hy),
             double(W ? hx * float(size.w) / float(W) : 0), double(H ? hy * float(size.h) / float(H) : 0),
             double(tc[0]), double(tc[1]), double(tc[2]), double(tc[3]), double(uv[0]), double(uv[1]));
    }
    std::sort(reads.begin(), reads.end()); reads.erase(std::unique(reads.begin(), reads.end()), reads.end());
    for (int r : reads) LOGI("vk-draw:  c[%d] (D3D c%d) = (%.4f, %.4f, %.4f, %.4f)", r, r - 96, double(s.vsc[size_t(r)][0]), double(s.vsc[size_t(r)][1]), double(s.vsc[size_t(r)][2]), double(s.vsc[size_t(r)][3]));
}
void VulkanBackend::Draw(const State& s, uint32_t prim, uint32_t start, uint32_t count, const uint8_t* indices) {
    VkPerfScope perfDraw{g_vkPerf.draw};
    g_vkPerf.draws++;
    if (cap_) CapStep(s.rt);
    if (DepthCopyDraw(s)) return;
    Assembled& a = asm_;
    if (!Assemble(s, prim, start, count, indices, a)) { g_vkPerf.skipped++; if (count) LOG_FIRST(16, "vulkan: draw skipped, nothing to assemble: rt %08X prim %u count %u", s.rt.data, prim, count); return; }
    g_vkPerf.verts += a.vertex_count; g_vkPerf.raw += a.vertices.empty() ? 1 : 0;
    if (recording_ && Cur().used > kUploadSize - kUploadReserve) Submit(true);  // Cur().used is only meaningful in the open segment
    if (!BindTargets(s)) { g_vkPerf.skipped++; return; }
    DrawShaders sh = ClassifyShaders(s, a, count);
    if (sh.prog && !s.vs->hash) s.vs->hash = Hash(s.vs->program.data(), s.vs->program.size() * 4) | 1;
    sh.vkey = VertexKeyOf(s, a, sh); EnsureVertexShader(s, a, sh);
    DrawTextures stages; if (!BindStageTextures(s, stages)) { g_vkPerf.skipped++; return; }
    PixelKey pk = MakePixelKey(s, stages.info); pk.quad = sh.quad; sh.pkey = Hash(&pk, sizeof(pk)); EnsurePixelShader(pk, sh);
    if (!sh.vs || !sh.ps) { g_vkPerf.skipped++; g_vkPerf.nopipeline++; g_vkPerf.nopipeShader++; LOG_FIRST(16, "vulkan: draw skipped, no shader: rt %08X", s.rt.data); return; }
    PsoKey k; BuildPsoKey(s, a, sh.vh, sh.ph, boundRt_, boundDs_, k);
    VkPipeline pso = EnsurePipeline(k, sh.vs, sh.ps);
    if (!pso) { g_vkPerf.skipped++; g_vkPerf.nopipeline++; g_vkPerf.nopipePso++; LOG_FIRST(16, "vulkan: draw skipped, no pipeline: rt %08X vs %llx ps %llx", s.rt.data, (unsigned long long)sh.vh, (unsigned long long)sh.ph); return; }
    GuestExtent size = BoundExtent(boundRt_ ? s.rt : s.zs);
    DrawConstants c{}; FillUvRect(s, prim, count, a, sh, stages, c); FillConstants(s, sh, stages, size, c);
    static const bool debugDraw = std::getenv("XBE_VK_DEBUG_DRAW") != nullptr;
    if (debugDraw && presents_ % 60 == 0) LogDraw(s, prim, count, a, sh, stages, c, size);
    Alloc vc, dc;
    bool sameVsc, sameDc;
    EnsureSetRoom();  // before the constants: a flush here restarts the upload ring
    sameVsc = vscValid_ && std::memcmp(vscLast_.data(), s.vsc.data(), sizeof(s.vsc)) == 0;
    if (sameVsc) vc = vscAlloc_;
    else {
        vc = UploadAlloc(sizeof(s.vsc), 256);
        if (!vc.cpu) { g_vkPerf.skipped++; return; }
        std::memcpy(vc.cpu, s.vsc.data(), sizeof(s.vsc));
        std::memcpy(vscLast_.data(), s.vsc.data(), sizeof(s.vsc));
        vscAlloc_ = vc; vscValid_ = true;
    }
    sameDc = dcValid_ && std::memcmp(&dcLast_, &c, sizeof(c)) == 0;
    if (sameDc) dc = dcAlloc_;
    else {
        dc = UploadAlloc(sizeof(DrawConstants), 256);
        if (!dc.cpu) { g_vkPerf.skipped++; return; }
        std::memcpy(dc.cpu, &c, sizeof(c));
        dcLast_ = c; dcAlloc_ = dc; dcValid_ = true;
    }
    float bf[4]; ColorF(SimpleState(s, kMBlendColor), bf);
    std::array<VkSampler, 4> samplers; bool clampCoords = UnitTexcoordsOf(a);
    for (int i = 0; i < 4; i++) samplers[i] = StageSamplerFor(s, i, stages, clampCoords);
    VkDescriptorSet set;
    // Nothing the set describes changed, so the previous one still describes it.
    bool same = setValid_ && sameVsc && sameDc && samplers == samplerLast_;
    for (int i = 0; same && i < 4; i++) same = stages.view[i] == viewLast_[i];
    if (same) set = setLast_;
    else {
        set = AllocDrawSet(vc.buffer, vc.off, dc.buffer, dc.off, stages, samplers);
        for (int i = 0; i < 4; i++) viewLast_[i] = stages.view[i];
        samplerLast_ = samplers; setLast_ = set; setValid_ = set != VK_NULL_HANDLE;
    }
    if (!set) { g_vkPerf.skipped++; g_vkPerf.nopipeline++; g_vkPerf.nopipeSet++; LOG_FIRST(16, "vulkan: draw skipped, no descriptor set: rt %08X present %llu", s.rt.data, (unsigned long long)presents_); return; }
    VkFramebuffer fb; BeginTargetPass(boundRt_, boundDs_, fb);
    if (pso != boundPipeline_) { vkCmdBindPipeline(Cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS, pso); boundPipeline_ = pso; }
    if (set != boundSet_) { vkCmdBindDescriptorSets(Cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_, 0, 1, &set, 0, nullptr); boundSet_ = set; }
    if (std::memcmp(boundBlend_, bf, sizeof(bf)) != 0) { vkCmdSetBlendConstants(Cmd(), bf); std::memcpy(boundBlend_, bf, sizeof(bf)); }
    SetViewportScissor(s, size, boundRt_ ? boundRt_->w : boundDs_->w, boundRt_ ? boundRt_->h : boundDs_->h);
    DrawGeometry(a);
    if (cap_) std::fprintf(cap_, "%04u draw rt %08X ds %08X prim %u count %u vs %08X\n", capI_++, s.rt.data, boundDs_ ? s.zs.data : 0, prim, count, s.vs_handle);
}
// ---- Clear ----
struct ScaledRectI { int32_t x0, y0, x1, y1; };
ScaledRectI ScaledRectOf(const Target& t, int x0, int y0, int x1, int y1) {
    float gw = float(t.gw) + t.dx * 2; float sc = float(t.w) / gw;
    int32_t rx0 = int32_t((x0 + t.dx) * sc), rx1 = int32_t((x1 + t.dx) * sc);
    if (t.dx != 0) { if (x0 <= 0) rx0 = 0; if (x1 >= int(t.gw)) rx1 = int32_t(t.w); }
    return {std::clamp<int32_t>(rx0, 0, int32_t(t.w)), std::clamp<int32_t>(int32_t(y0 * sc), 0, int32_t(t.h)),
            std::clamp<int32_t>(rx1, 0, int32_t(t.w)), std::clamp<int32_t>(int32_t(y1 * sc), 0, int32_t(t.h))};
}
void VulkanBackend::Clear(const State& s, uint32_t count, const uint8_t* rects, uint32_t flags, uint32_t color, float z, uint32_t stencil) {
    VkPerfScope perfClear{g_vkPerf.clear}; g_vkPerf.clears++;
    if (cap_) { CapStep(s.rt); std::fprintf(cap_, "%04u clear rt %08X ds %08X flags %X color %08X z %.3f\n", capI_++, s.rt.data, s.zs.data, flags, color, z); }
    Target* rt = (flags & kClearTarget) ? EnsureTarget(s.rt, false) : nullptr;
    Target* ds = (flags & (kClearZ | kClearStencil)) ? EnsureTarget(s.zs, true) : nullptr;
    for (Target* t : {rt, ds}) {
        if (!t) continue;
        std::vector<ScaledRectI> rs;
        if (count && rects) for (uint32_t i = 0; i < count; i++) { int32_t r[4]; std::memcpy(r, rects + size_t(i) * sizeof(r), sizeof(r)); rs.push_back(ScaledRectOf(*t, r[0], r[1], r[2], r[3])); }
        else { const auto& v = s.viewport; rs.push_back(ScaledRectOf(*t, int(v.x), int(v.y), int(v.x + v.w), int(v.y + v.h))); }
        if (!t->depth && LatchCpuWrite(*t)) UploadTarget(*t, s.rt);
        VkFramebuffer fb; BeginTargetPass(t->depth ? nullptr : t, t->depth ? t : nullptr, fb);
        std::vector<VkClearRect> vrs;
        for (auto& r : rs) if (r.x1 > r.x0 && r.y1 > r.y0) { VkClearRect cr{}; cr.rect.offset = {r.x0, r.y0}; cr.rect.extent = {uint32_t(r.x1 - r.x0), uint32_t(r.y1 - r.y0)}; cr.baseArrayLayer = 0; cr.layerCount = 1; vrs.push_back(cr); }
        if (!vrs.empty()) {
            VkClearAttachment a{};
            if (t->depth) {
                if (flags & kClearZ) a.aspectMask |= VK_IMAGE_ASPECT_DEPTH_BIT;
                if (flags & kClearStencil) a.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
                a.clearValue.depthStencil = {std::clamp(z, 0.0f, 1.0f), stencil & 0xFF};
            } else {
                // kClearTarget (0xF0) is the full ARGB mask; a partial mask clear is done as a
                // full clear here (ClearAttachments has no channel mask) - the full mask is
                // what titles pass, per backend.h.
                float col[4]; ColorF(color, col);
                a.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; std::memcpy(a.clearValue.color.float32, col, sizeof(col));
            }
            vkCmdClearAttachments(Cmd(), 1, &a, uint32_t(vrs.size()), vrs.data());
        }
        t->drawn = true;
    }
}
// ---- pixel images, blits, shrink, read-back ----
ImageRes VulkanBackend::CreatePixelImage(const std::vector<uint8_t>& px, uint32_t pitch, uint32_t w, uint32_t h) {
    ImageRes tex = CreateImage(w, h, kBgra, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    if (!tex.Valid()) return tex;
    Transition(tex, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    Alloc a = UploadAlloc(uint64_t(pitch) * h, 16);
    if (a.cpu) {
        std::memcpy(a.cpu, px.data(), std::min<uint64_t>(px.size(), uint64_t(pitch) * h));
        VkBufferImageCopy region{}; region.bufferOffset = a.off; region.bufferRowLength = pitch / 4;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; region.imageSubresource.layerCount = 1; region.imageExtent = {w, h, 1};
        vkCmdCopyBufferToImage(Cmd(), a.buffer, tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    }
    Transition(tex, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL); return tex;
}
void VulkanBackend::UploadTarget(Target& t, const Surface& s) {
    TextureDesc d; if (!Describe(s.header, d) || Format(d.format).depth) return;
    std::vector<uint8_t> px; uint32_t pitch;
    bool over = !t.drawn && d.width == t.gw && d.height == t.gh && t.read.size() == size_t(t.gw) * t.gh * 4;
    if (over) {
        pitch = t.gw * 4; px.assign(t.read.size(), 0);
        uint32_t guestPitch = GuestPitch(t.gw); const uint8_t* guest = static_cast<const uint8_t*>(kernel::HostPtr(t.data));
        for (uint32_t y = 0; y < t.gh; y++) for (uint32_t x = 0; x < t.gw; x++) {
            const uint8_t* cc = guest + size_t(y) * guestPitch + x * 4; size_t i = (size_t(y) * t.gw + x) * 4;
            if (!std::memcmp(cc, &t.read[i], 4)) continue;
            std::memcpy(&t.read[i], cc, 4); std::memcpy(&px[i], cc, 3); px[i + 3] = 255;
        }
    } else DecodeLevel(d, 0, 0, static_cast<const uint8_t*>(kernel::HostPtr(d.data)), nullptr, px, pitch);
    ImageRes tex = CreatePixelImage(px, pitch, d.width, d.height);
    if (!tex.Valid()) return;
    float uv[4] = {0, 0, 1, 1};
    if (t.dx != 0) {
        // Guest pixels are the console's width: they land in the widened
        // target's central columns, not stretched across the whole target.
        float sc = float(t.w) / (float(t.gw) + t.dx * 2);
        std::array<float, 4> vp{t.dx * sc, 0, float(t.gw) * sc, float(t.h)};
        PrepareBlitSource(tex, false);
        VkFramebuffer fb; BeginTargetPass(&t, nullptr, fb);
        BlitToImage(kBgra, t.w, t.h, tex, uv, false, true, 1, 1.0f, over, &vp);
        t.drawn = true; boundRt_ = boundDs_ = nullptr;
    } else BlitInto(t, tex, uv, false, true, 1, 1.0f, over);
    DeferDestroyImage(tex);
    t.epoch = DirtyEpoch(t.data); t.sig = MemSig(t.data, t.bytes);
}
void VulkanBackend::PrepareBlitSource(ImageRes& src, bool gamma) {
    // Layout transitions are image barriers: they MUST be recorded outside any render
    // pass. BlitToImage is called from inside the destination pass (its caller has
    // already begun it), so the source transition happens here, before that pass
    // begins. Recording it inside the pass is a validation error and, on RADV, leaves
    // the source in COLOR_ATTACHMENT_OPTIMAL while the shader reads it as
    // SHADER_READ_ONLY - the read then returns stale/DCC-garbage colours. The same
    // holds for the descriptor set BlitToImage allocates inside that pass: its room
    // is made here, where submitting a full segment is still possible.
    EnsureSetRoom();
    Transition(src, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    if (gamma) Transition(lut_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}
void VulkanBackend::BlitToImage(VkFormat rtvFormat, uint32_t w, uint32_t h, ImageRes& src, const float uv[4], bool gamma, bool alpha, uint32_t taps, float mul, bool over, const std::array<float, 4>* viewportRect, bool swap) {
    // Caller has begun the destination render pass and called PrepareBlitSource.
    VkPipeline pipe = EnsureBlitPipeline(rtvFormat, over, false, swap); if (!pipe) return;
    DrawTextures stages; stages.view[0] = src.SampleView(); stages.view[1] = lut_.SampleView();
    std::array<VkSampler, 4> samplers{nullSampler_, nullSampler_, nullSampler_, nullSampler_};
    { VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO}; si.magFilter = si.minFilter = VK_FILTER_LINEAR; si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; si.maxLod = VK_LOD_CLAMP_NONE; samplers[0] = GetSampler(si);
      si.magFilter = si.minFilter = VK_FILTER_NEAREST; si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST; samplers[1] = GetSampler(si); }
    Alloc cb = UploadAlloc(256, 256); if (!cb.cpu) return;
    float cc[8] = {uv[0], uv[1], uv[2], uv[3], gamma ? 1.0f : 0, alpha ? 1.0f : 0, float(taps), mul};
    std::memcpy(cb.cpu, cc, sizeof(cc));
    VkDescriptorSet set = AllocDrawSet(cb.buffer, cb.off, cb.buffer, cb.off, stages, samplers); if (!set) { LOG_FIRST(16, "vulkan: blit dropped, no descriptor set (rtv %d %ux%u present %llu)", int(rtvFormat), w, h, (unsigned long long)presents_); return; }
    vkCmdBindPipeline(Cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdBindDescriptorSets(Cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_, 0, 1, &set, 0, nullptr);
    if (viewportRect) {
        VkViewport vp{(*viewportRect)[0], (*viewportRect)[1] + (*viewportRect)[3], (*viewportRect)[2], -(*viewportRect)[3], 0, 1};
        vkCmdSetViewport(Cmd(), 0, 1, &vp); VkRect2D sc{{0, 0}, {w, h}}; vkCmdSetScissor(Cmd(), 0, 1, &sc);
    } else { VkViewport vp{0, float(h), float(w), -float(h), 0, 1}; vkCmdSetViewport(Cmd(), 0, 1, &vp); VkRect2D sc{{0, 0}, {w, h}}; vkCmdSetScissor(Cmd(), 0, 1, &sc); }
    vkCmdDraw(Cmd(), 3, 1, 0, 0);
    InvalidateBinds();
}
void VulkanBackend::BlitInto(Target& dst, ImageRes& src, const float uv[4], bool gamma, bool alpha, uint32_t taps, float mul, bool over) {
    PrepareBlitSource(src, gamma);
    VkFramebuffer fb; BeginTargetPass(&dst, nullptr, fb);
    BlitToImage(kBgra, dst.w, dst.h, src, uv, gamma, alpha, taps, mul, over, nullptr);
    dst.drawn = true; boundRt_ = boundDs_ = nullptr;
}
ImageRes VulkanBackend::ShrinkImage(ImageRes& src, uint32_t w, uint32_t h, uint32_t taps, float mul, ImageRes* reuse) {
    ImageRes down;
    if (reuse && reuse->Valid() && reuse->w == w && reuse->h == h) down = *reuse;
    else { down = CreateImage(w, h, kBgra, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT); if (!down.Valid()) return down; }
    Target tmp; tmp.res = down; tmp.w = down.w; tmp.h = down.h; tmp.gw = w; tmp.gh = h;
    float uv[4] = {0, 0, 1, 1}; BlitInto(tmp, src, uv, false, true, taps, mul, false);
    down = tmp.res; if (reuse) *reuse = down; return down;
}
bool VulkanBackend::ReadBackImage(ImageRes& src, uint32_t w, uint32_t h, std::vector<uint8_t>& out) {
    Transition(src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    VkBuffer buf = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE;
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; ci.size = uint64_t(w) * h * 4; ci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    // D6: every step is checked; a failure returns false with nothing mapped, instead
    // of memcpy'ing from a null map of memory bound through a guessed type 0.
    if (vkCreateBuffer(device_, &ci, nullptr, &buf) != VK_SUCCESS) return false;
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(device_, buf, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = mr.size;
    bool coherent = true;
    ai.memoryTypeIndex = FindHostMemory(mr.memoryTypeBits, coherent);
    if (ai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(device_, &ai, nullptr, &mem) != VK_SUCCESS) { vkDestroyBuffer(device_, buf, nullptr); return false; }
    if (vkBindBufferMemory(device_, buf, mem, 0) != VK_SUCCESS) { vkDestroyBuffer(device_, buf, nullptr); vkFreeMemory(device_, mem, nullptr); return false; }
    VkBufferImageCopy region{}; region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; region.imageSubresource.layerCount = 1; region.imageExtent = {w, h, 1};
    vkCmdCopyImageToBuffer(Cmd(), src.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1, &region);
    Flush();
    void* p = nullptr;
    if (vkMapMemory(device_, mem, 0, VK_WHOLE_SIZE, 0, &p) != VK_SUCCESS || !p) { vkDestroyBuffer(device_, buf, nullptr); vkFreeMemory(device_, mem, nullptr); return false; }
    if (!coherent) {  // D6: HOST_VISIBLE-only memory needs an invalidate before the CPU read
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = mem; range.offset = 0; range.size = VK_WHOLE_SIZE;
        if (vkInvalidateMappedMemoryRanges(device_, 1, &range) != VK_SUCCESS) { vkUnmapMemory(device_, mem); vkDestroyBuffer(device_, buf, nullptr); vkFreeMemory(device_, mem, nullptr); return false; }
    }
    out.resize(size_t(w) * h * 4); std::memcpy(out.data(), p, out.size());
    vkUnmapMemory(device_, mem); vkDestroyBuffer(device_, buf, nullptr); vkFreeMemory(device_, mem, nullptr); return true;
}
ImageRes VulkanBackend::PrepareCopySource(Target& t, ImageRes* reuseDown) {
    if (t.dx != 0) {
        // A widened frame target holds more world at the sides; guest memory is
        // still the console's width, so a read-back (Sync, captures) delivers the
        // central guest columns, never the squeezed widened picture.
        ImageRes down;
        if (reuseDown && reuseDown->Valid() && reuseDown->w == t.gw && reuseDown->h == t.gh) down = *reuseDown;
        else {
            down = CreateImage(t.gw, t.gh, kBgra, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
            if (!down.Valid()) return down;
        }
        Target tmp; tmp.res = down; tmp.w = down.w; tmp.h = down.h; tmp.gw = t.gw; tmp.gh = t.gh;
        float wide = float(t.gw) + t.dx * 2;
        float uv[4] = {t.dx / wide, 0, float(t.gw) / wide, 1};
        BlitInto(tmp, t.res, uv, false, true, 1, 1.0f, false);
        down = tmp.res; if (reuseDown) *reuseDown = down; return down;
    }
    if (t.w != t.gw || t.h != t.gh) return ShrinkImage(t.res, t.gw, t.gh, 1, 1.0f, reuseDown);
    return t.res;
}
// ---- Copy ----
struct GuestRectI { int32_t left, top, right, bottom; };
GuestRectI ReadGuestRect(uint32_t rects, uint32_t i) { uint32_t r = rects + i * 16; return {int32_t(kernel::R32(r)), int32_t(kernel::R32(r + 4)), int32_t(kernel::R32(r + 8)), int32_t(kernel::R32(r + 12))}; }
struct GuestPointI { int32_t x, y; };
GuestPointI ReadGuestPoint(uint32_t points, uint32_t i) { uint32_t p = points + i * 8; return {int32_t(kernel::R32(p)), int32_t(kernel::R32(p + 4))}; }
void VulkanBackend::Copy(const Surface& src, const Surface& dst, uint32_t count, uint32_t srcRects, uint32_t dstPoints) {
    VkPerfScope perfCopy{g_vkPerf.copy}; g_vkPerf.copies++;
    Target* srcT = FindTarget(src, false);
    if (!srcT || !srcT->drawn) {
        if (!CopySurfaceMemory(src, dst, count, srcRects, dstPoints)) LOG_FIRST(8, "vulkan: CopyRects %08X to %08X skipped", src.data, dst.data);
        else if (Target* dstT = FindTarget(dst, false)) UploadTarget(*dstT, dst);
        return;
    }
    Target* dstT = EnsureTarget(dst, false);
    if (!dstT || srcT == dstT) return;
    if (cap_) std::fprintf(cap_, "%04u copy %08X to %08X\n", capI_++, src.data, dst.data);
    bool listed = count && srcRects; uint32_t n = listed ? count : 1;
    // Host pixels per guest pixel across the whole (possibly widened) target;
    // a guest x lands at (x + dx) * scale inside a widened frame target.
    float srcScale = float(srcT->w) / (float(srcT->gw) + srcT->dx * 2), dstScale = float(dstT->w) / (float(dstT->gw) + dstT->dx * 2);
    if (srcT->w == dstT->w && srcT->h == dstT->h) {
        Transition(srcT->res, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL); Transition(dstT->res, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        for (uint32_t i = 0; i < n; i++) {
            VkImageCopy region{};
            region.srcSubresource.aspectMask = region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.srcSubresource.layerCount = region.dstSubresource.layerCount = 1;
            region.extent = {std::min(srcT->w, dstT->w), std::min(srcT->h, dstT->h), 1};
            if (listed) {
                GuestRectI r = ReadGuestRect(srcRects, i);
                region.srcOffset = {int32_t((r.left + srcT->dx) * srcScale), int32_t(uint32_t(r.top) * srcScale), 0};
                region.dstOffset = {int32_t((r.left + dstT->dx) * dstScale), int32_t(uint32_t(r.top) * dstScale), 0};
                if (dstPoints) { GuestPointI p = ReadGuestPoint(dstPoints, i); region.dstOffset = {int32_t((p.x + dstT->dx) * dstScale), int32_t(uint32_t(p.y) * dstScale), 0}; }
                region.extent.width = std::min(uint32_t(std::max<int32_t>(0, r.right - r.left)) * uint32_t(srcScale), srcT->w);
                region.extent.height = std::min(uint32_t(std::max<int32_t>(0, r.bottom - r.top)) * uint32_t(srcScale), srcT->h);
                if (!region.extent.width || !region.extent.height) continue;
                if (region.dstOffset.x < 0 || region.dstOffset.y < 0) continue;
                region.extent.width = std::min(region.extent.width, dstT->w - uint32_t(region.dstOffset.x));
                region.extent.height = std::min(region.extent.height, dstT->h - uint32_t(region.dstOffset.y));
            }
            vkCmdCopyImage(Cmd(), srcT->res.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstT->res.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        }
        Transition(dstT->res, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    } else {
        for (uint32_t i = 0; i < n; i++) {
            float x0 = 0, y0 = 0, x1 = float(std::min(srcT->gw, dstT->gw)), y1 = float(std::min(srcT->gh, dstT->gh)), dx = 0, dy = 0;
            if (listed) { GuestRectI r = ReadGuestRect(srcRects, i); x0 = float(r.left); y0 = float(r.top); x1 = float(r.right); y1 = float(r.bottom); dx = x0; dy = y0; if (dstPoints) { GuestPointI p = ReadGuestPoint(dstPoints, i); dx = float(p.x); dy = float(p.y); } }
            if (x1 <= x0 || y1 <= y0) continue;
            float srcWide = float(srcT->gw) + srcT->dx * 2;
            float uv[4] = {(x0 + srcT->dx) / srcWide, y0 / srcT->gh, (x1 - x0) / srcWide, (y1 - y0) / srcT->gh};
            std::array<float, 4> vp{(dx + dstT->dx) * dstScale, dy * dstScale, (x1 - x0) * dstScale, (y1 - y0) * dstScale};
            PrepareBlitSource(srcT->res, false);
            VkFramebuffer fb; BeginTargetPass(dstT, nullptr, fb);
            BlitToImage(kBgra, dstT->w, dstT->h, srcT->res, uv, false, true, uint32_t(std::ceil(std::max(srcScale / dstScale, 1.0f) - 1e-3f)), 1.0f, false, &vp);
        }
    }
    dstT->drawn = true; boundRt_ = boundDs_ = nullptr;
}
bool VulkanBackend::DepthCopyDraw(const State& s) {
    TextureDesc d;
    if (!s.rt.header || !s.textures[0] || !Describe(s.textures[0], d) || Format(d.format).depth || d.cube || d.depth != 1) return false;
    uint32_t gw, gh; SurfaceSize(s.rt, gw, gh);
    auto di = targets_.find(SizedTargetKey(mem::VirtualFromPhysical(s.rt.data), true, gw, gh));
    auto si = targets_.find(SizedTargetKey(d.data, true, d.width, d.height));
    if (di == targets_.end() || si == targets_.end() || di == si) return false;
    VkPipeline pipe = EnsureBlitPipeline(VK_FORMAT_UNDEFINED, false, true); if (!pipe) return false;
    Target& dst = di->second; Target& src = si->second;
    EnsureSetRoom();  // outside the pass and before the set's constants, as in Draw
    Transition(src.res, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    DrawTextures stages; stages.view[0] = src.res.SampleView();
    std::array<VkSampler, 4> samplers{nullSampler_, nullSampler_, nullSampler_, nullSampler_};
    Alloc cb = UploadAlloc(256, 256);
    VkDescriptorSet set = cb.cpu ? AllocDrawSet(cb.buffer, cb.off, cb.buffer, cb.off, stages, samplers) : VK_NULL_HANDLE;
    if (!set) { g_vkPerf.skipped++; LOG_FIRST(16, "vulkan: depth copy skipped, no descriptor set: rt %08X present %llu", s.rt.data, (unsigned long long)presents_); return true; }
    VkFramebuffer fb; BeginTargetPass(nullptr, &dst, fb);
    vkCmdBindPipeline(Cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdBindDescriptorSets(Cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_, 0, 1, &set, 0, nullptr);
    VkViewport vp{0, float(dst.h), float(dst.w), -float(dst.h), 0, 1}; vkCmdSetViewport(Cmd(), 0, 1, &vp);
    VkRect2D sc{{0, 0}, {dst.w, dst.h}}; vkCmdSetScissor(Cmd(), 0, 1, &sc);
    vkCmdDraw(Cmd(), 3, 1, 0, 0);
    InvalidateBinds();
    dst.drawn = true; boundRt_ = boundDs_ = nullptr; return true;
}
// ---- Present ----
void VulkanBackend::UploadGammaRamp() {
    rampDirty_ = false;
    Alloc a = UploadAlloc(256 * 4, 16); if (!a.cpu) return;
    bool identity = true;
    for (int i = 0; i < 256; i++) { uint8_t* p = a.cpu + i * 4; p[0] = ramp_[512 + i]; p[1] = ramp_[256 + i]; p[2] = ramp_[i]; p[3] = 255; identity &= ramp_[i] == i && ramp_[256 + i] == i && ramp_[512 + i] == i; }
    gamma_ = !identity;
    Transition(lut_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy region{}; region.bufferOffset = a.off; region.bufferRowLength = 256;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; region.imageSubresource.layerCount = 1; region.imageExtent = {256, 1, 1};
    vkCmdCopyBufferToImage(Cmd(), a.buffer, lut_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    Transition(lut_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}
// The aspect an unwidened title picture is shown at.
//
// "auto" is the output's own aspect, not 4:3. The title is already projecting
// widescreen whenever the host answers ExQueryNonVolatileSetting index 8 with
// the widescreen bit (kernel/xboxkrnl.cpp, VideoFlags() when app::Widescreen()),
// which is any output 3:2 or wider - so a 640x448 buffer on a 16:9 window holds
// a 16:9 picture and presenting it at 4:3 was pillarboxing a picture that
// already filled the frame. app::TargetAspect() is the same signal the widescreen
// flag is derived from, so present and projection read one source and cannot
// drift: 16:9 on a 1920x1080 window, the display's aspect on an ultrawide, 4:3
// on an output narrower than 3:2, where the title projects 4:3 too.
//
// An explicit aspect_ratio still wins, so 4:3 remains available for anyone who
// wants the pillarboxed console shape.
//
// auto presents the aspect the guest is ACTUALLY projecting, which is not the
// same as the window's shape whenever host widening is off. The title projects
// 16:9 on any output 3:2 or wider (the widescreen bit the host answers), and
// only the Hor+ path makes it project wider than that. Presenting a 16:9-
// projected frame at a 21:9 window would stretch it 1.34x - the artifact the
// Hor+ gate exists to avoid - so with widening off a wider display gets 16:9
// pillarboxed, and once widening is on it gets the widened aspect and fills.
float PresentSrcAspect() {
    const std::string& ar = app::Get().aspect_ratio;
    if (ar == "4:3") return 4.0f / 3.0f;
    if (ar == "16:9") return 16.0f / 9.0f;
    if (ar == "16:10") return 16.0f / 10.0f;
    if (ar == "21:9") return 21.0f / 9.0f;
    if (ar == "32:9") return 32.0f / 9.0f;
    float host = app::HostAspect();
    if (host > 0.0f) return host;                                  // widened: the frame really is this shape
    return app::Widescreen() ? 16.0f / 9.0f : 4.0f / 3.0f;         // what the title is projecting
}
void VulkanBackend::BlitFrame(Target& t, uint32_t w, uint32_t h) {
    // Fit the picture inside the window at its own aspect, centred: black
    // bars wherever the window is a different shape, never a crop and never
    // a stretch to the window's aspect. A widened frame target (dx != 0) is
    // presented whole at its own widened aspect; anything else at the chosen
    // aspect the title authored (PresentSrcAspect).
    float srcAspect = t.dx != 0 ? (float(t.gw) + t.dx * 2) / float(t.gh) : PresentSrcAspect(); float fw = float(w), fh = float(h);
    float vh = std::min(fh, fw / srcAspect), vw = vh * srcAspect;
    float v0 = 0, dv = 1;
    if (app::kGame.picture_height && guestH_) { v0 = float(app::kGame.picture_top) / float(guestH_); dv = float(app::kGame.picture_height) / float(guestH_); }
    float uv[4] = {0, v0, 1, dv};
    std::array<float, 4> vp{(fw - vw) / 2, (fh - vh) / 2, vw, vh};
    BlitToImage(swapFormat_, w, h, t.res, uv, gamma_, false, 1, 1.0f, false, &vp, true);
}
void VulkanBackend::RenderUi(VkCommandBuffer cmd) {
#ifdef XBE_VK_IMGUI
    if (!ImGui::GetCurrentContext()) return;
    if (!ImGui::GetIO().BackendRendererUserData) return;  // ImGui_ImplVulkan_Init has not run yet
    std::lock_guard lock(app::UiLock());
    ImGui_ImplVulkan_NewFrame();
#ifdef XBE_VK_IMGUI_SDL3
    ImGui_ImplSDL3_NewFrame();
#endif
    ImGui::NewFrame();
    if (app::kGame.ui) app::kGame.ui(fps_);
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
#else
    (void)cmd;
#endif
}
void VulkanBackend::LogPerf() {
    perfFrames_++;
    double el = VkPerfMs(perfT_);
    if (el < 1000.0) return;
    VkPerf& p = g_vkPerf;
    double n = double(perfFrames_);
    double host = p.draw + p.present + p.clear + p.copy;
    LOGI("perf: %.1f fps %.2f ms/frame | host %.2f (draw %.2f clear %.2f copy %.2f present %.2f) other %.2f",
         n * 1000.0 / el, el / n, host / n, p.draw / n, p.clear / n, p.copy / n, p.present / n, (el - host) / n);
    LOGI("perf: title thread per frame: recording %.2f ms, waiting %.2f ms (%.1f waits: read-back %.2f "
         "resource %.2f fence %.2f flush %.2f copy %.2f, next frame %.2f) | in XDK draws %.2f (of it sendstate %.2f) | read-backs waited for: %.1f lock, "
         "%.1f NOOVERWRITE",
         g_perf_capture_ns.exchange(0) / n / 1e6, g_perf_wait_ns.exchange(0) / n / 1e6,
         g_perf_waits.exchange(0) / n, g_perf_wait_kind_ns[0].exchange(0) / n / 1e6,
         g_perf_wait_kind_ns[1].exchange(0) / n / 1e6, g_perf_wait_kind_ns[2].exchange(0) / n / 1e6,
         g_perf_wait_kind_ns[3].exchange(0) / n / 1e6, g_perf_wait_kind_ns[4].exchange(0) / n / 1e6,
         g_perf_wait_kind_ns[5].exchange(0) / n / 1e6, g_perf_hle_ns.exchange(0) / n / 1e6,
         g_perf_sendstate_ns.exchange(0) / n / 1e6, g_perf_sync_block.exchange(0) / n,
         g_perf_sync_late.exchange(0) / n);
    LOGI("perf: per frame %.0f draws (%.0f raw) %.0f skipped %.0f nopipeline (shader %.0f pso %.0f set %.0f) stalealloc %llu %.0fk verts %.1f MB upload %.1f overflow | "
         "%.1f tex uploads %.2f MB, %.1f new tex, %.1f new targets, %.1f target uploads, %.1f read-backs | %llu shaders %llu psos this second | %.0f clears %.0f copies per frame",
         p.draws / n, p.raw / n, p.skipped / n, p.nopipeline / n, p.nopipeShader / n, p.nopipePso / n, p.nopipeSet / n, (unsigned long long)p.staleAlloc, p.verts / n / 1000.0, p.upload / n / 1048576.0, p.overflow / n,
         p.texups / n, p.texupBytes / n / 1048576.0, p.newtex / n, p.targets / n, p.rtups / n, p.syncs / n,
         (unsigned long long)p.shaders, (unsigned long long)p.psos, p.clears / n, p.copies / n);
    g_vkPerf = VkPerf{};
    perfT_ = VkPerfClock::now();
    perfFrames_ = 0;
}
void VulkanBackend::PeriodicCapture(Target* front) {
    static const uint32_t every = [] { const char* v = std::getenv("XBE_VK_CAPTURE_EVERY"); return v ? uint32_t(std::strtoul(v, nullptr, 10)) : 0u; }();
    if (!every || diag::CaptureDir().empty() || !front) return;
    if (presents_ % every != 0) return;
    std::error_code ec;
    std::filesystem::path dir = std::filesystem::weakly_canonical(app::ExeDir() / diag::CaptureDir(), ec);
    std::filesystem::create_directories(dir, ec);
    char name[64]; std::snprintf(name, sizeof(name), "front_%05llu.bmp", (unsigned long long)presents_);
    CapSaveColorAs(*front, dir / name);
    {   // the host-resolution target as well: the guest dump above shrinks the
        // (possibly widened) target to guest size and hides the widening geometry
        std::vector<uint8_t> px;
        if (ReadBackImage(front->res, front->w, front->h, px)) {
            char hname[64]; std::snprintf(hname, sizeof(hname), "front_%05llu_host.bmp", (unsigned long long)presents_);
            FILE* f = std::fopen((dir / hname).string().c_str(), "wb");
            if (f) {
                uint8_t hd[54] = {'B', 'M'}; auto put = [&](int o, uint32_t v) { std::memcpy(hd + o, &v, 4); };
                put(2, 54 + uint32_t(px.size())); put(10, 54); put(14, 40); put(18, front->w); put(22, uint32_t(-int32_t(front->h))); hd[26] = 1; hd[28] = 32; put(34, uint32_t(px.size()));
                std::fwrite(hd, 1, 54, f); std::fwrite(px.data(), 1, px.size(), f); std::fclose(f);
            }
        }
    }
    LOGI("vulkan: periodic front capture %s (present %llu, guest %ux%u host %ux%u drawn=%d)", (dir / name).string().c_str(), (unsigned long long)presents_, front->gw, front->gh, front->w, front->h, int(front->drawn));
}
void VulkanBackend::Present(const Surface& front) {
    VkPerfScope perfPresent{g_vkPerf.present};
    if (cap_) {
        CapStep(Surface{});
        if (Target* f = FindTarget(front, false)) CapSaveColor(*f, "final");
        std::fclose(cap_); cap_ = nullptr;
        LOGI("vulkan: frame captured to %s", capDir_.string().c_str());
    } else if (g_capture.exchange(false) && !diag::CaptureDir().empty()) {
        std::error_code ec; auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        char stamp[32]; std::strftime(stamp, sizeof(stamp), "%H%M%S", std::localtime(&t));
        capDir_ = std::filesystem::weakly_canonical(app::ExeDir() / diag::CaptureDir() / stamp, ec);
        std::filesystem::create_directories(capDir_, ec);
        cap_ = std::fopen((capDir_ / "frame.txt").string().c_str(), "w"); capI_ = capRt_ = 0;
    }
    ResizeSwapIfNeeded();
    Target* t = EnsureTarget(front, false);
    if (t && LatchCpuWrite(*t)) { UploadTarget(*t, front); g_vkPerf.rtups++; }
    PeriodicCapture(t);
    if (rampDirty_) UploadGammaRamp();
    uint32_t imageIndex = 0; VkSemaphore waitSem = VK_NULL_HANDLE; bool haveSwap = swapchain_ != VK_NULL_HANDLE;
    if (haveSwap) {
        VkResult r = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, Cur().imageAvailable, VK_NULL_HANDLE, &imageIndex);
        if (r == VK_ERROR_OUT_OF_DATE_KHR) { ResizeSwapIfNeeded(true); if (swapchain_) r = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, Cur().imageAvailable, VK_NULL_HANDLE, &imageIndex); }
        if (r == VK_SUBOPTIMAL_KHR) swapStale_ = true;  // still usable this frame; recreate at the boundary below
        if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
            LOGW("vulkan: vkAcquireNextImageKHR failed %d", int(r));
            // D1: the frame's draws/uploads are already recorded into this slot's open
            // segment. Submit it (waited) while frame_ still names that slot, so the
            // segment is closed, its garbage retires, and its fence state stays
            // coherent, instead of advancing frame_ and leaving the next frame to
            // record into a never-begun command buffer and reset a pending fence.
            // This is also the routine VK_ERROR_OUT_OF_DATE_KHR-after-resize path;
            // the acquire failed, so its semaphore was not signalled and this submit
            // must not wait on it.
            if (recording_) Submit(true);
            presents_++; presentsSeen_.store(presents_); frame_++; if (diag::Perf()) LogPerf(); return;
        }
        if (imageIndex >= swapFbs_.size()) {
            LOGE("vulkan: acquire returned image %u of %zu", imageIndex, swapFbs_.size());
            // D1: same desync as above, except this acquire did signal
            // Cur().imageAvailable: the closing submit must wait on (consume) it, or
            // the next acquire of this slot re-signals a binary semaphore whose
            // signal is still pending. EnsureRecording opens an (empty) segment
            // when none is open so there is a submission to consume it with.
            if (!recording_) EnsureRecording();
            Submit(true, Cur().imageAvailable);
            presents_++; presentsSeen_.store(presents_); frame_++; return;
        }
        waitSem = Cur().imageAvailable;
    }
    if (t) PrepareBlitSource(t->res, gamma_);
    EnsureRecording();
    VkFramebuffer fb = haveSwap ? swapFbs_[imageIndex] : offscreenFb_;
    VkRenderPassBeginInfo bi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    bi.renderPass = swapPass_; bi.framebuffer = fb; bi.renderArea = {{0, 0}, {bbW_, bbH_}};
    VkClearValue clear{}; clear.color.float32[3] = 1; bi.clearValueCount = 1; bi.pClearValues = &clear;
    vkCmdBeginRenderPass(Cmd(), &bi, VK_SUBPASS_CONTENTS_INLINE);
    if (t) BlitFrame(*t, bbW_, bbH_);
    else LOG_FIRST(8, "vulkan: present with no guest front target, clearing swap image %u only", imageIndex);
    RenderUi(Cmd());
    vkCmdEndRenderPass(Cmd());
    Submit(false, waitSem, haveSwap ? Cur().renderFinished : VK_NULL_HANDLE);
    if (haveSwap) {
        VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        pi.waitSemaphoreCount = 1; pi.pWaitSemaphores = &Cur().renderFinished; pi.swapchainCount = 1; pi.pSwapchains = &swapchain_; pi.pImageIndices = &imageIndex;
        VkResult r = vkQueuePresentKHR(queue_, &pi);
        if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) swapStale_ = true;
        else if (r != VK_SUCCESS) LOGW("vulkan: vkQueuePresentKHR failed %d (image %u)", int(r), imageIndex);
    }
    if (swapStale_) { swapStale_ = false; ResizeSwapIfNeeded(true); }
    if (diag::Perf() && t && presents_ % 60 == 0) {
        float aspect = t->dx != 0 ? (float(t->gw) + t->dx * 2) / float(t->gh) : PresentSrcAspect(); float vh = std::min(float(bbH_), float(bbW_) / aspect), vw = vh * aspect;
        float v0 = app::kGame.picture_height && guestH_ ? float(app::kGame.picture_top) / float(guestH_) : 0.0f;
        float dv = app::kGame.picture_height && guestH_ ? float(app::kGame.picture_height) / float(guestH_) : 1.0f;
        float sx = 0, sy = v0 * float(t->h), sw = float(t->w), sh = dv * float(t->h);
        LOGI("vulkan: present %llu image %u front %08X guest %ux%u target %ux%u scale %.3f widen %.3f targetAspect %.3f srcAspect %.3f window %ux%u srcPx (%.0f,%.0f %.0fx%.0f) srcUv (0,%.3f)-(1,%.3f) dst (%.0f,%.0f %.0fx%.0f) gamma=%d",
             (unsigned long long)presents_, imageIndex, t->data, t->gw, t->gh, t->w, t->h, scale_, widen_, app::TargetAspect(), aspect, bbW_, bbH_, sx, sy, sw, sh, v0, dv, (float(bbW_) - vw) / 2, (float(bbH_) - vh) / 2, vw, vh, int(gamma_));
    }
    presents_++; presentsSeen_.store(presents_); frame_++;
    ReleaseOldTextures();
    { Frame& f = Cur(); if (f.fencePending) WaitFrame(f); }
    CountFps(); ApplySettings();
    if (diag::Perf()) LogPerf();
    if (pipelineCache_ && presents_ % 300 == 0) SavePipelineCache();
}
void VulkanBackend::SavePipelineCache() {
    size_t bytes = 0;
    if (vkGetPipelineCacheData(device_, pipelineCache_, &bytes, nullptr) != VK_SUCCESS || !bytes) return;
    std::vector<uint8_t> blob(bytes);
    if (vkGetPipelineCacheData(device_, pipelineCache_, &bytes, blob.data()) != VK_SUCCESS) return;
    std::error_code ec; std::filesystem::create_directories(pipelineCacheFile_.parent_path(), ec);
    std::ofstream f(pipelineCacheFile_, std::ios::binary | std::ios::trunc);
    if (f) f.write(reinterpret_cast<const char*>(blob.data()), std::streamsize(bytes));
}
void VulkanBackend::ReleaseOldTextures() {
    if (presents_ % 60 != 0 || textures_.size() <= 2048) return;
    for (auto it = textures_.begin(); it != textures_.end();)
        if (it->second.frame + 600 < presents_) { DeferDestroyImage(it->second.res); it = textures_.erase(it); } else ++it;
}
void VulkanBackend::CountFps() {
    fpsN_++; auto now = std::chrono::steady_clock::now();
    double el = std::chrono::duration<double>(now - fpsT_).count();
    if (el >= 0.5) { fps_ = fpsN_ / el; fpsN_ = 0; fpsT_ = now; }
}
void VulkanBackend::ApplySettings() {
    float want = app::RenderScale(guestH_); float wide = app::AspectWiden(); float aspect = app::HostAspect();
    if (std::fabs(want - scale_) > 1e-4f || std::fabs(wide - widen_) > 1e-4f || std::fabs(aspect - hostAspect_) > 1e-4f) {
        Flush();
        for (auto& [key, tg] : targets_) { DeferDestroyImage(tg.res); DeferDestroyImage(tg.fb); }
        targets_.clear(); scale_ = want; widen_ = wide; hostAspect_ = aspect; boundRt_ = boundDs_ = nullptr;
    }
}
void VulkanBackend::CapSaveColorAs(Target& t, const std::filesystem::path& file) {
    ImageRes src = PrepareCopySource(t, nullptr);
    std::vector<uint8_t> px; if (!ReadBackImage(src, t.gw, t.gh, px)) return;
    if (src.image != t.res.image) DestroyImage(src);
    FILE* f = std::fopen(file.string().c_str(), "wb"); if (!f) return;
    uint8_t hd[54] = {'B', 'M'}; auto put = [&](int o, uint32_t v) { std::memcpy(hd + o, &v, 4); };
    put(2, 54 + uint32_t(px.size())); put(10, 54); put(14, 40); put(18, t.gw); put(22, uint32_t(-int32_t(t.gh))); hd[26] = 1; hd[28] = 32; put(34, uint32_t(px.size()));
    std::fwrite(hd, 1, 54, f); std::fwrite(px.data(), 1, px.size(), f); std::fclose(f);
}
void VulkanBackend::CapSaveColor(Target& t, const char* name) { CapSaveColorAs(t, capDir_ / (std::string(name) + ".bmp")); }
void VulkanBackend::CapStep(const Surface& rt) {
    uint32_t d = rt.header ? mem::VirtualFromPhysical(rt.data) : 0;
    if (d == capRt_) return;
    auto it = targets_.find(TargetMapKey(capRt_, false));
    if (capRt_ && it != targets_.end()) { char name[32]; std::snprintf(name, sizeof(name), "%04u_%08X", capI_, capRt_); CapSaveColor(it->second, name); }
    capRt_ = d;
}
// ---- Sync / late read-back ----
void VulkanBackend::Sync(uint32_t data, uint32_t size) {
    g_vkPerf.syncs++;
    if (LowPhysical(data)) { LOG_FIRST(8, "vulkan: read-back to low physical %08X refused", data); return; }
    auto it = targets_.find(TargetMapKey(data, false));
    if (it == targets_.end() || !it->second.drawn || TargetStale(data)) return;
    if (!ReadBackFits(data, it->second.gw, it->second.gh, size)) return;
    Target& t = it->second;
    ImageRes src = PrepareCopySource(t, nullptr); if (!src.Valid()) return;
    std::vector<uint8_t> px; bool ok = ReadBackImage(src, t.gw, t.gh, px);
    if (src.image != t.res.image) DestroyImage(src);
    if (!ok) return;
    uint32_t pitch = GuestPitch(t.gw); uint8_t* guest = static_cast<uint8_t*>(kernel::HostPtr(data));
    for (uint32_t y = 0; y < t.gh; y++) std::memcpy(guest + size_t(y) * pitch, &px[size_t(y) * t.gw * 4], t.gw * 4);
    t.read = std::move(px); t.drawn = false; t.sig = MemSig(t.data, t.bytes);
}
void VulkanBackend::SyncLate(uint32_t data, uint32_t size) { if (!DeliverLate(data, size)) Sync(data, size); RecordLate(data); }
bool VulkanBackend::DeliverLate(uint32_t data, uint32_t size) {
    std::lock_guard lock(lateLock_);
    if (LowPhysical(data)) return false;
    auto it = late_.find(data); if (it == late_.end()) return false;
    Late& l = it->second;
    if (!ReadBackFits(data, l.gw, l.gh, size)) return false;
    uint64_t now = presentsSeen_.load(); int best = -1;
    // D2: this runs on the title thread while Submit runs on the render thread. It
    // touches no shared container: l is under lateLock_, and "finished" is the atomic
    // submitCounter_ (published by Submit after vkQueueSubmit) plus the VkFence this
    // slot resolved at record time. The old submitFences_ unordered_map, find()ed
    // here while Submit inserted/erased it, is gone.
    uint64_t submitted = submitCounter_.load(std::memory_order_acquire);
    for (int i = 0; i < kFrames; i++) {
        if (!l.fence[i] || !l.submitFence[i]) continue;
        // D3: a stamp ahead of the submitted counter was recorded into a segment
        // that has not been submitted yet: it is not finished, however the buffer
        // looks, and it must not be marked delivered below.
        bool finished = l.fence[i] <= submitted && vkGetFenceStatus(device_, l.submitFence[i]) == VK_SUCCESS;
        bool recent = now - l.frame[i] <= uint64_t(kFrames) + 1;
        bool newest = best < 0 || l.fence[i] > l.fence[best];
        if (finished && recent && newest) best = i;
    }
    if (best < 0) return false;
    if (l.fence[best] > l.delivered && l.rb[best]) {
        void* p = nullptr; if (vkMapMemory(device_, l.rbm[best], 0, VK_WHOLE_SIZE, 0, &p) != VK_SUCCESS || !p) return false;
        if (!l.rbCoherent[best]) {  // D6: invalidate HOST_VISIBLE-only memory before reading it
            VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            range.memory = l.rbm[best]; range.offset = 0; range.size = VK_WHOLE_SIZE;
            if (vkInvalidateMappedMemoryRanges(device_, 1, &range) != VK_SUCCESS) { vkUnmapMemory(device_, l.rbm[best]); return false; }
        }
        uint32_t pitch = GuestPitch(l.gw); uint8_t* dst = static_cast<uint8_t*>(kernel::HostPtr(data)); const uint8_t* sp = static_cast<const uint8_t*>(p);
        for (uint32_t y = 0; y < l.gh; y++) std::memcpy(dst + size_t(y) * pitch, sp + size_t(y) * l.gw * 4, l.gw * 4);
        vkUnmapMemory(device_, l.rbm[best]); l.delivered = l.fence[best];
    }
    return true;
}
void VulkanBackend::RecordLate(uint32_t data) {
    auto it = targets_.find(TargetMapKey(data, false));
    if (it == targets_.end() || TargetStale(data)) return;
    Target& t = it->second; t.sig = MemSig(t.data, t.bytes); t.read.clear();
    std::lock_guard lock(lateLock_);
    Late& l = late_[data];
    if (l.gw != t.gw || l.gh != t.gh) {
        for (int i = 0; i < kFrames; i++) if (l.rb[i]) { VkBuffer rb = l.rb[i]; VkDeviceMemory rbm = l.rbm[i]; Cur().garbage.push_back([this, rb, rbm] { vkDestroyBuffer(device_, rb, nullptr); vkFreeMemory(device_, rbm, nullptr); }); }
        if (l.down.Valid()) DeferDestroyImage(l.down);
        l = Late{}; l.gw = t.gw; l.gh = t.gh;
    }
    uint32_t slot = l.next;
    if (l.fence[slot]) {
        // D3: reuse only a slot whose copy is known finished: a stamp ahead of the
        // submitted counter is still sitting in an unsubmitted segment, and a
        // submitted stamp needs its resolved fence signalled. Missing state never
        // counts as finished.
        if (l.fence[slot] > submitCounter_.load(std::memory_order_acquire)) return;
        if (l.submitFence[slot] && vkGetFenceStatus(device_, l.submitFence[slot]) != VK_SUCCESS) return;
    }
    ImageRes src = PrepareCopySource(t, &l.down); if (!src.Valid()) return;
    if (!l.rb[slot]) {
        VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; ci.size = uint64_t(l.gw) * l.gh * 4; ci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (vkCreateBuffer(device_, &ci, nullptr, &l.rb[slot]) != VK_SUCCESS) return;
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(device_, l.rb[slot], &mr);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = mr.size;
        // D6: sentinel/allocate/bind failures destroy the partial buffer and record
        // nothing, instead of binding through a guessed type 0.
        ai.memoryTypeIndex = FindHostMemory(mr.memoryTypeBits, l.rbCoherent[slot]);
        if (ai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(device_, &ai, nullptr, &l.rbm[slot]) != VK_SUCCESS) { vkDestroyBuffer(device_, l.rb[slot], nullptr); l.rb[slot] = VK_NULL_HANDLE; return; }
        if (vkBindBufferMemory(device_, l.rb[slot], l.rbm[slot], 0) != VK_SUCCESS) { vkDestroyBuffer(device_, l.rb[slot], nullptr); l.rb[slot] = VK_NULL_HANDLE; vkFreeMemory(device_, l.rbm[slot], nullptr); l.rbm[slot] = VK_NULL_HANDLE; return; }
    }
    Transition(src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    VkBufferImageCopy region{}; region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; region.imageSubresource.layerCount = 1; region.imageExtent = {l.gw, l.gh, 1};
    vkCmdCopyImageToBuffer(Cmd(), src.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, l.rb[slot], 1, &region);
    // D2/D3: stamp the slot recorded-but-unsubmitted, with the fence the next Submit
    // of this segment will signal resolved now, on the render thread.
    l.fence[slot] = submitCounter_.load(std::memory_order_relaxed) + 1; l.submitFence[slot] = Cur().fence; l.frame[slot] = presents_; l.next = (slot + 1) % kFrames;
}
void VulkanBackend::SetGamma(const uint8_t* ramp) {
    std::memcpy(ramp_, ramp, sizeof(ramp_));
    bool black = true; for (uint8_t v : ramp_) if (v != 0) { black = false; break; }
    if (black) { LOG_FIRST(4, "vulkan: all black gamma ramp ignored, the title blanks it for the overlay"); for (int i = 0; i < 256; i++) ramp_[i] = ramp_[256 + i] = ramp_[512 + i] = uint8_t(i); }
    {
        static int lastMean = -1, lastMax = -1, logged = 0;
        long sum = 0; int mx = 0; for (uint8_t v : ramp_) { sum += v; mx = std::max(mx, int(v)); }
        int mean = int(sum / 768);
        if (mean != lastMean || mx != lastMax || logged < 4) {
            lastMean = mean; lastMax = mx; logged++;
            LOGI("vulkan: gamma ramp mean=%d max=%d R[0]=%u R[255]=%u G[0]=%u G[255]=%u B[0]=%u B[255]=%u (after all-black handling)", mean, mx, ramp_[0], ramp_[255], ramp_[256], ramp_[511], ramp_[512], ramp_[767]);
        }
    }
    rampDirty_ = true;
}
void VulkanBackend::Destroy() {
    if (destroyed_ || !device_) return;
    destroyed_ = true; vkDeviceWaitIdle(device_);
    for (auto& f : frames_) RunGarbage(f);
    for (auto& [k, t] : targets_) { DestroyImage(t.res); DestroyImage(t.fb); }
    targets_.clear();
    for (auto& [k, tx] : textures_) DestroyImage(tx.res);
    textures_.clear();
    for (auto& [k, l] : late_) { DestroyImage(l.down); for (int i = 0; i < kFrames; i++) if (l.rb[i]) { vkDestroyBuffer(device_, l.rb[i], nullptr); vkFreeMemory(device_, l.rbm[i], nullptr); } }
    DestroyImage(lut_); DestroyImage(nullImage_);
    if (offscreenFb_) vkDestroyFramebuffer(device_, offscreenFb_, nullptr);
    DestroyImage(offscreen_); DestroySwapchain();
    if (swapPass_ && surface_) vkDestroyRenderPass(device_, swapPass_, nullptr);  // headless: swapPass_ is in passes_
    for (auto& [k, rp] : passes_) vkDestroyRenderPass(device_, rp, nullptr);
    for (auto& [k, p] : pipelines_) if (p) vkDestroyPipeline(device_, p, nullptr);
    for (auto& [k, p] : blitPipelines_) if (p) vkDestroyPipeline(device_, p, nullptr);
    for (auto& [k, m] : modules_) if (m) vkDestroyShaderModule(device_, m, nullptr);
    for (auto& [k, s] : samplers_) if (s) vkDestroySampler(device_, s, nullptr);
    if (nullSampler_) vkDestroySampler(device_, nullSampler_, nullptr);
    if (pipelineCache_) { SavePipelineCache(); vkDestroyPipelineCache(device_, pipelineCache_, nullptr); }
    if (imguiPool_) vkDestroyDescriptorPool(device_, imguiPool_, nullptr);
    for (auto& f : frames_) {
        vkDestroyDescriptorPool(device_, f.pool, nullptr); vkUnmapMemory(device_, f.uploadMem);
        vkDestroyBuffer(device_, f.upload, nullptr); vkFreeMemory(device_, f.uploadMem, nullptr);
        vkDestroySemaphore(device_, f.imageAvailable, nullptr); vkDestroySemaphore(device_, f.renderFinished, nullptr); vkDestroyFence(device_, f.fence, nullptr);
    }
    vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr); vkDestroyDescriptorSetLayout(device_, setLayout_, nullptr);
    vkDestroyCommandPool(device_, pool_, nullptr); vkDestroyDevice(device_, nullptr); device_ = VK_NULL_HANDLE;
    if (surface_) { vkDestroySurfaceKHR(instance_, surface_, nullptr); surface_ = VK_NULL_HANDLE; std::lock_guard lock(g_handover_lock); g_surface = VK_NULL_HANDLE; }
    if (instance_) { DestroyVkMessenger(); vkDestroyInstance(instance_, nullptr); instance_ = VK_NULL_HANDLE; std::lock_guard lock(g_handover_lock); g_instance = VK_NULL_HANDLE; g_handles = VulkanHandles{}; }
}
}  // namespace
std::unique_ptr<Backend> CreateBackend() { return std::make_unique<VulkanBackend>(); }
}  // namespace gpu
#endif  // !defined(_WIN32)
