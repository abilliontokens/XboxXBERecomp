// Direct3D 8 (XDK 5849) entry points the title calls, implemented on the host renderer.
// Resource creation, locking and the XG helpers stay recompiled: they only manage memory.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <map>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include "app/app.h"
#if !defined(_WIN32)
#include "app/platform.h"  // app::HostAspect (Linux Hor+ widening)
#endif
#include "core/diag.h"
#include "gpu/backend.h"
#include "gpu/texture.h"
#include "gpu/vertex.h"
#include "xdk/d3d8.h"
#include "xdk/hle.h"
#include "xdk/layout.h"

using namespace kernel;

// configs written before the SDK knew the name
#ifndef SYM_D3DDevice_BeginPush
#define SYM_D3DDevice_BeginPush FFFFFE00
#define HAS_D3DDevice_BeginPush 0
#endif
#ifndef SYM_D3DDevice_EndPush
#define SYM_D3DDevice_EndPush FFFFFE01
#define HAS_D3DDevice_EndPush 0
#endif
#ifndef HAS_D3D_FvfShader
#define HAS_D3D_FvfShader 0
#endif
#ifndef SYM_D3DDevice_SetPalette
#define SYM_D3DDevice_SetPalette FFFFFE02
#define HAS_D3DDevice_SetPalette 0
#endif
#ifndef SYM_D3DDevice_GetVertexShaderFunction
#define SYM_D3DDevice_GetVertexShaderFunction FFFFFE03
#define HAS_D3DDevice_GetVertexShaderFunction 0
#endif
#ifndef SYM_D3DDevice_Reset
#define SYM_D3DDevice_Reset FFFFFE04
#define HAS_D3DDevice_Reset 0
#endif
// XDK 3925 names, which no later build's config has
#ifndef SYM_D3DDevice_Present
#define SYM_D3DDevice_Present FFFFFE05
#define HAS_D3DDevice_Present 0
#endif
#ifndef SYM_D3DDevice_GetBackBuffer
#define SYM_D3DDevice_GetBackBuffer FFFFFE06
#define HAS_D3DDevice_GetBackBuffer 0
#endif
#ifndef SYM_D3DDevice_GetDepthStencilSurface
#define SYM_D3DDevice_GetDepthStencilSurface FFFFFE07
#define HAS_D3DDevice_GetDepthStencilSurface 0
#endif

namespace d3d {

namespace {

// A/B switch: 0 draws on the title's thread, as before
#ifndef XBE_RENDER_THREAD
#define XBE_RENDER_THREAD 1
#endif
constexpr bool kRenderThread = XBE_RENDER_THREAD;

// perf diagnostic: title time inside one XDK draw call, into gpu::g_perf_hle_ns; the clock is read only with XBE_PERF
struct HleScope {
    bool on = diag::Perf();
    std::chrono::steady_clock::time_point t =
        on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    ~HleScope() {
        if (on)
            gpu::g_perf_hle_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                               std::chrono::steady_clock::now() - t).count());
    }
};

// D3D globals of this build
constexpr uint32_t kDevice = SYM(D3D_Device);        // the static CDevice, g_pDevice points at it
constexpr uint32_t kDevicePtr = SYM(D3D_pDevice);
constexpr uint32_t kIndexData = SYM(D3D_IndexData);  // index data base the inline draws read
constexpr uint32_t kDirtyFlags = SYM(D3D_DirtyFlags);
constexpr uint32_t kTextureState = SYM(D3D_DeferredTextureState);  // [4][32], by title:: type
constexpr uint32_t kRenderState = SYM(D3D_RenderState);            // [title::kRsMax], by title:: index
constexpr uint32_t kStreams = SYM(D3D_Streams);                    // [16] stride, offset, buffer
constexpr uint32_t kSimpleEncode = SYM(D3D_SimpleRenderStateEncode);

// CDevice fields mirrored so the recompiled getters and resource code see what the XDK would
constexpr uint32_t kDevPush = XDK_CDevice_Push, kDevThreshold = XDK_CDevice_PushThreshold,
                   kDevBaseVertex = XDK_CDevice_BaseVertex, kDevIndexBuffer = XDK_CDevice_IndexBuffer,
                   kDevTextures = XDK_CDevice_Textures, kDevRenderTarget = XDK_CDevice_RenderTarget,
                   kDevDepth = XDK_CDevice_Depth, kDevSwapCount = XDK_CDevice_SwapCount,
                   kDevBackBuffer = XDK_CDevice_BackBuffer, kDevFrontBuffer = XDK_CDevice_FrontBuffer,
                   kDevZScale = XDK_CDevice_ZScale, kDevSuperX = XDK_CDevice_SuperSampleX,
                   kDevSuperY = XDK_CDevice_SuperSampleY, kDevTransforms = XDK_CDevice_Transforms,
                   kDevPalettes = XDK_CDevice_Palettes;
constexpr uint32_t kStageStates = XDK_TextureState_Stage, kStreamSize = XDK_Stream_Size;

// The title's D3D_RenderState and D3D_DeferredTextureState are indexed by its build's numbering of the states, g_state
// and the renderer by the host's (gpu::kRs*, gpu::kTss*: XDK 5849's), and a build's config may number them otherwise
// (xdk/layout.h). Whatever reads or writes those two arrays maps between the numberings here, and a state one side has
// no number for is never written on the other. With 5849's numbering every map is the identity.
namespace title {
// D3D_RenderState: simple states below kRsDeferredFirst, deferred ones below kRsComplexFirst, complex ones below
// kRsMax; D3D_DeferredTextureState: deferred types below kTssDeferredMax
constexpr uint32_t kRsDeferredFirst = XDK_RENDER_STATE(DeferredFirst),
                   kRsComplexFirst = XDK_RENDER_STATE(ComplexFirst), kRsMax = XDK_RENDER_STATE(Max),
                   kTssDeferredMax = XDK_TEXTURE_STATE(DeferredMax);
}  // namespace title
static_assert(title::kRsDeferredFirst <= title::kRsComplexFirst && title::kRsComplexFirst <= title::kRsMax &&
                  title::kTssDeferredMax <= gpu::kTssCount,
              "the config's RenderState_ or TextureState_ bounds are out of order");

// a state's number in the host's numbering and in the build's, xdk::kAbsent when the build has no such state
struct StatePair {
    uint32_t host, title;
};
constexpr StatePair kRsPairs[] = {
#define RS_PAIR(name) {gpu::kRs##name, XDK_RENDER_STATE(name)},
    GPU_RENDER_STATES(RS_PAIR)
#undef RS_PAIR
};
constexpr StatePair kTssPairs[] = {
#define TSS_PAIR(name) {gpu::kTss##name, XDK_TEXTURE_STATE(name)},
    GPU_TEXTURE_STAGE_STATES(TSS_PAIR)
#undef TSS_PAIR
};

// The build's number of each host one (title) and the host's of each of the build's (host), xdk::kAbsent where the
// other side has none. A build with the host's bounds that numbers every named state as the host does gets the
// identity, the unnamed slots between the named ones included.
template <size_t HostN, size_t TitleN>
struct StateMap {
    std::array<uint32_t, HostN> title{};
    std::array<uint32_t, TitleN> host{};
    bool valid = true;  // no number of the build's past its table or given twice
};
template <size_t HostN, size_t TitleN, size_t N>
constexpr StateMap<HostN, TitleN> MakeStateMap(const StatePair (&pairs)[N], bool same_bounds) {
    StateMap<HostN, TitleN> m;
    m.title.fill(xdk::kAbsent);
    m.host.fill(xdk::kAbsent);
    bool same = same_bounds;
    for (const StatePair& p : pairs) same = same && p.title == p.host;
    if (same) {
        for (uint32_t i = 0; i < std::min(HostN, TitleN); i++) m.title[i] = m.host[i] = i;
        return m;
    }
    for (const StatePair& p : pairs) {
        if (p.title == xdk::kAbsent) continue;
        if (p.title >= TitleN || m.host[p.title] != xdk::kAbsent) {
            m.valid = false;
            continue;
        }
        m.title[p.host] = p.title;
        m.host[p.title] = p.host;
    }
    return m;
}
constexpr auto kRsMap = MakeStateMap<gpu::kRsMax, title::kRsMax>(
    kRsPairs, title::kRsDeferredFirst == gpu::kRsDeferredFirst && title::kRsComplexFirst == gpu::kRsComplexFirst &&
                  title::kRsMax == gpu::kRsMax);
constexpr auto kTssMap =
    MakeStateMap<gpu::kTssCount, gpu::kTssCount>(kTssPairs, title::kTssDeferredMax == gpu::kTssDeferredMax);
static_assert(kRsMap.valid, "a RenderState_ number in the config is not below RenderState_Max, or is given twice");
static_assert(kTssMap.valid, "a TextureState_ number in the config is not below 32, or is given twice");

// The slots of a block of the build's from first that have a host number, in runs consecutive in both numberings,
// which copy whole: with the host's numbering the block is one run.
struct Run {
    uint32_t title, host, count;
};
template <size_t N>
struct Runs {
    std::array<Run, N> run{};
    size_t n = 0;
    constexpr const Run* begin() const { return run.data(); }
    constexpr const Run* end() const { return run.data() + n; }
};
template <size_t N, size_t TitleN>
constexpr Runs<N> MakeRuns(const std::array<uint32_t, TitleN>& host, uint32_t first) {
    Runs<N> r;
    for (uint32_t t = first; t < first + N; t++) {
        uint32_t h = host[t];
        if (h == xdk::kAbsent) continue;
        if (r.n) {
            Run& last = r.run[r.n - 1];
            if (last.title + last.count == t && last.host + last.count == h) {
                last.count++;
                continue;
            }
        }
        r.run[r.n++] = {t, h, 1};
    }
    return r;
}
// the deferred blocks, which the title's inline SetRenderState and SetTextureStageState write
constexpr auto kRsDeferred =
    MakeRuns<title::kRsComplexFirst - title::kRsDeferredFirst>(kRsMap.host, title::kRsDeferredFirst);
constexpr auto kTssDeferred = MakeRuns<title::kTssDeferredMax>(kTssMap.host, 0);

// The title's state. g_state is const here: every write goes through Set, Load or Refresh, which mark the 64 byte
// blocks they write (gpu::State::Mark). The render thread's recorder compares only marked blocks with what it sent,
// so a write that did not mark would never reach the renderer.
gpu::State g_state_data;
const gpu::State& g_state = g_state_data;

// field = v, marked
template <typename T>
void Set(const T& field, const std::type_identity_t<T>& v) {
    const_cast<T&>(field) = v;
    g_state.Mark(&field, sizeof(T));
}

// the field's bytes from the title's memory at va, marked
template <typename T>
void Load(const T& field, uint32_t va) {
    static_assert(std::is_trivially_copyable_v<T>);
    std::memcpy(const_cast<T*>(&field), HostPtr(va), sizeof(T));
    g_state.Mark(&field, sizeof(T));
}

// n bytes of the title's memory at va over g_state's bytes at p, marked when they differ
void Refresh(const void* p, uint32_t va, size_t n) {
    if (!std::memcmp(p, HostPtr(va), n)) return;
    std::memcpy(const_cast<void*>(p), HostPtr(va), n);
    g_state.Mark(p, n);
}

std::unique_ptr<gpu::Backend> g_backend;
bool g_frame_widened = true;  // gpu::Backend::WidensFrame of the backend CreateBackend made, before any wrapper
std::recursive_mutex g_lock;

std::unordered_map<uint32_t, std::shared_ptr<gpu::VertexShader>> g_vs;
std::unordered_map<uint32_t, std::shared_ptr<gpu::PixelShader>> g_ps;
std::array<uint32_t, 3> g_buffers{};  // front, back surfaces
uint32_t g_buffer_count = 2;
uint32_t g_depth = 0;
uint32_t g_frame_w = 640, g_frame_h = 480;  // the console's frame, what the title asked CreateDevice for
uint32_t g_scratch_push = 0, g_scratch_size = 0;
uint32_t g_walk = 0;  // first scratch command not read yet
// The NV2A's own vertex-constant load register. kVscPush sets it and every constant packet advances
// it, exactly as the hardware does, because an upload longer than the 32 dwords one packet carries
// arrives as several packets and only the first is preceded by its kVscPush header.
uint32_t g_vsc_reg = 0;

uint32_t Dev(uint32_t off) { return R32(kDevice + off); }
void SetDev(uint32_t off, uint32_t v) { W32(kDevice + off, v); }

// The device's own references: a count in bits 19 to 22 of the resource's first dword. A surface's first one
// also holds its parent. The library's functions do it when the image has them, inlined ones are done here.
void AddRef(X86Context& ctx, uint32_t res) {
    if (!res) return;
#if HAS_D3D_InternalAddRef
    GUEST_CALL(D3D_InternalAddRef, ctx, {res});
#else
    (void)ctx;
    uint32_t common = R32(res);
    if (!(common & 0x780000) && (common & 0x70000) == 0x50000)
        if (uint32_t parent = R32(res + 0x14)) W32(parent, R32(parent) + 0x80000);
    W32(res, R32(res) + 0x80000);
#endif
}
void Release(X86Context& ctx, uint32_t res) {
    if (!res) return;
#if HAS_D3D_InternalRelease
    GUEST_CALL(D3D_InternalRelease, ctx, {res});
#else
    (void)ctx;
    // the library frees a resource nothing holds any more; without its function the memory stays
    auto drop = [](uint32_t r) {
        W32(r, R32(r) - 0x80000);
        if (!(R32(r) & 0x78FFFF)) LOG_FIRST(8, "d3d: resource %08X is not freed, the image has no D3D_InternalRelease", r);
    };
    uint32_t common = R32(res);
    if ((common & 0x780000) == 0x80000 && (common & 0x70000) == 0x50000)
        if (uint32_t parent = R32(res + 0x14)) drop(parent);
    drop(res);
#endif
}

void Walk(uint32_t end);

// Hardware commands a caller still pushes go to a scratch buffer, rewound at every request once Walk has read it
uint32_t ScratchPush(uint32_t bytes = 0) {
    Walk(Dev(kDevPush));
    uint32_t need = std::max<uint32_t>(0x10000, (bytes + 0x1FFF) & ~0xFFFu);
    if (need > g_scratch_size) {
        if (g_scratch_push) mem::FreeContiguous(g_scratch_push);
        g_scratch_push = mem::AllocContiguous(need, 0, 0xFFFFFFFF, 0x1000);
        g_scratch_size = need;
    }
    SetDev(kDevPush, g_scratch_push);
    SetDev(kDevThreshold, g_scratch_push + g_scratch_size - 0x1000);
    g_walk = g_scratch_push;
    return g_scratch_push;
}

gpu::Surface SurfaceOf(uint32_t header) {
    gpu::Surface s;
    s.header = header;
    if (header) {
        s.data = R32(header + 4);
        s.format = R32(header + 12);
        s.size = R32(header + 16);
    }
    return s;
}

uint32_t NewSurface(uint32_t w, uint32_t h, uint32_t fmt, uint32_t bpp) {
    uint32_t pitch = (w * bpp + 63) & ~63u;
    uint32_t data = mem::AllocContiguous(pitch * h, 0, 0xFFFFFFFF, 0x1000);
    uint32_t hdr = mem::AllocSystem(24);
    W32(hdr + 0, 0x000D0001);  // surface, one reference, one internal reference held by the device
    W32(hdr + 4, mem::PhysicalFromVirtual(data));
    W32(hdr + 8, 0);
    W32(hdr + 12, 0x00010021 | (fmt << 8));  // dma A, 2D, one level
    W32(hdr + 16, (w - 1) | (h - 1) << 12 | (pitch / 64 - 1) << 24);
    W32(hdr + 20, 0);
    return hdr;
}

void SyncDeferred() {
    // deferred states live in the title's memory, the inline SetRenderState and SetTextureStageState write them
    for (const Run& r : kRsDeferred) Refresh(&g_state.rs[r.host], kRenderState + r.title * 4, r.count * 4);
    for (uint32_t s = 0; s < 4; s++)
        for (const Run& r : kTssDeferred)
            Refresh(&g_state.tss[s][r.host], kTextureState + (s * kStageStates + r.title) * 4, r.count * 4);
}

float ZScale(const gpu::Surface& zs) {
    uint32_t f = zs.header ? (zs.format >> 8) & 0xFF : 0x2E;
    return f == 0x2C || f == 0x30 ? 65535.0f : 16777215.0f;
}

// The widened frame's guest width at app::HostAspect(), 0 when host widening is
// off. Linux only: the frame target is built this wide (gpu/vulkan.cpp
// WidenedGuestWidth, the same formula), and everything below centres on it.
// Windows keeps its AspectWiden-based width in its own backend and its own
// numbers below, unchanged.
uint32_t WidenedFrameWidth() {
#if defined(_WIN32)
    return 0;
#else
    float aspect = app::HostAspect();
    if (aspect <= 0.0f) return 0;
    return std::max(g_frame_w, uint32_t(std::lround(float(g_frame_h) * aspect)));
#endif
}

// The main frame is drawn wider than the console's so an ultrawide display shows more world. Only the main
// frame grows, so a pass drawn into an off-screen surface of another size keeps its own centre and its own
// numbers. The centre moves to the widened frame's centre (this shift, in render target pixels: a supersampled
// back buffer has SuperSampleScale of them to each pixel of the frame), the 3-D fold's horizontal scale grows
// with the frame (FrameFoldScale), and the 2-D screen matrix keeps its authored scale so menus and HUD keep
// their size and only move to the new centre. Hor+, no stretch, vertical field of view untouched. Zero on
// 16:9, with ultrawide off, or when the backend draws the console's width (gpu::Backend::WidensFrame), and
// then nothing below it runs.
float FrameCentreShift() {
#if defined(_WIN32)
    if (!g_frame_widened) return 0;
    float dx = float(std::lround(g_frame_w * app::AspectWiden()) - long(g_frame_w)) * 0.5f;
#else
    uint32_t wide_w = WidenedFrameWidth();
    if (!wide_w) return 0;
    float dx = (float(wide_w) - float(g_frame_w)) * 0.5f;
#endif
    if (dx == 0) return 0;
    uint32_t w = 0, h = 0;
    gpu::SurfaceSize(g_state.rt.header ? g_state.rt : g_state.zs, w, h);
    return w == g_frame_w && h == g_frame_h ? dx * g_state.super_x : 0;
}

// How much wider the widened frame is than the console's, for the 3-D fold's
// horizontal scale: the fold maps post-projection NDC to frame pixels, so its
// scale grows with the frame or the widened world would squeeze back into the
// console's columns. 1 when widening is off or the current target is not the
// frame (an off-screen pass keeps its own fold, scale 1 offset 0 included).
float FrameFoldScale() {
#if defined(_WIN32)
    return 1.0f;
#else
    uint32_t wide_w = WidenedFrameWidth();
    if (!wide_w || FrameCentreShift() == 0) return 1.0f;
    return float(wide_w) / float(g_frame_w);
#endif
}

// D3DMULTISAMPLE_TYPE: x samples in bits 4 to 7, y in bits 0 to 3, 0x1000 a multisampling type, 0x2000 a supersampling
// one. The library's mode setup (5558: 0x24FB40) makes the back buffer x by y times the frame and keeps that as its
// BackBufferScale, and its render target setup (0x24AAC0) takes SuperSampleScale from it, halved by the
// D3DRS_MULTISAMPLEMODE a multisampling type starts with: so a supersampled back buffer has x by y render target pixels
// to each of the frame's (this title asks for 0x2222, D3DMULTISAMPLE_4_SAMPLES_SUPERSAMPLE_GAUSSIAN, and doubles its own
// vertex program positions to match), and NONE (0x11) keeps the frame's. Every multisampling type is left at 1 here,
// which is the library's own value for the 2x1 and 2x2 ones; for 3x3 (0x1233) it gives 1.5, and the whole
// BackBufferScale to a title that sets D3DRS_MULTISAMPLEMODE to 0. Neither is modelled.
uint32_t g_super_x = 1, g_super_y = 1;  // the presentation's, on its back buffers

void SetMultiSampleType(uint32_t type) {
    bool super = type & 0x2000;
    g_super_x = super ? std::max(1u, type >> 4 & 0xF) : 1;
    g_super_y = super ? std::max(1u, type & 0xF) : 1;
}

// The render target's SuperSampleScale as the library's render target setup (5558: 0x24AAC0) keeps it: the
// presentation's when the target is the device's back buffer, 1 on any other surface. Into the device, where the
// library's own viewport, scissor and clear code reads it, and into g_state for the renderer. The title's surfaces
// stay the frame's size: only screen space positions are in the larger pixels. A device without a BackBuffer field
// (3925's holds its frame buffer surfaces themselves) is compared with g_buffers[1], what SetupPresentation puts there.
void UpdateSuperSample() {
    uint32_t back_buffer = kDevBackBuffer != xdk::kUnknown ? Dev(kDevBackBuffer) : g_buffers[1];
    bool back = Dev(kDevRenderTarget) == back_buffer;
    float x = back ? float(g_super_x) : 1.0f, y = back ? float(g_super_y) : 1.0f;
    std::memcpy(HostPtr(kDevice + kDevSuperX), &x, 4);
    std::memcpy(HostPtr(kDevice + kDevSuperY), &y, 4);
    Set(g_state.super_x, x);
    Set(g_state.super_y, y);
}

constexpr uint32_t kVscCount = 192;
// Where that shift lands, read off the vertex programs this title really ran - the renderer's own translations
// in out/build/verbose/cap*/*/vs_*.hlsl and the constants dumped beside them in frame.txt:
//  - world: oPos = dot(v0, c[129..132]), then oPos.xyz *= c[97].xyz, then oPos.xyz = oPos.xyz / oPos.w +
//    c[98].xyz. c97 was (320, -224, ...) and c98 (320.031, 240.031, ...): the engine's own copy of the viewport
//    fold the XDK writes to c[58] and c[59]. Its horizontal centre is c98.x.
//  - 2D and HUD: oPos = dot(v0, c[121..124]) and nothing more, never touching c[97] or c[98]. c25 to c28
//    (hardware 121 to 124) held (640,0,0,0.03125) (0,448,0,16.0312) (0,0,z,0) (0,0,0,1) in all 1770 captured
//    draws: unit screen coordinates to console pixels by dp4, with every quad's w at 1, so its horizontal
//    translation is c25.w. The 2D batcher (guest sub_001D9ED0) also pushes its model matrix to c65 to c68, but
//    no program reads that as a position transform, so it is left alone.
// Of the six programs captured, three read the fold and three read the screen matrix and none read both, so a
// draw is shifted once or not at all.
constexpr uint32_t kVscFoldScale = 97;   // the engine's viewport scale, horizontal in .x
constexpr uint32_t kVscFoldOffset = 98;  // the engine's viewport offset, horizontal in .x
constexpr uint32_t kVscScreen = 121;     // the 2D unit to pixel matrix, horizontal in .w

// a constant has just been stored: centre it when it is the horizontal translation into console pixels,
// and widen the 3-D fold's horizontal scale onto the widened frame (FrameFoldScale). This is the one
// place both moves happen: every push-buffer reader and every SetVertexShaderConstant path stores
// through here, so world draws (fold) and 2-D draws (screen matrix) cannot drift apart.
// The centring depends on which surface is bound, and a title may store these constants while the frame
// is still bound and only then switch to the off-screen surface it meant them for. Baking the shift in at
// store time therefore pushes such a pass off its own target: it draws nothing and the surface keeps its
// clear colour, which is how an arena floor that samples a cleared-white pass came out flat white. Keep
// the value the title actually wrote and re-derive the shifted one whenever the bound surface changes.
float g_vsc_raw[kVscCount][4];
bool g_vsc_raw_set[kVscCount];
void CentreVsConstant(uint32_t r) {
    if (r != kVscFoldScale && r != kVscFoldOffset && r != kVscScreen) return;
    std::memcpy(g_vsc_raw[r], g_state.vsc[r].data(), sizeof(g_vsc_raw[r]));
    g_vsc_raw_set[r] = true;
    if (r == kVscFoldScale) {
        float f = FrameFoldScale();
        if (f == 1.0f) return;
        const float& x = g_state.vsc[r][0];
        Set(x, g_vsc_raw[r][0] * f);
        return;
    }
    float dx = FrameCentreShift();
    if (dx == 0) return;
    uint32_t i = r == kVscFoldOffset ? 0 : 3;
    const float& x = g_state.vsc[r][i];
    Set(x, g_vsc_raw[r][i] + dx);
}

// the bound surface changed: the stored constants were centred for the old one
void RecentreVsConstants() {
    for (uint32_t r : {kVscFoldScale, kVscFoldOffset, kVscScreen}) {
        if (!g_vsc_raw_set[r]) continue;
        uint32_t i = r == kVscScreen ? 3 : 0;
        const float& x = g_state.vsc[r][i];
        Set(x, g_vsc_raw[r][i]);  // back to what the title wrote, then centre for the surface now bound
        CentreVsConstant(r);
    }
}

// the vertex shader constants of one inline push packet: [kVscPush, register + 96, (dwords << 18) | kVscMethod,
// data...]. Both push buffer readers go through here, so the centre shift cannot drift between them.
constexpr uint32_t kVscPush = 0x41EA4;
constexpr uint32_t kVscMethod = 0xB80;
void StoreVsConstants(uint32_t reg, uint32_t n, uint32_t data) {
    for (uint32_t i = 0; i + 4 <= n; i += 4) {
        uint32_t r = reg + i / 4;
        if (r >= kVscCount) continue;
        Load(g_state.vsc[r], data + i * 4);
        CentreVsConstant(r);
    }
}

bool g_own_constants = false;  // D3DSCM_NORESERVEDCONSTANTS: c-38 and c-37 are the title's

// c-38 and c-37 hold the viewport scale and offset, as the library's GetViewportOffsetAndScale (5558: 0x249980)
// computes them: the viewport in render target pixels, SuperSampleScale of them to each of its own, with the screen
// space offset added after the scale. Summed in the library's order, which with a scale of 1 is the frame's own.
void UpdateViewportConstants() {
    auto& v = g_state.viewport;
    Set(g_state.zscale, ZScale(g_state.zs));
    float hw = v.w * 0.5f, hh = v.h * 0.5f, sx = g_state.super_x, sy = g_state.super_y;
    if (!g_own_constants) {
        Set(g_state.vsc[58], {hw * sx * FrameFoldScale(), -hh * sy, (v.max_z - v.min_z) * g_state.zscale, 0});
        Set(g_state.vsc[59], {v.x * sx + g_state.ss_offset_x + hw * sx + FrameCentreShift(),
                              v.y * sy + g_state.ss_offset_y + hh * sy, v.min_z * g_state.zscale, 0});
    }
    // the viewport as the library's SetViewport stores it, for inlined readers
    if (XDK_CDevice_Viewport != xdk::kUnknown) {
        uint32_t p = kDevice + XDK_CDevice_Viewport;
        W32(p, v.x);
        W32(p + 4, v.y);
        W32(p + 8, v.w);
        W32(p + 12, v.h);
        std::memcpy(HostPtr(p + 16), &v.min_z, 4);
        std::memcpy(HostPtr(p + 20), &v.max_z, 4);
    }
}

// rs by the host's number; the title's table has it at the build's, unless the build has no such state
void SetComplex(uint32_t rs, uint32_t v) {
    Set(g_state.rs[rs], v);
    if (uint32_t at = kRsMap.title[rs]; at != xdk::kAbsent) W32(kRenderState + at * 4, v);
    else LOG_FIRST(8, "d3d: render state %u is not in this build, the title's D3D_RenderState is left alone", rs);
}

// The library's default render and texture stage states; its pushes land in the scratch buffer
void ApplyDefaultStates(X86Context& ctx) {
#if HAS_D3D_InitDefaultState
    GUEST_CALL(D3D_InitDefaultState, ctx, {});
#else
    (void)ctx;
    LOG_FIRST(1, "d3d: the image has no D3D_InitDefaultState, default states were not applied");
#endif
}

constexpr bool kVBlank = XDK_CDevice_VBlankCount != xdk::kUnknown && XDK_CDevice_VBlankEvent != xdk::kUnknown &&
                         XDK_CDevice_VBlankCallback != xdk::kUnknown;

// what the library's vertical blank interrupt leaves for the title: count, event, callback
void VerticalBlank(X86Context& ctx) {
    using clock = std::chrono::steady_clock;
    uint32_t info = mem::AllocSystem(12), shown = 0;
    auto next = clock::now();
    for (;;) {
        next = std::max(next + std::chrono::microseconds(16667), clock::now());
        std::this_thread::sleep_until(next);
        uint32_t count = Dev(XDK_CDevice_VBlankCount) + 1, swaps = Dev(kDevSwapCount);
        SetDev(XDK_CDevice_VBlankCount, count);
        ObjectRef event = FromPointer(kDevice + XDK_CDevice_VBlankEvent);
        SetEvent(*event);
        if (uint32_t callback = Dev(XDK_CDevice_VBlankCallback)) {
            // blank count, swaps shown, 1 when this blank showed one
            W32(info, count);
            W32(info + 4, swaps);
            W32(info + 8, swaps != shown);
            CallGuest(ctx, callback, {info});
        }
        shown = swaps;
        // the title clears the event in its own memory before a wait, which the host object does not see
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        ResetEvent(*event);
    }
}

}  // namespace

const gpu::State& State() { return g_state; }
std::recursive_mutex& Lock() { return g_lock; }

}  // namespace d3d

using namespace d3d;

// device

namespace {
// Frame surfaces and device fields for one set of presentation parameters, shared by
// Direct3D_CreateDevice (first creation, device already zeroed by the caller) and
// D3DDevice_Reset (mode change on a live device). A Reset that asks for the surfaces
// already in place reuses them: this title Resets at every movie/scene transition, and
// allocating a fresh 640x480 front/back/depth triple each time leaked ~3.6 MB of the
// 64 MB physical space per Reset until surface allocation started failing (data
// physical 0, whose read-back then wrote over low kernel memory). Changed parameters
// still allocate fresh surfaces and leak the old ones, as before: the title may hold
// their headers, and a size/format change is rare.
void SetupPresentation(X86Context& ctx, uint32_t w, uint32_t h, uint32_t fmt, uint32_t count, uint32_t multisample,
                       uint32_t auto_depth, uint32_t depth_fmt, bool fresh) {
    g_frame_w = w;
    g_frame_h = h;
    SetMultiSampleType(multisample);
    uint32_t color_fmt = gpu::LinearColorFormat(fmt);
    const uint32_t want_count = std::min<uint32_t>(count + 1, 3);
    static uint32_t pres_w = 0, pres_h = 0, pres_fmt = 0, pres_count = 0, pres_auto_depth = 0, pres_depth_fmt = 0;
    const bool reuse = g_buffers[0] != 0 && pres_w == w && pres_h == h && pres_fmt == color_fmt &&
                       pres_count == want_count && pres_auto_depth == auto_depth && pres_depth_fmt == depth_fmt;
    if (!reuse) {
        g_buffer_count = want_count;
        for (uint32_t i = 0; i < g_buffer_count; i++) g_buffers[i] = NewSurface(w, h, color_fmt, 4);
        g_depth = 0;
        if (auto_depth) {
            uint32_t zf = gpu::LinearDepthFormat(depth_fmt);
            g_depth = NewSurface(w, h, zf, zf == 0x30 || zf == 0x31 ? 2 : 4);  // 0x30 LIN_D16, 0x31 LIN_F16
        }
        pres_w = w; pres_h = h; pres_fmt = color_fmt; pres_count = want_count;
        pres_auto_depth = auto_depth; pres_depth_fmt = depth_fmt;
    }
    if (!g_backend) {
        g_backend = gpu::CreateBackend();
        if (!g_backend->Init(w, h)) LOGE("d3d: renderer init failed");
        g_frame_widened = g_backend->WidensFrame();
        if (kRenderThread) g_backend = gpu::CreateThreadedBackend(std::move(g_backend));
    }
    // CreateDevice starts from a clean slate; Reset (like the guest body, which only
    // recreates the surfaces, re-points the target and clears) keeps the bound
    // shaders, streams and textures, so only the targets and viewport move.
    if (fresh) g_state_data = gpu::State{};  // every block marked
    Set(g_state.viewport, {0, 0, w, h, 0.0f, 1.0f});
    SetDev(kDevRenderTarget, g_buffers[1]);
    SetDev(kDevDepth, g_depth);
    // the frame buffer pointers, where the device has them: 3925's holds the surfaces themselves, which a pointer
    // written there would overwrite
    if (kDevBackBuffer != xdk::kUnknown) SetDev(kDevBackBuffer, g_buffers[1]);
    if (kDevFrontBuffer != xdk::kUnknown) SetDev(kDevFrontBuffer, g_buffers[1]);
    float one = 1.0f, zs = g_depth && (R32(g_depth + 12) >> 8 & 0xFF) == 0x30 ? 65535.0f : 16777215.0f;
    std::memcpy(HostPtr(kDevice + kDevZScale), &zs, 4);
    // fields the library's init sets that inlined library code and the default states read
    if (XDK_CDevice_AutoDepth != xdk::kUnknown) SetDev(XDK_CDevice_AutoDepth, g_depth);
    if (XDK_CDevice_BackBufferScale != xdk::kUnknown) {
        std::memcpy(HostPtr(kDevice + XDK_CDevice_BackBufferScale), &one, 4);
        std::memcpy(HostPtr(kDevice + XDK_CDevice_BackBufferScale + 4), &one, 4);
    }
    // 60 Hz, widescreen
    if (XDK_CDevice_DisplayFlags != xdk::kUnknown) SetDev(XDK_CDevice_DisplayFlags, 0x400000 | (app::Widescreen() ? 0x10000 : 0));
#if HAS_D3D_FvfShader
    if (XDK_CDevice_VertexShader != xdk::kUnknown) SetDev(XDK_CDevice_VertexShader, SYM(D3D_FvfShader));
#endif
    if (kVBlank) {
        // notification event, signalled
        SetDev(XDK_CDevice_VBlankEvent, 0x00040000);
        SetDev(XDK_CDevice_VBlankEvent + 4, 1);
        static bool started = false;
        if (!started) StartHostThread("vertical blank", VerticalBlank);
        started = true;
    }
    Set(g_state.rt, SurfaceOf(g_buffers[1]));
    Set(g_state.zs, SurfaceOf(g_depth));
    UpdateSuperSample();
    UpdateViewportConstants();
    if (fresh) ApplyDefaultStates(ctx);
}
}  // namespace

HLE_STDCALL(Direct3D_CreateDevice, 6) {
    XDK_REQUIRE("d3d", XDK_DATA(D3D_Device), XDK_DATA(D3D_pDevice), XDK_DATA(D3D_DirtyFlags),
                XDK_DATA(D3D_DeferredTextureState), XDK_DATA(D3D_RenderState), XDK_DATA(D3D_Streams),
                XDK_DATA(D3D_SimpleRenderStateEncode));
    XDK_REQUIRE("d3d", XDK_KEY(CDevice_Push), XDK_KEY(CDevice_PushThreshold), XDK_KEY(CDevice_BaseVertex),
                XDK_KEY(CDevice_IndexBuffer), XDK_KEY(CDevice_ZScale), XDK_KEY(CDevice_SuperSampleX),
                XDK_KEY(CDevice_SuperSampleY), XDK_KEY(CDevice_Transforms), XDK_KEY(CDevice_Textures),
                XDK_KEY(CDevice_RenderTarget), XDK_KEY(CDevice_Depth), XDK_KEY(CDevice_SwapCount),
                XDK_KEY(CDevice_Size), XDK_KEY(DirtyFlags_Texture), XDK_KEY(TextureState_Stage),
                XDK_KEY(Stream_Size), XDK_KEY(Stream_Stride), XDK_KEY(Stream_Buffer));
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    uint32_t pp = a[4];  // D3DPRESENT_PARAMETERS
    uint32_t w = R32(pp), h = R32(pp + 4), fmt = R32(pp + 8), count = std::max<uint32_t>(R32(pp + 12), 1);
    uint32_t auto_depth = R32(pp + 0x20), depth_fmt = R32(pp + 0x24), interval = R32(pp + 0x30);
    if (!w) w = 640;
    if (!h) h = 480;
    LOGI("d3d: CreateDevice %ux%u format %X buffers %u multisample %X depth %u format %X interval %X flags %X", w, h, fmt,
         count, R32(pp + 0x10), auto_depth, depth_fmt, interval, R32(pp + 0x28));
    std::memset(HostPtr(kDevice), 0, XDK_CDevice_Size);
    W32(kDevicePtr, kDevice);
    ScratchPush();
    SetupPresentation(ctx, w, h, fmt, count, R32(pp + 0x10), auto_depth, depth_fmt, true);
    if (a[5]) W32(a[5], kDevice);
    return 0;
}

HLE_STDCALL(D3DDevice_Release, 0) { return 1; }

// D3DDevice_Reset(pPresentationParameters), stdcall ret 4: the guest body (Sims 2 0x271570)
// kicks the GPU idle, tears down and recreates the display surfaces through the NV2A
// miniport at CDevice+0x1C28, then re-establishes the render target and clears. The host
// has no miniport, so it applies the parameters to its own device state instead, through
// the same SetupPresentation path as CreateDevice, and returns S_OK.
HLE_STDCALL(D3DDevice_Reset, 1) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    uint32_t pp = a[0];  // D3DPRESENT_PARAMETERS
    if (!pp) return 0x88760869;  // D3DERR_INVALIDCALL
    uint32_t w = R32(pp), h = R32(pp + 4), fmt = R32(pp + 8), count = std::max<uint32_t>(R32(pp + 12), 1);
    uint32_t auto_depth = R32(pp + 0x20), depth_fmt = R32(pp + 0x24), interval = R32(pp + 0x30);
    if (!w) w = 640;
    if (!h) h = 480;
    LOGI("d3d: Reset %ux%u format %X buffers %u multisample %X depth %u format %X flags %X refresh %u interval %X",
         w, h, fmt, count, R32(pp + 0x10), auto_depth, depth_fmt, R32(pp + 0x28), R32(pp + 0x2C), interval);
    if (g_backend) g_backend->Flush();  // the guest's KickOffAndWaitForIdle
    ScratchPush();
    SetupPresentation(ctx, w, h, fmt, count, R32(pp + 0x10), auto_depth, depth_fmt, false);
    // the guest tail: SetRenderTarget(back buffer, auto depth) then Clear(0, 0, 3, 0, 1.0f, 0);
    // SetupPresentation already pointed the device and g_state at the new buffers
    if (g_backend) g_backend->Clear(g_state, 0, nullptr, 3, 0, 1.0f, 0);
    return 0;
}

// synchronisation, the renderer never runs behind

namespace {
// bytes a linear pixel container spans in guest memory, from its Size dword (pitch / 64 - 1 in bits 24..31, height - 1
// in 12..23, as PersistDisplay reads it); 0 for a swizzled one, whose Size is 0 and whose span is unknown here
uint32_t SurfaceBytes(uint32_t res) {
    uint32_t size = R32(res + 0x10);
    if (!size) return 0;
    return ((size >> 24 & 0xFF) + 1) * 64 * ((size >> 12 & 0xFFF) + 1);
}
}  // namespace

HLE_STDCALL(D3D_BlockOnResource, 1) {
    uint32_t res = a[0];
    if (res && g_backend) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        uint32_t type = R32(res) & 0x70000;  // 0x40000 texture, 0x50000 surface
        if (type == 0x40000 || type == 0x50000) {
            uint32_t data = mem::VirtualFromPhysical(R32(res + 4));
            // perf diagnostic: who blocks on which surface, once each
            static std::unordered_set<uint32_t> seen;
            if (diag::Perf() && seen.insert(data).second) {
                uint32_t code[4] = {}, n = 0;
                for (uint32_t i = 0; i < 32 && n < 4; i++) {
                    uint32_t v = R32(ctx.esp + i * 4);
                    if (v >= X86CodeBase && v < X86CodeEnd) code[n++] = v;
                }
                LOGI("d3d: BlockOnResource on surface %08X, stack %08X %08X %08X %08X", data, code[0], code[1], code[2], code[3]);
            }
            g_backend->Sync(data, SurfaceBytes(res));
        } else {
            g_backend->WaitResource(res);
        }
    }
    gpu::MarkDirty(res);
    return 0;
}
// lock(resource, face, level, locked, rect, flags): D3DLOCK_NOOVERWRITE skips the block above, yet a
// title may read what the GPU drew
HLE_WRAP(Lock2DSurface, 6) {
    uint32_t res = a[0];
    if ((a[5] & 0x20) && res && g_backend) {  // D3DLOCK_NOOVERWRITE
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        g_backend->SyncLate(mem::VirtualFromPhysical(R32(res + 4)), SurfaceBytes(res));
    }
}
HLE_STDCALL(D3DResource_BlockUntilNotBusy, 1) {
    if (a[0] && g_backend) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        g_backend->WaitResource(a[0]);
    }
    gpu::MarkDirty(a[0]);
    return 0;
}
HLE_STDCALL(D3DResource_IsBusy, 1) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    uint32_t res = a[0];
    if (!res) return 0;
    // set on the device, a surface also through its texture
    uint32_t common = R32(res), parent = (common & 0x70000) == 0x50000 ? R32(res + 0x14) : 0;
    if ((common & 0x780000) || (parent && (R32(parent) & 0x780000))) return 1;  // internal reference count
    return g_backend && g_backend->Busy(res) ? 1 : 0;
}
HLE_STDCALL(D3D_BlockOnTime, 2) { return 0; }
HLE_STDCALL(D3DDevice_BlockOnFence, 1) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    if (g_backend) g_backend->WaitFence(a[0]);
    return 0;
}
HLE_STDCALL(D3DDevice_InsertFence, 0) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    return g_backend ? g_backend->Fence() : 0;
}
HLE_STDCALL(D3DDevice_IsFencePending, 1) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    return g_backend && g_backend->FencePending(a[0]) ? 1 : 0;
}
HLE_STDCALL(D3D_SetFence, 1) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    return g_backend ? g_backend->Fence() : 0;
}
HLE_STDCALL(D3D_KickOffAndWaitForIdle, 0) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    if (g_backend) g_backend->Flush();
    return 0;
}
HLE_FASTCALL(CDevice_KickOff, 0) { return 0; }

HLE_STDCALL(D3DDevice_MakeSpace, 0) {
    LOG_FIRST(16, "d3d: MakeSpace from %08X, some caller still pushes hardware commands", R32(ctx.esp));
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    return ScratchPush();
}
HLE_STDCALL(D3D_MakeRequestedSpace, 2) {
    LOG_FIRST(16, "d3d: MakeRequestedSpace from %08X", R32(ctx.esp));
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    return ScratchPush();
}

HLE_STDCALL(D3DDevice_BlockUntilVerticalBlank, 0) {
    // Wait for the blank the title is actually counting, not a fixed sleep. 16 ms is neither the
    // 16.667 ms period nor in phase with it, so a frame that finished just after one blank waited
    // past the next and the title locked to half the refresh.
    if (!kVBlank) {
        std::this_thread::sleep_for(std::chrono::microseconds(16667));
        return 0;
    }
    const uint32_t start = Dev(XDK_CDevice_VBlankCount);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(34);
    while (Dev(XDK_CDevice_VBlankCount) == start) {
        if (std::chrono::steady_clock::now() >= deadline) break;  // the thread is not running: never hang
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    return 0;
}
HLE_STDCALL(D3DDevice_SetFlickerFilter, 1) { return 0; }
HLE_STDCALL(D3DDevice_SetSoftDisplayFilter, 1) { return 0; }
HLE_STDCALL(D3DDevice_EnableOverlay, 1) { return 0; }
// the front buffer, read back at the guest size, is kept for the next instance after a quick reboot, which may
// fade from it at its first frames
HLE_STDCALL(D3DDevice_PersistDisplay, 0) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    uint32_t front = g_buffers[0], size = R32(front + 0x10);
    uint32_t pitch = ((size >> 24 & 0xFF) + 1) * 64, height = (size >> 12 & 0xFFF) + 1, bytes = pitch * height;
    uint32_t data = mem::VirtualFromPhysical(R32(front + 4));
    if (g_backend) g_backend->Sync(data, bytes);
    kernel::SavePersistedDisplay(R32(front), R32(front + 0xC), size, HostPtr(data), bytes);
    LOGI("d3d: PersistDisplay, %u bytes", bytes);
    return 0;
}
HLE_STDCALL(D3DDevice_FlushVertexCache, 0) { return 0; }

HLE_STDCALL(D3DDevice_SetGammaRamp, 2) {
    uint8_t ramp[768];
    std::memcpy(ramp, HostPtr(a[1]), 768);
    if (g_backend) g_backend->SetGamma(ramp);
    return 0;
}

// targets and presentation

HLE_STDCALL(D3DDevice_SetRenderTarget, 2) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    uint32_t rt = a[0] ? a[0] : Dev(kDevRenderTarget), zs = a[1];
    AddRef(ctx, rt);
    uint32_t old_rt = Dev(kDevRenderTarget), old_zs = Dev(kDevDepth);
    SetDev(kDevRenderTarget, rt);
    SetDev(kDevDepth, zs);
    AddRef(ctx, zs);
    Release(ctx, old_rt);
    Release(ctx, old_zs);
    Set(g_state.rt, SurfaceOf(rt));
    Set(g_state.zs, SurfaceOf(zs));
    UpdateSuperSample();
    RecentreVsConstants();  // the stored fold and screen constants were centred for the old surface
    // SetRenderTarget resets the viewport to the whole target
    uint32_t w, h;
    gpu::SurfaceSize(g_state.rt, w, h);
    Set(g_state.viewport, {0, 0, w, h, 0.0f, 1.0f});
    UpdateViewportConstants();
    return 0;
}

namespace {
// the surface GetBackBuffer2 returns for an index, -1 the front buffer, with a reference added
uint32_t BackBufferRef(int32_t i) {
    uint32_t s = i < 0 ? g_buffers[0] : g_buffers[std::min<uint32_t>(uint32_t(i) + 1, g_buffer_count - 1)];
    W32(s, R32(s) + 1);
    return s;
}
// the depth surface GetDepthStencilSurface2 returns, with a reference added when there is one
uint32_t DepthStencilRef() {
    uint32_t s = Dev(kDevDepth);
    if (s) W32(s, R32(s) + 1);
    return s;
}
}  // namespace

HLE_STDCALL(D3DDevice_GetBackBuffer2, 1) { return BackBufferRef(int32_t(a[0])); }

// GetBackBuffer(BackBuffer, Type, ppSurface), XDK 3925's form: the same surface through the pointer, and S_OK
HLE_STDCALL(D3DDevice_GetBackBuffer, 3) {
    W32(a[2], BackBufferRef(int32_t(a[0])));
    return 0;
}

HLE_STDCALL(D3DDevice_GetRenderTarget2, 0) {
    uint32_t s = Dev(kDevRenderTarget);
    if (s) W32(s, R32(s) + 1);
    return s;
}

HLE_STDCALL(D3DDevice_GetDepthStencilSurface2, 0) { return DepthStencilRef(); }

// GetDepthStencilSurface(ppSurface), XDK 3925's form: the same surface through the pointer, and S_OK, or
// D3DERR_NOTFOUND with NULL stored when the device has none, as the library does it
HLE_STDCALL(D3DDevice_GetDepthStencilSurface, 1) {
    uint32_t s = DepthStencilRef();
    W32(a[0], s);
    return s ? 0 : 0x88760866;  // D3DERR_NOTFOUND
}

namespace {
std::atomic<uint32_t> g_presents{0};

// One frame to the window, shared by D3DDevice_Swap and XDK 3925's D3DDevice_Present (name, for the log): the gamma
// ramp left pending, the back buffer shown, the frame buffers' memory rotated, the swap count incremented and returned.
uint32_t PresentFrame(const char* name) {
    app::SplashFirstFrame();
    // handed over from the previous instance: the window opens on the second frame, the first picture with
    // anything drawn over it right after
    if (app::HandingOver() && g_presents >= 2) {
        {
            std::lock_guard<std::recursive_mutex> lock(g_lock);
            if (g_backend) g_backend->Flush();
        }
        app::FinishHandover();
    }
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    g_presents++;
    LOG_FIRST(8, "d3d: %s %u", name, g_presents.load());
    // the kept display is shown until the title presents its own
    kernel::SetSavedDataAddress(0);
    // the ramp an inlined SetGammaRamp left for the vertical blank that shows this swap
    if (g_backend && XDK_CDevice_GammaPending != xdk::kUnknown && XDK_CDevice_GammaRamps != xdk::kUnknown) {
        // the slot an inlined SetGammaRamp writes does not always match this swap's parity, so take
        // whichever slot is pending, the one for this swap last
        uint32_t cur = Dev(kDevSwapCount) & 1;
        for (uint32_t k = 0; k < 2; k++) {
            uint32_t i = (cur + 1 + k) & 1;
            if (Dev(XDK_CDevice_GammaPending + i * 4) == 1) {
                SetDev(XDK_CDevice_GammaPending + i * 4, 0);
                g_backend->SetGamma(static_cast<const uint8_t*>(HostPtr(kDevice + XDK_CDevice_GammaRamps + i * 0x300)));
            }
        }
    }
    if (g_backend) g_backend->Present(SurfaceOf(g_buffers[1]));
    app::Pace();
    // like the XDK, the surfaces stay put and their memory rotates: titles keep back buffer pointers and copy its data
    uint32_t last = g_buffers[g_buffer_count - 1], data = R32(last + 4), lk = R32(last + 8);
    for (uint32_t i = g_buffer_count - 1; i > 0; i--) {
        W32(g_buffers[i] + 4, R32(g_buffers[i - 1] + 4));
        W32(g_buffers[i] + 8, R32(g_buffers[i - 1] + 8));
    }
    W32(g_buffers[0] + 4, data);
    W32(g_buffers[0] + 8, lk);
    Set(g_state.rt, SurfaceOf(Dev(kDevRenderTarget)));
    RecentreVsConstants();
    uint32_t n = Dev(kDevSwapCount) + 1;
    SetDev(kDevSwapCount, n);
    return n;
}
}  // namespace

HLE_STDCALL(D3DDevice_Swap, 1) { return PresentFrame("Swap"); }

// Present(pSourceRect, pDestRect, hDestWindowOverride, pDirtyRegion), how XDK 3925 presents: it has no Swap. The
// same frame as Swap's, the arguments ignored as Swap's flags are, and S_OK, the HRESULT the library's own body
// leaves in eax whenever no display mode change is pending.
HLE_STDCALL(D3DDevice_Present, 4) {
    PresentFrame("Present");
    return 0;
}

HLE_STDCALL(D3DDevice_Clear, 6) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    const uint8_t* rects = a[0] && a[1] ? static_cast<const uint8_t*>(HostPtr(a[1])) : nullptr;
    if (g_backend) g_backend->Clear(g_state, a[0], rects, a[2], a[3], ArgF(a, 4), a[5]);
    return 0;
}

HLE_STDCALL(D3DDevice_CopyRects, 5) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    if (g_backend) g_backend->Copy(SurfaceOf(a[0]), SurfaceOf(a[3]), a[2], a[1], a[4]);
    return 0;
}

HLE_STDCALL(D3DDevice_SetViewport, 1) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    uint32_t v = a[0];
    Set(g_state.viewport, {R32(v), R32(v + 4), R32(v + 8), R32(v + 12), GF(v + 16), GF(v + 20)});
    Set(g_state.scissor_count, 0);
    UpdateViewportConstants();
    return 0;
}

HLE_STDCALL(D3DDevice_SetScissors, 3) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    Set(g_state.scissor_count, std::min<uint32_t>(a[0], 8));
    for (uint32_t i = 0; i < g_state.scissor_count * 4; i++) Set(g_state.scissors[i], R32(a[2] + i * 4));
    return 0;
}

HLE_STDCALL(D3DDevice_SetScreenSpaceOffset, 2) {
    Set(g_state.ss_offset_x, ArgF(a, 0));
    Set(g_state.ss_offset_y, ArgF(a, 1));
    UpdateViewportConstants();
    return 0;
}

// render states

HLE_FASTCALL(D3DDevice_SetRenderState_Simple, 0) {
    uint32_t method = a[0] & 0x1FFC, value = a[1];
    Set(g_state.simple[method >> 2], value);
    return 0;
}

HLE_STDCALL(D3DDevice_SetRenderStateNotInline, 2) {
    uint32_t rs = a[0], v = a[1];  // rs by the build's number
    if (rs < title::kRsDeferredFirst) {
        uint32_t method = R32(kSimpleEncode + rs * 4) & 0x1FFC;
        Set(g_state.simple[method >> 2], v);
        W32(kRenderState + rs * 4, v);  // titles read their states back from the library's table
    } else if (rs < title::kRsMax) {
        if (uint32_t host = kRsMap.host[rs]; host != xdk::kAbsent) {
            SetComplex(host, v);
        } else {
            // a state the host has no number for (3925's D3DRS_MULTISAMPLETYPE): only the library's table keeps it
            W32(kRenderState + rs * 4, v);
            LOG_FIRST(8, "d3d: render state %u of this build has no host meaning, it is only stored", rs);
        }
    }
    return 0;
}

#define COMPLEX_STATE(name, rs)                       \
    HLE_STDCALL(D3DDevice_SetRenderState_##name, 1) { \
        SetComplex(rs, a[0]);                         \
        return 0;                                     \
    }
COMPLEX_STATE(PSTextureModes, gpu::kRsPsTextureModes)
COMPLEX_STATE(VertexBlend, gpu::kRsVertexBlend)
COMPLEX_STATE(FogColor, gpu::kRsFogColor)
COMPLEX_STATE(FillMode, gpu::kRsFillMode)
COMPLEX_STATE(BackFillMode, gpu::kRsBackFillMode)
COMPLEX_STATE(TwoSidedLighting, gpu::kRsTwoSidedLighting)
COMPLEX_STATE(NormalizeNormals, gpu::kRsNormalizeNormals)
COMPLEX_STATE(ZEnable, gpu::kRsZEnable)
COMPLEX_STATE(StencilEnable, gpu::kRsStencilEnable)
COMPLEX_STATE(StencilFail, gpu::kRsStencilFail)
COMPLEX_STATE(FrontFace, gpu::kRsFrontFace)
COMPLEX_STATE(CullMode, gpu::kRsCullMode)
COMPLEX_STATE(TextureFactor, gpu::kRsTextureFactor)
COMPLEX_STATE(LogicOp, gpu::kRsLogicOp)
COMPLEX_STATE(EdgeAntiAlias, gpu::kRsEdgeAntiAlias)
COMPLEX_STATE(MultiSampleAntiAlias, gpu::kRsMultiSampleAntiAlias)
COMPLEX_STATE(MultiSampleMask, gpu::kRsMultiSampleMask)
COMPLEX_STATE(MultiSampleMode, gpu::kRsMultiSampleMode)
COMPLEX_STATE(MultiSampleRenderTargetMode, gpu::kRsMultiSampleRenderTargetMode)
COMPLEX_STATE(ShadowFunc, gpu::kRsShadowFunc)
COMPLEX_STATE(LineWidth, gpu::kRsLineWidth)
COMPLEX_STATE(SampleAlpha, gpu::kRsSampleAlpha)
COMPLEX_STATE(Dxt1NoiseEnable, gpu::kRsDxt1NoiseEnable)
COMPLEX_STATE(YuvEnable, gpu::kRsYuvEnable)
COMPLEX_STATE(OcclusionCullEnable, gpu::kRsOcclusionCullEnable)
COMPLEX_STATE(StencilCullEnable, gpu::kRsStencilCullEnable)
COMPLEX_STATE(RopZCmpAlwaysRead, gpu::kRsRopZCmpAlwaysRead)
COMPLEX_STATE(RopZRead, gpu::kRsRopZRead)
COMPLEX_STATE(DoNotCullUncompressed, gpu::kRsDoNotCullUncompressed)

HLE_STDCALL(D3DDevice_SetTextureState_TexCoordIndex, 2) {
    constexpr uint32_t type = kTssMap.title[gpu::kTssTexCoordIndex];  // the build's
    uint32_t stage = a[0] & 3;
    Set(g_state.tss[stage][gpu::kTssTexCoordIndex], a[1]);
    if (type != xdk::kAbsent) W32(kTextureState + (stage * kStageStates + type) * 4, a[1]);
    return 0;
}
HLE_STDCALL(D3DDevice_SetTextureState_BumpEnv, 3) {
    uint32_t type = a[1] < gpu::kTssCount ? kTssMap.host[a[1]] : xdk::kAbsent;  // the host's number of the build's
    if (type >= gpu::kTssBumpEnvMat00 && type <= gpu::kTssBumpEnvLOffset) Set(g_state.tss[a[0] & 3][type], a[2]);
    return 0;
}
HLE_STDCALL(D3DDevice_SetTextureState_BorderColor, 2) {
    Set(g_state.tss[a[0] & 3][gpu::kTssBorderColor], a[1]);
    return 0;
}
HLE_STDCALL(D3DDevice_SetTextureState_ColorKeyColor, 2) {
    Set(g_state.tss[a[0] & 3][gpu::kTssColorKeyColor], a[1]);
    return 0;
}
HLE_STDCALL(D3DDevice_SetTextureStageStateNotInline, 3) {
    uint32_t stage = a[0] & 3, type = a[1], v = a[2];  // type by the build's number
    if (type < gpu::kTssCount) {
        if (uint32_t host = kTssMap.host[type]; host != xdk::kAbsent) Set(g_state.tss[stage][host], v);
        else LOG_FIRST(8, "d3d: texture stage state %u of this build has no host meaning, it is only stored", type);
        W32(kTextureState + (stage * kStageStates + type) * 4, v);
    }
    return 0;
}

// the XDK's transform types: view 0, projection 1, texture 2-5, world 6-9; the renderer keeps Direct3D 8's numbers
// (view 2, projection 3, texture 16, world 256). The device copy is what GetTransform reads.
HLE_STDCALL(D3DDevice_SetTransform, 2) {
    uint32_t t = a[0];
    if (t >= 10) return 0;
    std::memcpy(HostPtr(kDevice + kDevTransforms + t * 64), HostPtr(a[1]), 64);
    uint32_t pc = t < 2 ? t + 2 : t < 6 ? t + 14 : t + 250;
    Load(g_state.transforms[pc], a[1]);
    return 0;
}

HLE_STDCALL(D3DDevice_SetMaterial, 1) {
    Load(g_state.material, a[0]);
    return 0;
}

// the XDK keeps its own light lists, the renderer gets a copy
HLE_WRAP(D3DDevice_SetLight, 2) {
    uint32_t i = a[0], l = a[1];
    if (i < 8 && l) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        Load(g_state.lights[i], l);
    }
}
HLE_WRAP(D3DDevice_LightEnable, 2) {
    uint32_t i = a[0], on = a[1];
    if (i < 8) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        Set(g_state.light_enable, on ? g_state.light_enable | 1u << i : g_state.light_enable & ~(1u << i));
    }
}

// resources in use

HLE_STDCALL(D3DDevice_SetTexture, 2) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    uint32_t stage = a[0] & 3, tex = a[1];
    uint32_t old = Dev(kDevTextures + stage * 4);
    AddRef(ctx, tex);
    SetDev(kDevTextures + stage * 4, tex);
    Release(ctx, old);
    Set(g_state.textures[stage], tex);
    W32(kDirtyFlags, R32(kDirtyFlags) | XDK_DirtyFlags_Texture);
    return 0;
}

// SetPalette(stage, palette): the colours a P8 texture of the stage looks up
HLE_STDCALL(D3DDevice_SetPalette, 2) {
    XDK_REQUIRE("d3d", XDK_KEY(CDevice_Palettes));
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    uint32_t stage = a[0] & 3, pal = a[1];
    uint32_t old = Dev(kDevPalettes + stage * 4);
    AddRef(ctx, pal);
    SetDev(kDevPalettes + stage * 4, pal);
    Release(ctx, old);
    Set(g_state.palettes[stage], pal);
    return 0;
}

// SwitchTexture(method, data, format), fastcall: a swizzled texture of the bound one's shape replaces it. It is
// bound through a header of the stage's own holding the bound texture's fields with the new data and format.
HLE_FASTCALL(D3DDevice_SwitchTexture, 1) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    static uint32_t headers[4];
    // texture methods from 0x1B00, 0x40 per stage
    uint32_t stage = (((a[0] & 0x1FFC) - 0x1B00) >> 6) & 3, cur = g_state.textures[stage];
    uint32_t& h = headers[stage];
    if (!h) h = mem::AllocSystem(20);
    if (cur && cur != h) std::memcpy(HostPtr(h), HostPtr(cur), 20);
    W32(h + 4, a[1]);
    W32(h + 12, a[2]);
    Set(g_state.textures[stage], h);
    return 0;
}

HLE_STDCALL(D3DDevice_SetStreamSource, 3) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    uint32_t n = a[0] & 15, vb = a[1], stride = a[2];
    uint32_t old = R32(kStreams + n * kStreamSize + XDK_Stream_Buffer);
    AddRef(ctx, vb);
    W32(kStreams + n * kStreamSize + XDK_Stream_Stride, stride);
    W32(kStreams + n * kStreamSize + XDK_Stream_Buffer, vb);
    Release(ctx, old);
    Set(g_state.streams[n], {vb, stride});
    return 0;
}

HLE_STDCALL(D3DDevice_SetIndices, 2) {
    XDK_REQUIRE("d3d", XDK_DATA(D3D_IndexData));
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    uint32_t ib = a[0];
    uint32_t old = Dev(kDevIndexBuffer);
    AddRef(ctx, ib);
    W32(kIndexData, ib ? R32(ib + 4) : 0);
    SetDev(kDevIndexBuffer, ib);
    SetDev(kDevBaseVertex, a[1]);
    Release(ctx, old);
    Set(g_state.index_buffer, ib);
    Set(g_state.index_data, ib ? R32(ib + 4) : 0);
    Set(g_state.base_vertex, a[1]);
    return 0;
}

// shaders

HLE_STDCALL(D3DDevice_CreateVertexShader, 4) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    uint32_t decl = a[0], func = a[1], out = a[2];
    auto vs = std::make_shared<gpu::VertexShader>();
    if (decl) {
        for (uint32_t i = 0; i < 1024; i++) {
            uint32_t t = R32(decl + i * 4);
            vs->decl.push_back(t);
            if (t == 0xFFFFFFFF) break;
        }
    }
    if (func) {
        uint32_t hdr = R32(func);
        vs->program_type = hdr & 0xFFFF;
        uint32_t n = hdr >> 16;
        for (uint32_t i = 0; i < n * 4; i++) vs->program.push_back(R32(func + 4 + i * 4));
    }
    gpu::ParseDeclaration(*vs);
    uint32_t rec = mem::AllocSystem(16, 16);
    vs->handle = rec | 1;
    g_vs[vs->handle] = vs;
    W32(out, vs->handle);
    LOG_FIRST(64, "d3d: vertex shader %08X, %zu decl tokens, %zu instructions, type %X", vs->handle, vs->decl.size(),
              vs->program.size() / 4, vs->program_type);
    return 0;
}

HLE_STDCALL(D3DDevice_DeleteVertexShader, 1) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    auto it = g_vs.find(a[0]);
    if (it != g_vs.end()) {
        mem::FreeSystem(a[0] & ~1u);
        g_vs.erase(it);
    }
    return 0;
}

// (handle, data, size in and out): the program as it was given, the size alone without data or room
HLE_STDCALL(D3DDevice_GetVertexShaderFunction, 3) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    auto it = g_vs.find(a[0]);
    uint32_t words = it == g_vs.end() ? 0 : uint32_t(it->second->program.size());
    uint32_t need = words ? 4 + words * 4 : 0, room = R32(a[2]);
    W32(a[2], need);
    if (need > room) return a[1] ? 0x88760867 : 0;  // D3DERR_MOREDATA
    if (!a[1] || !need) return 0;
    W32(a[1], it->second->program_type | words / 4 << 16);
    for (uint32_t i = 0; i < words; i++) W32(a[1] + 4 + i * 4, it->second->program[i]);
    return 0;
}

HLE_STDCALL(D3DDevice_SetVertexShader, 1) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    uint32_t h = a[0];
    Set(g_state.vs_handle, h);
    Set(g_state.vs, nullptr);
    if (h & 1) {
        auto it = g_vs.find(h);
        if (it != g_vs.end()) {
            Set(g_state.vs, it->second);
            // constants declared with the shader load when it is selected
            for (auto& [reg, v] : it->second->constants)
                if (reg < kVscCount) {
                    Set(g_state.vsc[reg], v);
                    CentreVsConstant(reg);
                }
        } else {
            LOG_FIRST(8, "d3d: SetVertexShader with unknown handle %08X", h);
        }
    }
    return 0;
}

// (register, source, dword count) in ecx/edx/stack: the guest original also copies
// the data into the fast-constant scratch at 0x259AE0 before tail-calling
// SetVertexShaderConstantNotInlineFast; the host only needs its own constant copy.
HLE_FASTCALL(D3DDevice_SetVertexShaderConstant1Fast, 1) {
    uint32_t reg = a[0], src = a[1], dwords = a[2];
    for (uint32_t i = 0; i < dwords / 4 && reg + i < kVscCount; i++) {
        Load(g_state.vsc[reg + i], src + i * 16);
        CentreVsConstant(reg + i);
    }
    return 0;
}

HLE_FASTCALL(D3DDevice_SetVertexShaderConstant1, 0) {
    uint32_t reg = a[0];
    if (reg < kVscCount) {
        Load(g_state.vsc[reg], a[1]);
        CentreVsConstant(reg);
    }
    return 0;
}

HLE_FASTCALL(D3DDevice_SetVertexShaderConstant4, 0) {
    uint32_t reg = a[0], src = a[1];
    for (uint32_t i = 0; i < 4 && reg + i < kVscCount; i++) {
        Load(g_state.vsc[reg + i], src + i * 16);
        CentreVsConstant(reg + i);
    }
    return 0;
}

HLE_FASTCALL(D3DDevice_SetVertexShaderConstantNotInlineFast, 1) {
    uint32_t reg = a[0], src = a[1], dwords = a[2];
    for (uint32_t i = 0; i < dwords / 4 && reg + i < kVscCount; i++) {
        Load(g_state.vsc[reg + i], src + i * 16);
        CentreVsConstant(reg + i);
    }
    return 0;
}

HLE_STDCALL(D3DDevice_LoadVertexShader, 2) {
    LOG_FIRST(4, "d3d: LoadVertexShader %08X at %u is not supported", a[0], a[1]);
    return 0;
}
HLE_STDCALL(D3DDevice_SelectVertexShader, 2) {
    LOG_FIRST(4, "d3d: SelectVertexShader %08X at %u is not supported", a[0], a[1]);
    return 0;
}
HLE_STDCALL(D3DDevice_SetShaderConstantMode, 1) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    g_own_constants = a[0] & 0x10;  // D3DSCM_NORESERVEDCONSTANTS
    UpdateViewportConstants();
    return 0;
}

HLE_STDCALL(D3DDevice_CreatePixelShader, 2) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    auto ps = std::make_shared<gpu::PixelShader>();
    std::memcpy(ps->def, HostPtr(a[0]), sizeof(ps->def));
    uint32_t rec = mem::AllocSystem(16, 16);
    ps->handle = rec;
    g_ps[rec] = ps;
    W32(a[1], rec);
    return 0;
}

HLE_STDCALL(D3DDevice_DeletePixelShader, 1) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    if (g_ps.erase(a[0])) mem::FreeSystem(a[0]);
    return 0;
}

HLE_STDCALL(D3DDevice_SetPixelShader, 1) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    Set(g_state.ps, nullptr);
    if (!a[0]) return 0;
    auto it = g_ps.find(a[0]);
    if (it != g_ps.end()) {
        Set(g_state.ps, it->second);
    } else if (XDK_PixelShader_Definition != xdk::kUnknown) {
        // made by library code the title inlined: the object points at its definition, handles are reused
        static std::unordered_map<uint32_t, std::shared_ptr<gpu::PixelShader>> made;
        const void* def = HostPtr(R32(a[0] + XDK_PixelShader_Definition));
        auto& ps = made[a[0]];
        if (!ps || std::memcmp(ps->def, def, sizeof(ps->def))) {
            ps = std::make_shared<gpu::PixelShader>();
            ps->handle = a[0];
            std::memcpy(ps->def, def, sizeof(ps->def));
        }
        Set(g_state.ps, ps);
    } else {
        LOG_FIRST(8, "d3d: SetPixelShader with unknown handle %08X", a[0]);
        return 0;
    }
    // the definition loads the combiner state like the XDK pushes it
    const uint32_t* d = g_state.ps->def;
    auto put = [](uint32_t method, uint32_t v) { Set(g_state.simple[method >> 2], v); };
    for (int i = 0; i < 8; i++) {
        put(gpu::kMPsAlphaInputs + i * 4, d[0 + i]);
        put(gpu::kMPsConstant0 + i * 4, d[10 + i]);
        put(gpu::kMPsConstant1 + i * 4, d[18 + i]);
        put(gpu::kMPsAlphaOutputs + i * 4, d[26 + i]);
        put(gpu::kMPsRgbInputs + i * 4, d[34 + i]);
        put(gpu::kMPsRgbOutputs + i * 4, d[45 + i]);
    }
    put(gpu::kMPsFinalAbcd, d[8]);
    put(gpu::kMPsFinalEfgh, d[9]);
    put(gpu::kMPsCompareMode, d[42]);
    put(gpu::kMPsFinalConstant0, d[43]);
    put(gpu::kMPsFinalConstant1, d[44]);
    put(gpu::kMPsCombinerCount, d[53]);
    put(gpu::kMPsDotMapping, d[55]);
    put(gpu::kMPsInputTexture, d[56]);
    SetComplex(gpu::kRsPsTextureModes, d[54]);
    return 0;
}

HLE_STDCALL(D3DDevice_SetPixelShaderConstant, 3) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    uint32_t reg = a[0], src = a[1], n = a[2];
    for (uint32_t i = 0; i < n && reg + i < 16; i++) Load(g_state.psc[reg + i], src + i * 16);
    gpu::ApplyPixelConstants(g_state_data, reg, n);
    // what it may write: the combiner constants, eight at kMPsConstant0 then eight at kMPsConstant1, and the final two
    g_state.Mark(&g_state.simple[gpu::kMPsConstant0 >> 2], gpu::kMPsConstant1 + 8 * 4 - gpu::kMPsConstant0);
    g_state.Mark(&g_state.simple[gpu::kMPsFinalConstant0 >> 2], gpu::kMPsFinalConstant1 + 4 - gpu::kMPsFinalConstant0);
    return 0;
}

// drawing

HLE_STDCALL(D3DDevice_DrawVertices, 3) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    HleScope hle;
    Walk(Dev(kDevPush));  // constants/attributes pushed inline since the last drain apply to this draw
    SyncDeferred();
    if (g_backend) g_backend->Draw(g_state, a[0], a[1], a[2], nullptr);
    return 0;
}

HLE_STDCALL(D3DDevice_DrawIndexedVertices, 3) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    HleScope hle;
    Walk(Dev(kDevPush));  // constants/attributes pushed inline since the last drain apply to this draw
    SyncDeferred();
    const uint8_t* indices = a[2] ? static_cast<const uint8_t*>(HostPtr(a[2])) : nullptr;
    if (g_backend) g_backend->Draw(g_state, a[0], 0, a[1], indices);
    return 0;
}

// DrawVerticesUP(prim, count, vertices, stride): the vertices are copied to a ring the render thread reads, stream 0
// points there through a vertex buffer header for the draw. A half of the ring is written again once the draws
// that read it are done.
namespace {
constexpr uint32_t kUpRing = 4u << 20, kUpHalf = kUpRing / 2;
uint32_t g_up_ring = 0, g_up_pos = 0, g_up_header = 0, g_up_fence[2] = {};

uint32_t UpVertices(const void* src, uint32_t bytes) {
    if (!g_up_ring) {
        g_up_ring = mem::AllocContiguous(kUpRing, 0, 0xFFFFFFFF, 0x1000);
        g_up_header = mem::AllocSystem(20);
        std::memset(HostPtr(g_up_header), 0, 20);
        W32(g_up_header, 1);  // a vertex buffer, one reference
    }
    bytes = std::min(bytes, kUpHalf);
    uint32_t room = (bytes + 63) & ~63u;
    uint32_t pos = g_up_pos + room > kUpRing ? 0 : g_up_pos;
    uint32_t from = g_up_pos ? (g_up_pos - 1) / kUpHalf : 1, to = (pos + room - 1) / kUpHalf;
    if (to != from) {
        g_up_fence[from] = g_backend->Fence();
        g_backend->WaitFence(g_up_fence[to]);
    }
    std::memcpy(HostPtr(g_up_ring + pos), src, bytes);
    g_up_pos = pos + room;
    W32(g_up_header + 4, mem::PhysicalFromVirtual(g_up_ring + pos));
    return g_up_header;
}

void DrawUp(uint32_t prim, uint32_t count, const void* vertices, uint32_t stride) {
    if (!g_backend || !count || !stride) return;
    Walk(Dev(kDevPush));  // constants/attributes pushed inline since the last drain apply to this draw
    SyncDeferred();
    auto stream = g_state.streams[0];
    Set(g_state.streams[0], {UpVertices(vertices, count * stride), stride});
    g_backend->Draw(g_state, prim, 0, count, nullptr);
    Set(g_state.streams[0], stream);
}

// bytes per vertex of the bound format: stream 0 of the FVF or of the shader's declaration
uint32_t VertexBytes() {
    if (g_state.vs_handle & 1) return g_state.vs ? g_state.vs->stream_stride[0] : 0;
    gpu::VertexShader fvf;
    gpu::FvfInputs(g_state.vs_handle, fvf);
    return fvf.stream_stride[0];
}

uint32_t g_push_begin = 0;
}  // namespace

// Begin/End immediate mode. Titles inline Begin and the vertex data calls, so their commands are read from the scratch buffer.
namespace d3d {
namespace {
uint32_t g_im_prim = 0, g_im_mask = 0;  // open primitive, slots it wrote
float g_im_attr[64];                    // 16 slots of x y z w
std::vector<float> g_im_vertices;       // every slot per vertex

// draws the open primitive: the slots it wrote are FLOAT4 inputs of the bound program, or of the fixed pipeline
void EndPrimitive() {
    struct Made {
        std::shared_ptr<gpu::VertexShader> from, vs;
    };
    // the render thread tells shaders apart by pointer: one object per program and slot set, the program kept alive
    static std::map<std::pair<gpu::VertexShader*, uint32_t>, Made> made;
    static std::vector<float> packed;
    uint32_t prim = g_im_prim, count = uint32_t(g_im_vertices.size() / 64), handle = g_state.vs_handle;
    g_im_prim = 0;
    if (!prim || !count) return;
    if (!(handle & 1) && (handle & 0xE) == 4) LOG_FIRST(1, "d3d: pretransformed immediate mode vertices are not supported");
    auto bound = g_state.vs, from = handle & 1 ? bound : nullptr;
    Made& m = made[{from.get(), g_im_mask}];
    if (!m.vs) {
        m.from = from;
        m.vs = std::make_shared<gpu::VertexShader>();
        if (from) {
            m.vs->program = from->program;
            m.vs->program_type = from->program_type;
        }
        for (uint32_t s = 0; s < 16; s++)
            if (g_im_mask >> s & 1) {
                m.vs->inputs[s] = {0, m.vs->stream_stride[0], 0x42};
                m.vs->stream_stride[0] += 16;
            }
    }
    packed.clear();
    for (uint32_t v = 0; v < count; v++)
        for (uint32_t s = 0; s < 16; s++)
            if (g_im_mask >> s & 1) packed.insert(packed.end(), &g_im_vertices[v * 64 + s * 4], &g_im_vertices[v * 64 + s * 4] + 4);
    Set(g_state.vs, m.vs);
    Set(g_state.vs_handle, handle | 1);
    DrawUp(prim, count, packed.data(), m.vs->stream_stride[0]);
    Set(g_state.vs, bound);
    Set(g_state.vs_handle, handle);
}

// reads the commands pushed since the last call: vertex data between a begin and an end is drawn, the rest is skipped
void Walk(uint32_t end) {
    if (!g_walk || end < g_walk || end > g_scratch_push + g_scratch_size) return;
    for (uint32_t p = g_walk; p + 4 <= end;) {
        // Vertex shader constants pushed inline (the library's Constant1Fast and the engine's own
        // writers): [0x41EA4, register + 96] sets the load register, then one or more
        // [(dwords << 18) | 0xB80, data...] packets write from it, the register advancing by one per
        // four dwords. One packet carries at most the 32 dwords of methods 0xB80..0xBFC, so anything
        // longer than eight registers - a skinning palette is tens of them - arrives as several
        // packets and only the first follows a 0x41EA4. Treating the header as mandatory dropped every
        // continuation into the generic path below, which logs and discards it: the world's four
        // register transform survived and skinned characters lost all but their first eight.
        if (R32(p) == kVscPush && p + 8 <= end) {
            g_vsc_reg = R32(p + 4);
            p += 8;
            continue;
        }
        if (uint32_t h = R32(p); (h & 0x1FFC) == kVscMethod && !(h & 0x40000000)) {
            uint32_t n = (h >> 18) & 0x7FF;
            if (p + 4 + n * 4 <= end) {
                StoreVsConstants(g_vsc_reg, n, p + 4);
                g_vsc_reg += n / 4;
                p += 4 + n * 4;
                continue;
            }
        }
        // count in bits 18 to 28, bit 30 keeps the method for every value
        uint32_t head = R32(p), method = head & 0x1FFC, n = (head >> 18) & 0x7FF;
        p += 4;
        if ((head & 0xA0030003) || n * 4 > end - p) break;  // anything but a method header
        for (uint32_t i = 0; i < n; i++, p += 4) {
            uint32_t m = head & 0x40000000 ? method : method + i * 4, v = R32(p);
            float f, *a = g_im_attr;
            bool last = true;  // the slot's last component
            std::memcpy(&f, &v, 4);
            if (m == 0x17FC) {  // begin/end, 0 ends
                if (v) {
                    g_im_prim = v;
                    g_im_mask = 0;
                    g_im_vertices.clear();
                } else {
                    EndPrimitive();
                }
                continue;
            } else if (m >= 0x1880 && m < 0x1900) {  // 2 floats, z = 0 and w = 1
                a += (m - 0x1880) / 8 * 4;
                last = (m - 0x1880) / 4 & 1;
                a[last] = f;
                a[2] = 0;
                a[3] = 1;
            } else if (m >= 0x1940 && m < 0x1980) {  // 4 bytes
                a += (m - 0x1940) / 4 * 4;
                for (int c = 0; c < 4; c++) a[c] = (v >> c * 8 & 0xFF) / 255.0f;
            } else if (m >= 0x1A00 && m < 0x1B00) {  // 4 floats
                a += (m - 0x1A00) / 16 * 4;
                last = ((m - 0x1A00) / 4 & 3) == 3;
                a[(m - 0x1A00) / 4 & 3] = f;
            } else if (m >= 0x1500 && m < 0x150C) {  // position, 3 floats and w = 1
                last = m == 0x1508;
                a[(m - 0x1500) / 4] = f;
                a[3] = 1;
            } else if (m >= 0x1518 && m < 0x1528) {  // position, 4 floats
                last = m == 0x1524;
                a[(m - 0x1518) / 4] = f;
            } else {
                static std::unordered_set<uint32_t> seen;
                if (seen.insert(m).second) LOGI("d3d: Walk drops method %04X (head %08X, %u values)", m, head, n);
                continue;
            }
            if (!g_im_prim) continue;
            g_im_mask |= 1u << (a - g_im_attr) / 4;
            if (a == g_im_attr && last) g_im_vertices.insert(g_im_vertices.end(), g_im_attr, g_im_attr + 64);
        }
    }
    g_walk = end;
}
}  // namespace
}  // namespace d3d

HLE_STDCALL(D3DDevice_DrawVerticesUP, 4) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    HleScope hle;
    DrawUp(a[0], a[1], HostPtr(a[2]), a[3]);
    return 0;
}

// BeginPush(dwords) hands out room for raw hardware commands, EndPush(end) runs what the title wrote there:
// a primitive between two begin/end writes with its vertices inline, any other method as a simple state.
HLE_STDCALL(D3DDevice_BeginPush, 1) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    g_push_begin = ScratchPush(a[0] * 4);
    return g_push_begin;
}

HLE_STDCALL(D3DDevice_EndPush, 1) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    HleScope hle;
    constexpr uint32_t kBeginEnd = 0x17FC, kInlineArray = 0x1818;
    static std::vector<uint32_t> vertices;
    uint32_t prim = 0;
    for (uint32_t p = g_push_begin, end = a[0]; g_push_begin && p + 4 <= end;) {
        // vertex shader constants, same packet as in Walk (see there)
        if (R32(p) == kVscPush && p + 12 <= end) {
            uint32_t reg = R32(p + 4), hdr = R32(p + 8), n = hdr >> 18;
            if ((hdr & 0x1FFC) == kVscMethod && p + 12 + n * 4 <= end) {
                StoreVsConstants(reg, n, p + 12);
                p += 12 + n * 4;
                continue;
            }
        }
        // count in bits 18 to 28, bit 30 keeps the method for every value
        uint32_t head = R32(p), method = head & 0x1FFC, n = (head >> 18) & 0x7FF;
        p += 4;
        if ((head & 0xA0030003) || n * 4 > end - p) {  // anything but a method header
            LOG_FIRST(8, "d3d: EndPush stops at command %08X, %u bytes before the end", head, end - p);
            break;
        }
        for (uint32_t i = 0; i < n; i++, p += 4) {
            uint32_t m = head & 0x40000000 ? method : method + i * 4, v = R32(p);
            if (m == kInlineArray) {
                if (prim) vertices.push_back(v);
            } else if (m != kBeginEnd) {
                if (m >> 2 < g_state.simple.size()) Set(g_state.simple[m >> 2], v);
            } else if (v) {
                prim = v;
                vertices.clear();
            } else if (prim) {
                uint32_t stride = VertexBytes(), bytes = uint32_t(vertices.size()) * 4;
                if (stride && bytes % stride == 0) DrawUp(prim, bytes / stride, vertices.data(), stride);
                else LOG_FIRST(8, "d3d: EndPush primitive %u dropped, %u bytes of vertices of %u", prim, bytes, stride);
                prim = 0;
            }
        }
    }
    g_push_begin = 0;
    ScratchPush();
    return a[0];
}

// frame buffer compression tiles are the console's memory controller's business
HLE_STDCALL(D3DDevice_SetTile, 2) { return 0; }

// the library's Begin pushes the start of the primitive, here it is opened
HLE_STDCALL(D3DDevice_Begin, 1) {
    XDK_REQUIRE("d3d", XDK_KEY(CDevice_Flags));
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    HleScope hle;
    Walk(Dev(kDevPush));
    g_im_prim = a[0];
    g_im_mask = 0;
    g_im_vertices.clear();
    SetDev(XDK_CDevice_Flags, Dev(XDK_CDevice_Flags) | 0x800);
    return 0;
}
// the library's End pushes the end of the primitive, here it is drawn
HLE_STDCALL(D3DDevice_End, 0) {
    XDK_REQUIRE("d3d", XDK_KEY(CDevice_Flags));
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    HleScope hle;
    Walk(Dev(kDevPush));
    EndPrimitive();
    SetDev(XDK_CDevice_Flags, Dev(XDK_CDevice_Flags) & ~0x1800u);
    return 0;
}
