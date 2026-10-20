// The host on Linux: SDL3 window, event loop and boot; the game is described
// by app::kGame exactly as on Windows (app/main.cpp). The whole file is the
// non-Windows counterpart of main.cpp and defines the same app:: symbols, so
// exactly one of the two is ever linked (see cmake/XbeRecomp.cmake).
#if !defined(_WIN32)

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>

#include <imgui.h>
#include <backends/imgui_impl_sdl3.h>
#include <backends/imgui_impl_vulkan.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <string>
#include <thread>

#include <unistd.h>
#include <ucontext.h>

#include "app/app.h"
#include "app/platform.h"
#include "common/input_kbm.h"  // kbm::NoteMouse (D11 mouse cache)
#include "core/input.h"
#include "core/log.h"
#include "gpu/backend.h"  // gpu::g_capture (F3)
#include "gpu/vulkan.h"
#include "image_symbols.h"
#include "kernel/kernel.h"
#include "kernel/memory.h"

namespace app {

namespace {

SDL_Window* g_window = nullptr;
char** g_argv = nullptr;
int g_argc = 0;
// Written on the window thread (creation, resize events, ApplyNow), read on
// any thread: SDL video calls belong to the main thread, guest threads do not.
std::atomic<uint32_t> g_client_w{0}, g_client_h{0};
std::atomic<bool> g_focus{false};
std::atomic<bool> g_imgui_vulkan{false};
// Cross-thread requests to the window thread, mirroring main.cpp's
// kApplyDisplay / kOpenWindow window messages.
Uint32 g_apply_event = 0, g_open_event = 0;

std::string Narrow(const wchar_t* wide) {
    std::string out;
    if (!wide) return out;
    for (const wchar_t* p = wide; *p; p++) {
        char32_t c = char32_t(*p);  // wchar_t is one UTF-32 code unit on Linux
        if (c < 0x80) {
            out += char(c);
        } else if (c < 0x800) {
            out += char(0xC0 | (c >> 6));
            out += char(0x80 | (c & 0x3F));
        } else if (c < 0x10000) {
            out += char(0xE0 | (c >> 12));
            out += char(0x80 | ((c >> 6) & 0x3F));
            out += char(0x80 | (c & 0x3F));
        } else {
            out += char(0xF0 | (c >> 18));
            out += char(0x80 | ((c >> 12) & 0x3F));
            out += char(0x80 | ((c >> 6) & 0x3F));
            out += char(0x80 | (c & 0x3F));
        }
    }
    return out;
}

bool Relaunched() {
    for (int i = 1; i < g_argc; i++)
        if (g_argv[i] && std::strstr(g_argv[i], "--relaunched")) return true;
    return false;
}

// D11: window thread only. SDL_GetMouseState is main-thread-only, so it is sampled
// here, in the event loop, and published through kbm::NoteMouse's atomics; guest
// polling threads read that cache and never call SDL_GetMouseState themselves.
void CacheMouseState() {
    float x = 0.0f, y = 0.0f;
    SDL_MouseButtonFlags buttons = SDL_GetMouseState(&x, &y);
    kbm::NoteMouse(x, y, (buttons & SDL_BUTTON_MASK(SDL_BUTTON_RIGHT)) != 0,
                   (buttons & SDL_BUTTON_MASK(SDL_BUTTON_MIDDLE)) != 0);
}

void CacheClientSize() {
    if (!g_window) return;
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(g_window, &w, &h);
    g_client_w = uint32_t(std::max(w, 0));
    g_client_h = uint32_t(std::max(h, 0));
}

// F3, as on Windows: dump every guest thread's stack and capture the frame.
void DumpThreads() {
    gpu::g_capture = true;
    for (auto& t : kernel::Threads())
        if (t->ctx) {
            LOGI("thread %X", t->id);
            kernel::LogGuestStack(*t->ctx, 32);
        }
}

// No minidump on Linux: the guest registers and return addresses go to the
// log, then the process dies with the signal, as the Windows filter leaves
// it to the system after writing its dump.
// No minidump on Linux: the guest registers and return addresses go to the
// log, then the process dies with the signal, as the Windows filter leaves
// it to the system after writing its dump. The fault address is logged too:
// a guest register dump alone cannot say whether a bad pointer is guest
// memory (and therefore the title's own fault) or a host pointer the runtime
// handed back, and two of this port's early crashes needed that distinction.
// A guest address is printed as its offset into the 4 GB guest reservation.
// Requires SA_SIGINFO, so it is installed with sigaction below rather than
// std::signal.
void CrashHandler(int sig, siginfo_t* info, void* uc) {
    const void* at = info ? info->si_addr : nullptr;
    const uint8_t* base = mem::Base();
    if (at && base) {
        const uint8_t* p = static_cast<const uint8_t*>(at);
        if (p >= base && p < base + (size_t{1} << 32))
            LOGE("fault at guest address %08X (host %p)", uint32_t(p - base), at);
    }
    if (uc) LOGE("host faulting instruction %p", reinterpret_cast<void*>(uintptr_t(reinterpret_cast<ucontext_t*>(uc)->uc_mcontext.gregs[REG_RIP])));
    if (kernel::Thread* t = kernel::Current(); t && t->ctx) {
        LOGE("signal %d on guest thread %X", sig, t->id);
        kernel::LogGuestStack(*t->ctx, 256);
    } else {
        LOGE("signal %d", sig);
    }
    logx::Flush();
    struct sigaction dfl{};
    dfl.sa_handler = SIG_DFL;
    sigaction(sig, &dfl, nullptr);
    std::raise(sig);
    _exit(128 + sig);
}

// Window thread only: put the settings' display mode on the SDL window.
// Visibility is preserved: a window held back for a splash or handover
// stays hidden until OpenGameWindow shows it.
void ApplyNow() {
    if (!g_window) return;
    const Settings& s = Get();
    Mode out = Output();
    platform::Size desk = platform::DesktopSize();
    bool hidden = (SDL_GetWindowFlags(g_window) & SDL_WINDOW_HIDDEN) != 0;
    const char* mode_name = s.display_mode == DisplayMode::Fullscreen ? "fullscreen"
                            : s.display_mode == DisplayMode::Borderless ? "borderless" : "windowed";
    bool mode_set = true;
    if (s.display_mode == DisplayMode::Fullscreen) {
        SDL_DisplayMode mode{};
        mode.w = int(s.width ? s.width : desk.width);
        mode.h = int(s.height ? s.height : desk.height);
        mode_set = SDL_SetWindowFullscreenMode(g_window, (s.width && s.height) ? &mode : nullptr);
        SDL_SetWindowFullscreen(g_window, true);
    } else if (s.display_mode == DisplayMode::Borderless) {
        SDL_SetWindowFullscreen(g_window, false);
        SDL_SetWindowBordered(g_window, false);
        SDL_Rect bounds{};
        if (SDL_GetDisplayBounds(SDL_GetDisplayForWindow(g_window), &bounds)) {
            SDL_SetWindowPosition(g_window, bounds.x, bounds.y);
            SDL_SetWindowSize(g_window, bounds.w, bounds.h);
        } else {
            SDL_SetWindowSize(g_window, int(desk.width), int(desk.height));
        }
    } else {  // Windowed
        SDL_SetWindowFullscreen(g_window, false);
        SDL_SetWindowBordered(g_window, true);
        SDL_SetWindowSize(g_window, int(std::max(out.width, 1u)), int(std::max(out.height, 1u)));
        SDL_SetWindowPosition(g_window, SDL_WINDOWPOS_CENTERED_DISPLAY(SDL_GetDisplayForWindow(g_window)),
                              SDL_WINDOWPOS_CENTERED_DISPLAY(SDL_GetDisplayForWindow(g_window)));
    }
    if (hidden) SDL_HideWindow(g_window);
    CacheClientSize();
    // What was asked for, and what the window reports right after asking. On Wayland a
    // size change lands through the compositor's configure, so the drawable here can
    // still be the old size: the SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED log below and the
    // backend's "swapchain %ux%u (window drawable ...)" line are the settled answer.
    LOGI("display: %s, settings %ux%u, output %ux%u, fullscreen mode set %d, window drawable now %ux%u",
         mode_name, s.width, s.height, out.width, out.height, int(mode_set),
         g_client_w.load(std::memory_order_relaxed), g_client_h.load(std::memory_order_relaxed));
}

struct OpenRequest {
    bool apply;
    SDL_Semaphore* done;
};

void DoOpenGameWindow(bool apply) {
    CloseSplash();
    if (apply) ApplyNow();
    SDL_ShowWindow(g_window);
    SDL_RaiseWindow(g_window);
}

// The Vulkan backend owns the device; once its Init has run (on a guest
// thread), finish the ImGui Vulkan backend on this thread under UiLock.
void TryInitImGuiVulkan() {
    if (g_imgui_vulkan.load(std::memory_order_relaxed) || !ImGui::GetCurrentContext()) return;
    const gpu::VulkanHandles& h = gpu::GetVulkanHandles();
    if (h.device == VK_NULL_HANDLE || h.renderPass == VK_NULL_HANDLE) return;
    std::lock_guard lock(UiLock());
    if (g_imgui_vulkan.load(std::memory_order_relaxed)) return;
    ImGui_ImplVulkan_InitInfo info{};
    info.Instance = h.instance;
    info.PhysicalDevice = h.physicalDevice;
    info.Device = h.device;
    info.QueueFamily = h.graphicsQueueFamily;
    info.Queue = h.graphicsQueue;
    info.PipelineCache = h.pipelineCache;
    info.DescriptorPool = h.descriptorPool;
    info.RenderPass = h.renderPass;
    info.Subpass = 0;
    info.MinImageCount = h.minImageCount;
    info.ImageCount = h.imageCount;
    info.MSAASamples = h.msaaSamples;
    info.UseDynamicRendering = false;
    ImGui_ImplVulkan_Init(&info);
    g_imgui_vulkan.store(true, std::memory_order_relaxed);
    LOGI("imgui: Vulkan backend ready (%u images)", h.imageCount);
}

bool HandleEvent(const SDL_Event& e) {  // window thread; true: quit
    // ImGui first, under the same lock the renderer builds its frame with,
    // as ImGui_ImplWin32_WndProcHandler is in main.cpp's WndProc.
    if (ImGui::GetCurrentContext()) {
        std::lock_guard lock(UiLock());
        ImGui_ImplSDL3_ProcessEvent(&e);
    }
    // GameInfo::message is a Win32-message hook; there are no window
    // messages to offer it here, and games leave it null off Windows.
    if (e.type == g_apply_event) {
        ApplyNow();
        return false;
    }
    if (e.type == g_open_event) {
        auto* req = static_cast<OpenRequest*>(e.user.data1);
        DoOpenGameWindow(req->apply);
        SDL_SignalSemaphore(req->done);
        return false;
    }
    if (e.type == SDL_EVENT_QUIT) return true;
    if (e.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED && e.window.windowID == SDL_GetWindowID(g_window)) return true;
    // Pixel size / display scale changes refresh the cached drawable size; the
    // Vulkan backend polls it (platform::ClientSize) in ResizeSwapIfNeeded and
    // recreates the swapchain at the new size - on Wayland the swapchain
    // extent is the window size, so this event is what drives fullscreen.
    if (e.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED || e.type == SDL_EVENT_WINDOW_RESIZED ||
        e.type == SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED) {
        uint32_t was_w = g_client_w.load(std::memory_order_relaxed), was_h = g_client_h.load(std::memory_order_relaxed);
        CacheClientSize();
        if (g_client_w.load(std::memory_order_relaxed) != was_w || g_client_h.load(std::memory_order_relaxed) != was_h)
            LOGI("display: window drawable now %ux%u", g_client_w.load(std::memory_order_relaxed),
                 g_client_h.load(std::memory_order_relaxed));
    }
    if (e.type == SDL_EVENT_WINDOW_FOCUS_GAINED) g_focus = true;
    if (e.type == SDL_EVENT_WINDOW_FOCUS_LOST) g_focus = false;
    if (e.type == SDL_EVENT_KEY_DOWN && !e.key.repeat) {
        if (e.key.scancode == SDL_SCANCODE_F3) {
            DumpThreads();
        } else if (e.key.key == SDLK_RETURN && (e.key.mod & SDL_KMOD_ALT)) {
            // Alt+Enter: desktop fullscreen and back, without rewriting the
            // saved display mode (the F1 panel owns that setting).
            SDL_SetWindowFullscreen(g_window, (SDL_GetWindowFlags(g_window) & SDL_WINDOW_FULLSCREEN) == 0);
            CacheClientSize();
        }
        // F1/F2 need no host handling: ImGui has the key from ProcessEvent
        // above and the overlay toggles itself in GameInfo::ui.
    }
    return false;
}

}  // namespace

std::filesystem::path ExeDir() {
    char self[4096] = {};
    ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (n <= 0) return std::filesystem::current_path();
    return std::filesystem::path(std::string(self, size_t(n))).parent_path();
}

int Instance() {
    static const int n = [] {
        auto clamp = [](int v) { return std::max(0, std::min(8, v)); };
        for (int i = 1; i < g_argc; i++) {
            if (!g_argv[i]) continue;
            // D10: the documented forms are "--instance 2" (two argv entries) and
            // "--instance=2"; the old strstr(p + 10) read the empty string / "=2"
            // for both and silently yielded instance 0. The glued "--instance2"
            // that used to be the only working form is still accepted.
            if (std::strcmp(g_argv[i], "--instance") == 0) {
                if (i + 1 < g_argc && g_argv[i + 1]) return clamp(std::atoi(g_argv[i + 1]));
                return 0;
            }
            if (std::strncmp(g_argv[i], "--instance=", 11) == 0) return clamp(std::atoi(g_argv[i] + 11));
            if (std::strncmp(g_argv[i], "--instance", 10) == 0 && g_argv[i][10] >= '0' && g_argv[i][10] <= '9')
                return clamp(std::atoi(g_argv[i] + 10));
        }
        return 0;
    }();
    return n;
}

void* Window() { return g_window; }

std::recursive_mutex& UiLock() {
    static std::recursive_mutex m;
    return m;
}

void ApplyDisplay() {
    if (!g_window || !g_apply_event) return;
    SDL_Event e{};
    e.type = g_apply_event;
    SDL_PushEvent(&e);
}

void OpenGameWindow(bool apply) {
    if (!g_window) return;
    if (!g_open_event) {
        DoOpenGameWindow(apply);
        return;
    }
    // SendMessage on Windows: returns once the window is up.
    SDL_Semaphore* done = SDL_CreateSemaphore(0);
    OpenRequest req{apply, done};
    SDL_Event e{};
    e.type = g_open_event;
    e.user.data1 = &req;
    if (!SDL_PushEvent(&e)) {
        DoOpenGameWindow(apply);
        SDL_DestroySemaphore(done);
        return;
    }
    SDL_WaitSemaphore(done);
    SDL_DestroySemaphore(done);
}

void Pace() {
    using clock = std::chrono::steady_clock;
    static clock::time_point next = clock::now();
    uint32_t cap = Get().fps_cap;
    auto now = clock::now();
    if (!cap) {
        next = now;
        return;
    }
    auto period = std::chrono::nanoseconds(1000000000ull / cap);
    next += period;
    if (next < now - period) next = now;
    while (clock::now() + std::chrono::milliseconds(2) < next) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    while (clock::now() < next) std::this_thread::yield();
}

namespace platform {

NativeWindow NativeHandle() { return g_window; }

SDL_Window* SdlWindow() { return g_window; }

Size ClientSize() { return {g_client_w.load(std::memory_order_relaxed), g_client_h.load(std::memory_order_relaxed)}; }

Size DesktopSize() {
    SDL_DisplayID display = g_window ? SDL_GetDisplayForWindow(g_window) : SDL_GetPrimaryDisplay();
    if (const SDL_DisplayMode* mode = SDL_GetCurrentDisplayMode(display)) return {uint32_t(mode->w), uint32_t(mode->h)};
    return {1280, 720};
}

bool HasFocus() { return g_focus.load(std::memory_order_relaxed); }

bool IsVisible() { return g_window && (SDL_GetWindowFlags(g_window) & SDL_WINDOW_HIDDEN) == 0; }

bool IsMinimized() { return g_window && (SDL_GetWindowFlags(g_window) & SDL_WINDOW_MINIMIZED) != 0; }

void SetVisible(bool visible) {
    if (!g_window) return;
    if (visible) SDL_ShowWindow(g_window);
    else SDL_HideWindow(g_window);
}

void Raise() {
    if (g_window) SDL_RaiseWindow(g_window);
}

void SetCursorVisible(bool visible) {
    if (visible) SDL_ShowCursor();
    else SDL_HideCursor();
}

bool CursorVisible() { return SDL_CursorVisible(); }

HdrInfo QueryHdr() {
    // No HDR swap chain exists in the Vulkan build; say so, never pretend.
    return {false, false, 0, "HDR is not available on Linux"};
}

}  // namespace platform

}  // namespace app

int main(int argc, char** argv) {
    app::g_argc = argc;
    app::g_argv = argv;
    // SA_SIGINFO so CrashHandler receives the fault address (see above); sigaction rather than
    // std::signal, because only the former can ask for it.
    struct sigaction crash{};
    crash.sa_sigaction = &app::CrashHandler;
    crash.sa_flags = SA_SIGINFO;
    for (int sig : {SIGABRT, SIGSEGV, SIGBUS, SIGILL, SIGFPE}) sigaction(sig, &crash, nullptr);
    // One log a session: a relaunch into another image carries on its predecessor's.
    bool relaunched = app::Relaunched();
    std::string log = app::kGame.log;
    if (app::Instance() > 1) log += "-" + std::to_string(app::Instance());
    logx::Init((app::ExeDir() / "logs").string(), log, relaunched);
    app::Load();
    if (app::kGame.init) app::kGame.init();
    if (relaunched && !logx::kVerbose) LOGR("Relaunched: %s", IMAGE_XBE);
    else app::LogSystem();

    app::GameFiles game;
    if (!app::FindGame(game)) return 1;

    bool splash = app::OpenSplash(), handover = app::StartHandover();

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        LOGE("sdl: video failed to start: %s", SDL_GetError());
        return 1;
    }
    app::g_apply_event = SDL_RegisterEvents(2);
    app::g_open_event = app::g_apply_event + 1;

    app::Mode out = app::Output();
    SDL_WindowFlags flags = SDL_WINDOW_VULKAN;
    if (app::Get().display_mode == app::DisplayMode::Windowed) flags |= SDL_WINDOW_RESIZABLE;
    else if (app::Get().display_mode == app::DisplayMode::Borderless) flags |= SDL_WINDOW_BORDERLESS;
    if (splash || handover) flags |= SDL_WINDOW_HIDDEN;
    std::string title = app::Narrow(app::kGame.title);
    app::g_window = SDL_CreateWindow(title.c_str(), int(std::max(out.width, 1u)), int(std::max(out.height, 1u)), flags);
    if (!app::g_window) {
        LOGE("sdl: window failed: %s", SDL_GetError());
        return 1;
    }
    SDL_SetWindowPosition(app::g_window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
    app::CacheClientSize();
    app::g_focus = SDL_GetKeyboardFocus() == app::g_window;
    // HDR is honestly off on this platform, from the start.
    app::SetHdr({false, false, 0, "HDR is not available on Linux"});

    // Vulkan handshake: the backend owns the instance; SDL names the
    // extensions it needs, the backend creates it, SDL makes the surface.
    uint32_t ext_count = 0;
    char const* const* exts = SDL_Vulkan_GetInstanceExtensions(&ext_count);
    if (!exts) {
        LOGE("sdl: no Vulkan instance extensions: %s", SDL_GetError());
        return 1;
    }
    gpu::SetVulkanInstanceExtensions(exts, ext_count);
    VkInstance instance = gpu::VulkanInstance();
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (instance == VK_NULL_HANDLE || !SDL_Vulkan_CreateSurface(app::g_window, instance, nullptr, &surface)) {
        LOGE("vulkan: surface failed: %s", SDL_GetError());
        return 1;
    }
    gpu::SetVulkanSurface(instance, surface);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;
    ImGui_ImplSDL3_InitForVulkan(app::g_window);
    // ImGui_ImplVulkan_Init waits for the backend's device: TryInitImGuiVulkan
    // in the loop below, once gpu::GetVulkanHandles() is valid.

    input::Init();
    if (app::kGame.window) app::kGame.window(app::g_window);
    std::string hdd = app::Instance() > 1 ? "hdd" + std::to_string(app::Instance()) : "hdd";
    if (app::Instance()) SDL_SetWindowTitle(app::g_window, (title + " [" + std::to_string(app::Instance()) + "]").c_str());
    // Behind a splash the window stays hidden until the game's first frame;
    // handed over, it is set up hidden.
    if (handover) {
        app::ApplyNow();
    } else if (!splash) {
        app::ApplyNow();
        SDL_ShowWindow(app::g_window);
    }
    if (!kernel::Boot(std::move(game.xbe), IMAGE_XBE, game.dir.string(), (app::ExeDir() / hdd).string())) {
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, app::Narrow(app::kGame.short_title).c_str(),
                                 "The game failed to start, see the log in the logs folder.", app::g_window);
        return 1;
    }

    bool quit = false;
    while (!quit) {
        SDL_Event e;
        if (SDL_WaitEventTimeout(&e, 16)) {
            if (app::HandleEvent(e)) quit = true;
            while (!quit && SDL_PollEvent(&e))
                if (app::HandleEvent(e)) quit = true;
        }
        app::CacheMouseState();  // D11: sample SDL mouse state here, on the window thread
        app::TryInitImGuiVulkan();
    }
    // Vulkan teardown handshake (the one main_sdl change in the present fix): the
    // render thread records ImGui draw data into the backend's command buffers, so
    // ImGui's pipeline can still be in flight here. Wait for the device to go idle
    // before ImGui destroys it - validation otherwise reports vkDestroyPipeline
    // on a pipeline still in use by a command buffer at shutdown.
    if (app::g_imgui_vulkan.load(std::memory_order_relaxed)) {
        const gpu::VulkanHandles& vh = gpu::GetVulkanHandles();
        if (vh.device != VK_NULL_HANDLE) vkDeviceWaitIdle(vh.device);
        ImGui_ImplVulkan_Shutdown();
    }
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    // The instance and surface belong to the Vulkan backend and die with it
    // in kernel::Shutdown; the window is the process's, exiting with it.
    logx::Flush();
    kernel::Shutdown(0);
}

#endif  // !defined(_WIN32)
