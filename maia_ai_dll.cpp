#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <TlHelp32.h>
#include <d3d9.h>

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <safetyhook.hpp>
#include "backends/imgui_impl_dx9.h"
#include "imgui.h"
#include "kiero.hpp"
#include "kiero_d3d9.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cwchar>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
constexpr std::uintptr_t kGameRootRva = 0x19AAF8;
constexpr std::uintptr_t kDifficultyRva = 0x199D00;
constexpr std::uintptr_t kGenerateValidMovesRva = 0x3D328;
constexpr std::uintptr_t kApplyMoveTemporarilyRva = 0x3C4E0;
constexpr std::uintptr_t kUndoTemporaryMoveRva = 0x3C6E0;
constexpr std::uintptr_t kIsSquareAttackedRva = 0x3DC94;
constexpr std::uintptr_t kAiMoveMidHookRva = 0x6373A;
constexpr std::uintptr_t kAiThreadStartRva = 0x4C6F0;
constexpr std::uintptr_t kBuiltinAiModeRva = 0x199D10;
constexpr std::uintptr_t kSkipBuiltinAiModeOneRva = 0x651C9;
constexpr std::uintptr_t kSkipBuiltinAiModeZeroRva = 0x65232;
constexpr std::uintptr_t kSetDifficultyRva = 0x64B40;
constexpr std::array<std::uint8_t, 6> kSkipBuiltinAiModeOnePatch{0xE9, 0xB8, 0, 0, 0, 0x90};
constexpr std::array<std::uint8_t, 6> kSkipBuiltinAiModeZeroPatch{0xE9, 0x4F, 0, 0, 0, 0x90};
constexpr std::uintptr_t kGameTurnOffset = 0x1C;
constexpr std::uintptr_t kGameAiBoardOffset = 0x50;
constexpr std::uintptr_t kAiBoardSideToMoveOffset = 0x10;
constexpr std::uintptr_t kAiBoardTemporarySideOffset = 0x14;
constexpr std::uintptr_t kAiBoardSideZeroPositionOffset = 0x18;
constexpr std::uintptr_t kAiBoardSideOnePositionOffset = 0x1C;
constexpr std::uintptr_t kAiBoardPieceTypesOffset = 0x20;
constexpr std::uintptr_t kAiBoardPieceColorsOffset = 0x220;
constexpr std::uintptr_t kAiBoardValidMovesOffset = 0xBA8;
constexpr std::uintptr_t kAiBoardValidMoveCountOffset = 0x218A8;
constexpr int kMoveTypePromoteKnight = 9;
constexpr int kMoveTypePromoteQueen = 12;

struct Game;
namespace AI {
struct Board;
}

struct AiMove {
    std::int32_t from, to, type, unk;
};

struct GameMove {
    std::int32_t type, fromX, fromY, toX, toY;
};

using GenerateValidMovesFn = void(__fastcall*)(AI::Board*);
using ApplyMoveTemporarilyFn = void(__fastcall*)(AI::Board*, const AiMove*);
using UndoTemporaryMoveFn = void(__fastcall*)(AI::Board*);
using IsSquareAttackedFn = bool(__fastcall*)(AI::Board*, std::int32_t, std::int32_t, std::uint32_t);

std::uintptr_t g_moduleBase;

std::atomic<float> g_currentWinRate{-1.0F};
std::atomic<float> g_currentDrawRate{-1.0F};

class MaiaRuntime {
public:
    explicit MaiaRuntime(const std::filesystem::path& path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "maia3-game"),
          memoryInfo_(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                                 OrtMemType::OrtMemTypeDefault)) {
        options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
        options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        session_ = std::make_unique<Ort::Session>(env_, path.c_str(), options_);
    }

    GameMove ChooseMove(AI::Board* board, std::span<const AiMove> moves, float elo) {
        auto [tokens, side] = Encode(board);
        constexpr std::array<std::int64_t, 3> tokenShape{1, 64, 12};
        constexpr std::array<std::int64_t, 1> eloShape{1};
        std::array inputs{
            Ort::Value::CreateTensor<float>(memoryInfo_, tokens.data(), tokens.size(),
                                            tokenShape.data(), tokenShape.size()),
            Ort::Value::CreateTensor<float>(memoryInfo_, &elo, 1, eloShape.data(), eloShape.size()),
            Ort::Value::CreateTensor<float>(memoryInfo_, &elo, 1, eloShape.data(), eloShape.size()),
        };
        constexpr std::array<const char*, 3> inputNames{"tokens", "elo_self", "elo_oppo"};
        constexpr std::array<const char*, 1> outputNames{"logits_move"};
        auto outputs = session_->Run(Ort::RunOptions{nullptr}, inputNames.data(), inputs.data(),
                                     inputs.size(), outputNames.data(), outputNames.size());
        const float* logits = outputs[0].GetTensorData<float>();

        const auto best = std::ranges::max_element(moves, {}, [&](const AiMove& move) {
            return logits[Policy(move, side)];
        });
        const AiMove move = *best;
        Observe(board, move, elo);
        return {move.type, move.from & 15, move.from >> 4, move.to & 15, move.to >> 4};
    }

    void Observe(AI::Board* board, const AiMove& move, float elo) {
        reinterpret_cast<ApplyMoveTemporarilyFn>(g_moduleBase + kApplyMoveTemporarilyRva)(board, &move);
        try {
            auto [tokens, side] = Encode(board);
            constexpr std::array<std::int64_t, 3> tokenShape{1, 64, 12};
            constexpr std::array<std::int64_t, 1> eloShape{1};
            std::array inputs{
                Ort::Value::CreateTensor<float>(memoryInfo_, tokens.data(), tokens.size(),
                                                tokenShape.data(), tokenShape.size()),
                Ort::Value::CreateTensor<float>(memoryInfo_, &elo, 1, eloShape.data(), eloShape.size()),
                Ort::Value::CreateTensor<float>(memoryInfo_, &elo, 1, eloShape.data(), eloShape.size()),
            };
            constexpr std::array<const char*, 3> inputNames{"tokens", "elo_self", "elo_oppo"};
            constexpr std::array<const char*, 1> outputNames{"logits_value"};
            auto outputs = session_->Run(Ort::RunOptions{nullptr}, inputNames.data(), inputs.data(),
                                         inputs.size(), outputNames.data(), outputNames.size());
            const float* logits = outputs[0].GetTensorData<float>();
            const float m = std::max({logits[0], logits[1], logits[2]});
            const float loss = std::exp(logits[0] - m), draw = std::exp(logits[1] - m),
                        win = std::exp(logits[2] - m), sum = loss + draw + win;
            reinterpret_cast<UndoTemporaryMoveFn>(g_moduleBase + kUndoTemporaryMoveRva)(board);

            const std::uint8_t mode =
                *reinterpret_cast<volatile const std::uint8_t*>(g_moduleBase + kBuiltinAiModeRva);
            const int playerSide = mode == 0 ? 1 : 0;
            const float playerWin = side == playerSide ? win : loss;
            g_currentWinRate.store(100.0F * playerWin / sum, std::memory_order_relaxed);
            g_currentDrawRate.store(100.0F * draw / sum, std::memory_order_relaxed);
        } catch (...) {
            reinterpret_cast<UndoTemporaryMoveFn>(g_moduleBase + kUndoTemporaryMoveRva)(board);
        }
    }

private:
    static std::pair<std::array<float, 64 * 12>, int> Encode(const AI::Board* board) {
        std::pair<std::array<float, 64 * 12>, int> result{};
        const auto* base = reinterpret_cast<const std::uint8_t*>(board);
        const int side =
            result.second = *reinterpret_cast<const int*>(base + kAiBoardSideToMoveOffset);
        const auto* types = reinterpret_cast<const int*>(base + kAiBoardPieceTypesOffset);
        const auto* colors = reinterpret_cast<const int*>(base + kAiBoardPieceColorsOffset);
        for (int y = 0; y < 8; ++y) {
            for (int x = 0; x < 8; ++x) {
                const int square = y * 16 + x;
                const int type = types[square];
                if (type < 1 || type > 6)
                    continue;
                const int channel = type - 1 + (colors[square] == side ? 0 : 6);
                result.first[((side ? y : 7 - y) * 8 + x) * 12 + channel] = 1.0F;
            }
        }
        return result;
    }

    static int Policy(const AiMove& move, int side) noexcept {
        const int fromX = move.from & 15;
        const int toX = move.to & 15;
        if (move.type >= kMoveTypePromoteKnight && move.type <= kMoveTypePromoteQueen)
            return 4096 + ((fromX * 8 + toX) * 4 + kMoveTypePromoteQueen - move.type);
        const int fromY = side ? move.from >> 4 : 7 - (move.from >> 4);
        const int toY = side ? move.to >> 4 : 7 - (move.to >> 4);
        return (fromY * 8 + fromX) * 64 + toY * 8 + toX;
    }

    Ort::Env env_;
    Ort::SessionOptions options_;
    Ort::MemoryInfo memoryInfo_;
    std::unique_ptr<Ort::Session> session_;
};

constexpr std::size_t kD3D9ResetMethodIndex = 16;
constexpr std::size_t kD3D9PresentMethodIndex = 17;

SafetyHookInline g_d3d9ResetHook;
SafetyHookInline g_d3d9PresentHook;
bool g_imguiInitialized = false;

bool InitializeImGuiD3D9(IDirect3DDevice9* device) noexcept {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.ConfigFlags |= ImGuiConfigFlags_NoMouseCursorChange;

    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 10.0F;
    style.WindowBorderSize = 1.0F;
    style.FrameRounding = 5.0F;
    style.WindowPadding = ImVec2(11.0F, 9.0F);
    style.ItemSpacing = ImVec2(7.0F, 5.0F);
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.045F, 0.052F, 0.070F, 0.94F);
    style.Colors[ImGuiCol_Border] = ImVec4(0.24F, 0.29F, 0.39F, 0.72F);
    style.Colors[ImGuiCol_Text] = ImVec4(0.94F, 0.96F, 1.00F, 1.00F);
    style.Colors[ImGuiCol_TextDisabled] = ImVec4(0.55F, 0.61F, 0.72F, 1.00F);

    if (!ImGui_ImplDX9_Init(device)) {
        ImGui::DestroyContext();
        return false;
    }
    return true;
}

void BeginImGuiD3D9Frame(IDirect3DDevice9* device) noexcept {
    ImGuiIO& io = ImGui::GetIO();

    D3DVIEWPORT9 viewport{};
    if (SUCCEEDED(device->GetViewport(&viewport)))
        io.DisplaySize = ImVec2(static_cast<float>(viewport.Width),
                                static_cast<float>(viewport.Height));

    static LARGE_INTEGER frequency{};
    static LARGE_INTEGER previous{};
    LARGE_INTEGER now{};

    if (frequency.QuadPart == 0)
        ::QueryPerformanceFrequency(&frequency);
    ::QueryPerformanceCounter(&now);

    if (previous.QuadPart != 0 && frequency.QuadPart != 0) {
        const double elapsed =
            static_cast<double>(now.QuadPart - previous.QuadPart) /
            static_cast<double>(frequency.QuadPart);
        io.DeltaTime = elapsed > 0.0 ? static_cast<float>(elapsed) : (1.0F / 60.0F);
    } else {
        io.DeltaTime = 1.0F / 60.0F;
    }
    previous = now;

    ImGui_ImplDX9_NewFrame();
    ImGui::NewFrame();
}

void RenderEvaluationOverlay() noexcept {
    const float winRate = g_currentWinRate.load(std::memory_order_relaxed);
    const float drawRate = g_currentDrawRate.load(std::memory_order_relaxed);

    constexpr ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoDecoration |
        ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoNav |
        ImGuiWindowFlags_NoInputs;

    ImGui::SetNextWindowPos(ImVec2(14.0F, 14.0F), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.94F);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(218.0F, 0.0F));
    ImGui::Begin("##Maia3Evaluation", nullptr, flags);

    if (winRate >= 0.0F && drawRate >= 0.0F) {
        const float lossRate = std::clamp(100.0F - winRate - drawRate, 0.0F, 100.0F);
        const float win01 = std::clamp(winRate * 0.01F, 0.0F, 1.0F);
        const float draw01 = std::clamp(drawRate * 0.01F, 0.0F, 1.0F);
        const float loss01 = std::clamp(lossRate * 0.01F, 0.0F, 1.0F);

        ImGui::TextUnformatted("WIN");
        ImGui::SameLine(150.0F);
        ImGui::Text("%.2f%%", winRate);
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.25F, 0.78F, 0.50F, 1.00F));
        ImGui::ProgressBar(win01, ImVec2(190.0F, 6.0F), "");
        ImGui::PopStyleColor();

        ImGui::Dummy(ImVec2(0.0F, 1.0F));
        ImGui::TextUnformatted("DRAW");
        ImGui::SameLine(150.0F);
        ImGui::Text("%.2f%%", drawRate);
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.88F, 0.67F, 0.25F, 1.00F));
        ImGui::ProgressBar(draw01, ImVec2(190.0F, 6.0F), "");
        ImGui::PopStyleColor();

        ImGui::Dummy(ImVec2(0.0F, 1.0F));
        ImGui::TextUnformatted("LOSS");
        ImGui::SameLine(150.0F);
        ImGui::Text("%.2f%%", lossRate);
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.82F, 0.30F, 0.32F, 1.00F));
        ImGui::ProgressBar(loss01, ImVec2(190.0F, 6.0F), "");
        ImGui::PopStyleColor();
    } else {
        ImGui::TextDisabled("Waiting for evaluation...");
    }

    ImGui::End();
    ImGui::PopStyleVar();
}

HRESULT STDMETHODCALLTYPE HookD3D9Present(IDirect3DDevice9* device, const RECT* sourceRect,
                                          const RECT* destRect, HWND destWindowOverride,
                                          const RGNDATA* dirtyRegion) noexcept {
    if (!g_imguiInitialized)
        g_imguiInitialized = InitializeImGuiD3D9(device);

    if (g_imguiInitialized) {
        BeginImGuiD3D9Frame(device);
        RenderEvaluationOverlay();
        ImGui::Render();
        ImGui_ImplDX9_RenderDrawData(ImGui::GetDrawData());
    }

    return g_d3d9PresentHook.stdcall<HRESULT>(device, sourceRect, destRect, destWindowOverride, dirtyRegion);
}

HRESULT STDMETHODCALLTYPE HookD3D9Reset(IDirect3DDevice9* device,
                                        D3DPRESENT_PARAMETERS* presentationParameters) noexcept {
    if (g_imguiInitialized)
        ImGui_ImplDX9_InvalidateDeviceObjects();

    const HRESULT result = g_d3d9ResetHook.stdcall<HRESULT>(device, presentationParameters);

    if (g_imguiInitialized && SUCCEEDED(result))
        ImGui_ImplDX9_CreateDeviceObjects();

    return result;
}

DWORD WINAPI D3D9OverlayBootstrapThread(LPVOID) noexcept {
    while (!::GetModuleHandleW(L"d3d9.dll"))
        ::Sleep(50);

    try {
        kiero::D3D9Output d3d9;
        if (kiero::locate<kiero::Implementation_D3D9>(nullptr, &d3d9) != kiero::Error_Nil)
            return 0;
        if (d3d9.device_methods.size() <= kD3D9PresentMethodIndex)
            return 0;

        auto resetHook = safetyhook::create_inline(
            d3d9.device_methods[kD3D9ResetMethodIndex],
            reinterpret_cast<void*>(&HookD3D9Reset),
            SafetyHookInline::StartDisabled);
        if (!resetHook)
            return 0;

        auto presentHook = safetyhook::create_inline(
            d3d9.device_methods[kD3D9PresentMethodIndex],
            reinterpret_cast<void*>(&HookD3D9Present),
            SafetyHookInline::StartDisabled);
        if (!presentHook)
            return 0;

        g_d3d9ResetHook = std::move(resetHook);
        g_d3d9PresentHook = std::move(presentHook);

        if (!g_d3d9ResetHook.enable())
            return 0;
        if (!g_d3d9PresentHook.enable()) {
            (void)g_d3d9ResetHook.disable();
            return 0;
        }
    } catch (...) {
    }
    return 0;
}

std::filesystem::path FindMaiaModel(const std::filesystem::path& dir) {
    namespace fs = std::filesystem;
    static constexpr std::array<const wchar_t*, 8> candidates{
        L"maia3-5m.fp32.onnx",
        L"maia3-5m.fp16.onnx",
        L"maia3-23m.fp32.onnx",
        L"maia3-23m.fp16.onnx",
        L"maia3-79m.fp32.onnx",
        L"maia3-79m.fp16.onnx",
        L"maia3-3m-ablation.fp32.onnx",
        L"maia3-3m-ablation.fp16.onnx",
    };

    for (const auto* name : candidates)
        if (const auto modelPath = dir / name; fs::exists(modelPath))
            return modelPath;
    throw std::runtime_error("No supported Maia3 model found");
}

std::unique_ptr<MaiaRuntime> g_runtime;
SafetyHookMid g_aiMoveHook;
SafetyHookInline g_aiThreadStartHook;
SafetyHookInline g_SetDifficultyHook;
SafetyHookInline g_createWindowExWHook;

std::array<std::uint8_t, kSkipBuiltinAiModeOnePatch.size()> g_skipBuiltinAiModeOneOriginal{};
std::array<std::uint8_t, kSkipBuiltinAiModeZeroPatch.size()> g_skipBuiltinAiModeZeroOriginal{};
bool g_skipBuiltinAiOriginalsCaptured = false;
PVOID g_builtinAiModeVeh = nullptr;
DWORD g_mainThreadId = 0;

void HookAiThreadStart(void* self, bool /*startAi*/) noexcept {
    g_aiThreadStartHook.call<void>(self, false);
}

bool HookSetDifficulty(void* self) noexcept {
    g_currentWinRate.store(-1.0F, std::memory_order_relaxed);
    g_currentDrawRate.store(-1.0F, std::memory_order_relaxed);
    return g_SetDifficultyHook.call<bool>(self);
}

HWND WINAPI HookCreateWindowExW(DWORD dwExStyle, LPCWSTR lpClassName, LPCWSTR lpWindowName, DWORD dwStyle,
                                int X, int Y, int nWidth, int nHeight, HWND hWndParent, HMENU hMenu,
                                HINSTANCE hInstance, LPVOID lpParam) noexcept {
    LPCWSTR title = lpWindowName;
    if (lpWindowName && std::wcscmp(lpWindowName, L"Chess Titans") == 0)
        title = L"Chess Titans - Maia 3";

    return g_createWindowExWHook.call<HWND>(dwExStyle, lpClassName, title, dwStyle, X, Y, nWidth, nHeight,
                                              hWndParent, hMenu, hInstance, lpParam);
}

template <class T>
T* At(std::uintptr_t base, std::uintptr_t offset) noexcept {
    return reinterpret_cast<T*>(base + offset);
}

bool WriteExecutableBytes(std::uintptr_t rva, std::span<const std::uint8_t> bytes) noexcept {
    auto* target = At<std::uint8_t>(g_moduleBase, rva);
    DWORD oldProtect{};
    if (!::VirtualProtect(target, bytes.size(), PAGE_EXECUTE_READWRITE, &oldProtect))
        return false;

    std::ranges::copy(bytes, target);
    ::FlushInstructionCache(::GetCurrentProcess(), target, bytes.size());

    DWORD ignored{};
    const BOOL restored = ::VirtualProtect(target, bytes.size(), oldProtect, &ignored);
    return restored != FALSE;
}

void CaptureSkipBuiltinAiOriginalBytes() noexcept {
    if (g_skipBuiltinAiOriginalsCaptured)
        return;

    const auto* modeOneTarget = At<const std::uint8_t>(g_moduleBase, kSkipBuiltinAiModeOneRva);
    const auto* modeZeroTarget = At<const std::uint8_t>(g_moduleBase, kSkipBuiltinAiModeZeroRva);
    std::ranges::copy(std::span{modeOneTarget, g_skipBuiltinAiModeOneOriginal.size()},
                      g_skipBuiltinAiModeOneOriginal.begin());
    std::ranges::copy(std::span{modeZeroTarget, g_skipBuiltinAiModeZeroOriginal.size()},
                      g_skipBuiltinAiModeZeroOriginal.begin());
    g_skipBuiltinAiOriginalsCaptured = true;
}

bool ApplySkipBuiltinAiPatch(std::uint8_t mode) noexcept {
    if (!g_skipBuiltinAiOriginalsCaptured || (mode != 0 && mode != 1))
        return false;

    if (!WriteExecutableBytes(kSkipBuiltinAiModeOneRva, g_skipBuiltinAiModeOneOriginal))
        return false;
    if (!WriteExecutableBytes(kSkipBuiltinAiModeZeroRva, g_skipBuiltinAiModeZeroOriginal))
        return false;

    if (mode == 1)
        return WriteExecutableBytes(kSkipBuiltinAiModeOneRva, kSkipBuiltinAiModeOnePatch);
    return WriteExecutableBytes(kSkipBuiltinAiModeZeroRva, kSkipBuiltinAiModeZeroPatch);
}

LONG CALLBACK BuiltinAiModeVectoredExceptionHandler(PEXCEPTION_POINTERS exceptionInfo) noexcept {
    if (!exceptionInfo || !exceptionInfo->ExceptionRecord || !exceptionInfo->ContextRecord ||
        exceptionInfo->ExceptionRecord->ExceptionCode != EXCEPTION_SINGLE_STEP ||
        ::GetCurrentThreadId() != g_mainThreadId)
        return EXCEPTION_CONTINUE_SEARCH;

    CONTEXT* context = exceptionInfo->ContextRecord;
    if ((context->Dr6 & 0x1) == 0)
        return EXCEPTION_CONTINUE_SEARCH;

    context->Dr6 &= ~static_cast<DWORD_PTR>(0x1);

    const auto mode = *At<volatile const std::uint8_t>(g_moduleBase, kBuiltinAiModeRva);
    if (mode == 0 || mode == 1)
        (void)ApplySkipBuiltinAiPatch(mode);

    return EXCEPTION_CONTINUE_EXECUTION;
}

bool InstallBuiltinAiModeHardwareBreakpoint() noexcept {
    const DWORD processId = ::GetCurrentProcessId();
    const DWORD currentThreadId = ::GetCurrentThreadId();
    HANDLE snapshot = ::CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snapshot == INVALID_HANDLE_VALUE)
        return false;

    HANDLE mainThread = nullptr;
    DWORD mainThreadId = 0;
    ULONGLONG earliestCreationTime = ~ULONGLONG{0};

    THREADENTRY32 entry{};
    entry.dwSize = sizeof(entry);
    if (::Thread32First(snapshot, &entry)) {
        do {
            if (entry.th32OwnerProcessID != processId || entry.th32ThreadID == currentThreadId)
                continue;

            HANDLE thread = ::OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT |
                                             THREAD_SUSPEND_RESUME | THREAD_QUERY_INFORMATION,
                                         FALSE, entry.th32ThreadID);
            if (!thread)
                continue;

            FILETIME creation{}, exit{}, kernel{}, user{};
            if (!::GetThreadTimes(thread, &creation, &exit, &kernel, &user)) {
                ::CloseHandle(thread);
                continue;
            }

            ULARGE_INTEGER creationValue{};
            creationValue.LowPart = creation.dwLowDateTime;
            creationValue.HighPart = creation.dwHighDateTime;
            if (creationValue.QuadPart < earliestCreationTime) {
                if (mainThread)
                    ::CloseHandle(mainThread);
                mainThread = thread;
                mainThreadId = entry.th32ThreadID;
                earliestCreationTime = creationValue.QuadPart;
            } else {
                ::CloseHandle(thread);
            }
        } while (::Thread32Next(snapshot, &entry));
    }
    ::CloseHandle(snapshot);

    if (!mainThread)
        return false;

    if (!g_builtinAiModeVeh) {
        g_builtinAiModeVeh = ::AddVectoredExceptionHandler(
            1, &BuiltinAiModeVectoredExceptionHandler);
        if (!g_builtinAiModeVeh) {
            ::CloseHandle(mainThread);
            return false;
        }
    }

    const DWORD suspendResult = ::SuspendThread(mainThread);
    if (suspendResult == static_cast<DWORD>(-1)) {
        ::RemoveVectoredExceptionHandler(g_builtinAiModeVeh);
        g_builtinAiModeVeh = nullptr;
        ::CloseHandle(mainThread);
        return false;
    }

    CONTEXT context{};
    context.ContextFlags = CONTEXT_DEBUG_REGISTERS;
    bool success = ::GetThreadContext(mainThread, &context) != FALSE;
    if (success) {
        context.Dr0 = static_cast<DWORD_PTR>(g_moduleBase + kBuiltinAiModeRva);
        context.Dr6 = 0;
        context.Dr7 &= ~static_cast<DWORD_PTR>((0x3ull << 0) | (0xFull << 16));
        context.Dr7 |= static_cast<DWORD_PTR>((0x1ull << 0) | (0x1ull << 16));
        success = ::SetThreadContext(mainThread, &context) != FALSE;
    }

    if (success)
        g_mainThreadId = mainThreadId;

    (void)::ResumeThread(mainThread);
    ::CloseHandle(mainThread);

    if (!success) {
        ::RemoveVectoredExceptionHandler(g_builtinAiModeVeh);
        g_builtinAiModeVeh = nullptr;
        g_mainThreadId = 0;
    }
    return success;
}

bool IsLegalMove(AI::Board* board, const AiMove& move) {
    const auto applyMoveTemporarily =
        reinterpret_cast<ApplyMoveTemporarilyFn>(g_moduleBase + kApplyMoveTemporarilyRva);
    const auto undoTemporaryMove =
        reinterpret_cast<UndoTemporaryMoveFn>(g_moduleBase + kUndoTemporaryMoveRva);
    const auto isSquareAttacked =
        reinterpret_cast<IsSquareAttackedFn>(g_moduleBase + kIsSquareAttackedRva);

    applyMoveTemporarily(board, &move);

    bool kingAttacked = false;
    try {
        const auto boardBase = reinterpret_cast<std::uintptr_t>(board);
        const int sideToCheck =
            *At<const int>(boardBase, kAiBoardTemporarySideOffset);
        const std::uintptr_t kingPositionOffset =
            sideToCheck == 0
                ? kAiBoardSideZeroPositionOffset
                : kAiBoardSideOnePositionOffset;
        const int kingPosition =
            *At<const int>(boardBase, kingPositionOffset);

        kingAttacked = isSquareAttacked(
            board,
            kingPosition,
            sideToCheck,
            sideToCheck == 0 ? 1u : 0u);
    } catch (...) {
        undoTemporaryMove(board);
        throw;
    }

    undoTemporaryMove(board);
    return !kingAttacked;
}

void AiMoveMidHook(SafetyHookContext& context) noexcept {
    if (!g_runtime || !context.rdx)
        return;

    auto* game = *At<Game*>(g_moduleBase, kGameRootRva);
    if (!game)
        return;

    const auto gameBase = reinterpret_cast<std::uintptr_t>(game);
    const int turn = *At<const std::int32_t>(gameBase, kGameTurnOffset);
    if (turn != 0 && turn != 1)
        return;

    const std::uint8_t mode = *At<volatile const std::uint8_t>(g_moduleBase, kBuiltinAiModeRva);
    if (mode != 0 && mode != 1)
        return;
    const int maiaSide = static_cast<int>(mode);

    auto* board = At<AI::Board>(gameBase, kGameAiBoardOffset);
    const float elo = 600.0F + (2000.0F / 9.0F) *
        std::clamp(*At<const int>(g_moduleBase, kDifficultyRva), 0, 9);

    if (turn != maiaSide) {
        const auto& move = *reinterpret_cast<const GameMove*>(context.rdx);
        g_runtime->Observe(board,
                           {(move.fromY << 4) + move.fromX,
                            (move.toY << 4) + move.toX,
                            move.type,
                            0},
                           elo);
        return;
    }

    reinterpret_cast<GenerateValidMovesFn>(g_moduleBase + kGenerateValidMovesRva)(board);
    const auto boardBase = reinterpret_cast<std::uintptr_t>(board);
    auto* moves = At<AiMove>(boardBase, kAiBoardValidMovesOffset);
    const int count = *At<const int>(boardBase, kAiBoardValidMoveCountOffset);
    std::vector<AiMove> legalMoves;
    legalMoves.reserve(static_cast<std::size_t>(count));
    for (const auto& move : std::span{moves, static_cast<std::size_t>(count)})
        if (IsLegalMove(board, move))
            legalMoves.push_back(move);

    *reinterpret_cast<GameMove*>(context.rdx) = g_runtime->ChooseMove(board, legalMoves, elo);
}

DWORD WINAPI MaiaBootstrapThread(LPVOID self) noexcept {
    try {
        std::array<wchar_t, 32768> path{};
        const DWORD length = ::GetModuleFileNameW(static_cast<HMODULE>(self), path.data(), path.size());
        if (!length || length >= path.size())
            return 0;
        const HMODULE module = ::GetModuleHandleW(nullptr);
        if (!module)
            return 0;
        g_moduleBase = reinterpret_cast<std::uintptr_t>(module);

        auto aiThreadStartHook = safetyhook::create_inline(
            reinterpret_cast<void*>(g_moduleBase + kAiThreadStartRva),
            reinterpret_cast<void*>(&HookAiThreadStart),
            SafetyHookInline::StartDisabled);
        if (!aiThreadStartHook)
            return 0;
        g_aiThreadStartHook = std::move(aiThreadStartHook);
        if (!g_aiThreadStartHook.enable())
            return 0;

        auto SetDifficultyHook = safetyhook::create_inline(
            reinterpret_cast<void*>(g_moduleBase + kSetDifficultyRva),
            reinterpret_cast<void*>(&HookSetDifficulty),
            SafetyHookInline::StartDisabled);
        if (!SetDifficultyHook)
            return 0;
        g_SetDifficultyHook = std::move(SetDifficultyHook);
        if (!g_SetDifficultyHook.enable())
            return 0;

        auto createWindowHook = safetyhook::create_inline(
            reinterpret_cast<void*>(&::CreateWindowExW),
            reinterpret_cast<void*>(&HookCreateWindowExW));
        if (!createWindowHook)
            return 0;
        g_createWindowExWHook = std::move(createWindowHook);
        const auto modelPath = FindMaiaModel(std::filesystem::path(path.data()).parent_path());
        g_runtime = std::make_unique<MaiaRuntime>(modelPath);
        auto hook = safetyhook::create_mid(
            reinterpret_cast<void*>(g_moduleBase + kAiMoveMidHookRva), &AiMoveMidHook,
            SafetyHookMid::StartDisabled);
        if (!hook)
            return 0;
        g_aiMoveHook = std::move(hook);
        if (!g_aiMoveHook.enable())
            return 0;
        CaptureSkipBuiltinAiOriginalBytes();
        if (!InstallBuiltinAiModeHardwareBreakpoint())
            return 0;

        const auto initialBuiltinAiMode =
            *At<const std::uint8_t>(g_moduleBase, kBuiltinAiModeRva);
        if (!ApplySkipBuiltinAiPatch(initialBuiltinAiMode))
            return 0;
    } catch (...) {
    }
    return 0;
}
}

extern "C" __declspec(dllexport) HRESULT SLGetWindowsInformationDWORD(PCWSTR, DWORD* value) {
    *value = 1;
    return S_OK;
}

BOOL APIENTRY DllMain(HMODULE module, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        ::DisableThreadLibraryCalls(module);
        if (const HANDLE thread = ::CreateThread(nullptr, 0, &MaiaBootstrapThread, module, 0, nullptr))
            ::CloseHandle(thread);
        if (const HANDLE overlayThread = ::CreateThread(nullptr, 0, &D3D9OverlayBootstrapThread, nullptr, 0, nullptr))
            ::CloseHandle(overlayThread);
    }
    return TRUE;
}
