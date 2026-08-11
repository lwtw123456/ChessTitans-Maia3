#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <safetyhook.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cwchar>
#include <filesystem>
#include <memory>
#include <span>
#include <utility>

namespace {

constexpr std::uintptr_t kGameRootRva = 0x19AAF8;
constexpr std::uintptr_t kDifficultyRva = 0x199D00;
constexpr std::uintptr_t kGenerateValidMovesRva = 0x3D328;
constexpr std::uintptr_t kAiMoveMidHookRva = 0x6373A;
constexpr std::uintptr_t kSkipBuiltinAiRva = 0x651C9;
constexpr std::array<std::uint8_t, 6> kSkipBuiltinAiPatch{0xE9, 0xB8, 0, 0, 0, 0x90};
constexpr std::uintptr_t kGameTurnOffset = 0x1C;
constexpr std::uintptr_t kGameAIBoardOffset = 0x50;
constexpr std::uintptr_t kAIBoardSideToMoveOffset = 0x10;
constexpr std::uintptr_t kAIBoardPieceTypesOffset = 0x20;
constexpr std::uintptr_t kAIBoardPieceColorsOffset = 0x220;
constexpr std::uintptr_t kAIBoardValidMovesOffset = 0xBA8;
constexpr std::uintptr_t kAIBoardValidMoveCountOffset = 0x218A8;
constexpr int kMoveTypePromoteKnight = 9;
constexpr int kMoveTypePromoteQueen = 12;

struct Game;
namespace AI {
struct Board;
}

struct AIMove {
    std::int32_t from, to, type, unk;
};

struct GameMove {
    std::int32_t type, fromX, fromY, toX, toY;
};

using GenerateValidMovesFn = void(__fastcall*)(AI::Board*);

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

    GameMove ChooseMove(const AI::Board* board, std::span<const AIMove> moves, float elo) {
        auto [tokens, side] = Encode(board);
        constexpr std::array<std::int64_t, 3> tokenShape{1, 64, 12};
        constexpr std::array<std::int64_t, 1> eloShape{1};
        std::array inputs{
            Ort::Value::CreateTensor<float>(memoryInfo_, tokens.data(), tokens.size(),
                                            tokenShape.data(), tokenShape.size()),
            Ort::Value::CreateTensor<float>(memoryInfo_, &elo, 1, eloShape.data(), eloShape.size()),
            Ort::Value::CreateTensor<float>(memoryInfo_, &elo, 1, eloShape.data(), eloShape.size())
        };
        constexpr const char* inputNames[]{"tokens", "elo_self", "elo_oppo"};
        constexpr const char* outputNames[]{"logits_move", "logits_value"};
        auto outputs = session_->Run(Ort::RunOptions{nullptr}, inputNames, inputs.data(),
                                     inputs.size(), outputNames, std::size(outputNames));
        const float* logits = outputs[0].GetTensorData<float>();
        const auto best = std::ranges::max_element(moves, {}, [&](const AIMove& move) {
            return logits[Policy(move, side)];
        });
        return {best->type, best->from & 15, best->from >> 4, best->to & 15, best->to >> 4};
    }

private:
    static std::pair<std::array<float, 64 * 12>, int> Encode(const AI::Board* board) {
        std::pair<std::array<float, 64 * 12>, int> result{};
        const auto* base = reinterpret_cast<const std::uint8_t*>(board);
        const int side =
            result.second = *reinterpret_cast<const int*>(base + kAIBoardSideToMoveOffset);
        const auto* types = reinterpret_cast<const int*>(base + kAIBoardPieceTypesOffset);
        const auto* colors = reinterpret_cast<const int*>(base + kAIBoardPieceColorsOffset);
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

    static int Policy(const AIMove& move, int side) noexcept {
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

std::filesystem::path FindMaiaModel(const std::filesystem::path& dir)
{
    namespace fs = std::filesystem;

	static constexpr const wchar_t* candidates[] = {
		L"maia3-5m.fp32.onnx",
		L"maia3-5m.fp16.onnx",
		L"maia3-23m.fp32.onnx",
		L"maia3-23m.fp16.onnx",
		L"maia3-79m.fp32.onnx",
		L"maia3-79m.fp16.onnx",
		L"maia3-3m-ablation.fp32.onnx",
		L"maia3-3m-ablation.fp16.onnx",
	};

    for (const auto* name : candidates) {
        fs::path modelPath = dir / name;

        if (fs::exists(modelPath)) {
            return modelPath;
        }
    }

    throw std::runtime_error("No supported Maia3 model found");
}

std::uintptr_t g_moduleBase;
std::unique_ptr<MaiaRuntime> g_runtime;
SafetyHookMid g_aiMoveHook;
SafetyHookInline g_createWindowExWHook;

HWND WINAPI HookCreateWindowExW(
    DWORD dwExStyle,
    LPCWSTR lpClassName,
    LPCWSTR lpWindowName,
    DWORD dwStyle,
    int X,
    int Y,
    int nWidth,
    int nHeight,
    HWND hWndParent,
    HMENU hMenu,
    HINSTANCE hInstance,
    LPVOID lpParam) noexcept
{
    LPCWSTR title = lpWindowName;

    if (lpWindowName &&
        std::wcscmp(lpWindowName, L"Chess Titans") == 0)
    {
        title = L"Chess Titans - Maia 3";
    }

    return g_createWindowExWHook.call<HWND>(
        dwExStyle,
        lpClassName,
        title,
        dwStyle,
        X,
        Y,
        nWidth,
        nHeight,
        hWndParent,
        hMenu,
        hInstance,
        lpParam);
}

template <class T>
T* At(std::uintptr_t base, std::uintptr_t offset) noexcept {
    return reinterpret_cast<T*>(base + offset);
}

void AiMoveMidHook(SafetyHookContext& context) noexcept {
    auto* game = *At<Game*>(g_moduleBase, kGameRootRva);
    const auto gameBase = reinterpret_cast<std::uintptr_t>(game);
    if (*At<const std::int32_t>(gameBase, kGameTurnOffset) != 1)
        return;
    auto* board = At<AI::Board>(gameBase, kGameAIBoardOffset);
    reinterpret_cast<GenerateValidMovesFn>(g_moduleBase + kGenerateValidMovesRva)(board);
    const auto boardBase = reinterpret_cast<std::uintptr_t>(board);
    auto* moves = At<AIMove>(boardBase, kAIBoardValidMovesOffset);
    const int count = *At<const int>(boardBase, kAIBoardValidMoveCountOffset);
    const float elo =
        1250.0F + 150.0F * std::clamp(*At<const int>(g_moduleBase, kDifficultyRva), 0, 9);
    *reinterpret_cast<GameMove*>(context.rdx) =
        g_runtime->ChooseMove(board, {moves, static_cast<std::size_t>(count)}, elo);
}

DWORD WINAPI MaiaBootstrapThread(LPVOID self) noexcept {
    try {
        std::array<wchar_t, 32768> path{};
        const DWORD length =
            ::GetModuleFileNameW(static_cast<HMODULE>(self), path.data(), path.size());
        if (!length || length >= path.size())
            return 0;
        const HMODULE module = ::GetModuleHandleW(nullptr);
        if (!module)
            return 0;
        g_moduleBase = reinterpret_cast<std::uintptr_t>(module);

        auto createWindowHook = safetyhook::create_inline(
            reinterpret_cast<void*>(&::CreateWindowExW),
            reinterpret_cast<void*>(&HookCreateWindowExW));
        if (!createWindowHook)
            return 0;
        g_createWindowExWHook = std::move(createWindowHook);

		auto modelPath =
			FindMaiaModel(std::filesystem::path(path.data()).parent_path());
		g_runtime = std::make_unique<MaiaRuntime>(modelPath);
        auto hook = safetyhook::create_mid(
            reinterpret_cast<void*>(g_moduleBase + kAiMoveMidHookRva), &AiMoveMidHook,
            SafetyHookMid::StartDisabled);
        if (!hook)
            return 0;
        g_aiMoveHook = std::move(hook);
        if (!g_aiMoveHook.enable())
            return 0;
        auto* target = At<std::uint8_t>(g_moduleBase, kSkipBuiltinAiRva);
        DWORD oldProtect{};
        if (!::VirtualProtect(target, kSkipBuiltinAiPatch.size(), PAGE_EXECUTE_READWRITE,
                              &oldProtect))
            return 0;
        std::ranges::copy(kSkipBuiltinAiPatch, target);
        ::FlushInstructionCache(::GetCurrentProcess(), target, kSkipBuiltinAiPatch.size());
        DWORD ignored{};
        ::VirtualProtect(target, kSkipBuiltinAiPatch.size(), oldProtect, &ignored);
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
        if (const HANDLE thread =
                ::CreateThread(nullptr, 0, &MaiaBootstrapThread, module, 0, nullptr))
            ::CloseHandle(thread);
    }
    return TRUE;
}
