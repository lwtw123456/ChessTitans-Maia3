# Chess Titans — Maia3

为 Windows 7 x64 自带国际象棋（Chess Titans）接入 Maia3 棋力模型的 DLL。

编译产物为 `slc.dll`。DLL 被游戏加载后，会接管原有 AI 的选步流程，使用 Maia3 ONNX 模型从游戏当前局面和合法着中选择落子，并在游戏左上角显示每步后的 `WIN / DRAW / LOSS` 评估。

## 实现

- 禁止游戏原 AI 线程启动，并跳过原 AI 的计算流程。
- 使用 **SafetyHook** 在 AI 落子流程中安装 hook。
- 从游戏 AI Board 读取棋子类型、颜色和当前行棋方。
- 将局面编码为 `1 x 64 x 12` one-hot tensor。
- 读取游戏内部难度值 `1 ~ 10`，映射为 Elo： `600 ~ 2600`。
- 使用 **ONNX Runtime** 在 CPU 上执行 Maia3 ONNX 模型。
- 调用游戏内部的合法着生成函数，取得当前全部合法着。
- 对游戏生成的合法着计算 Maia policy 索引，在 `logits_move` 中选择分数最高的一步。
- 将选中的着法写回游戏原 AI 输出结构。
- 玩家和 AI 每次落子前，都会先通过游戏内部函数临时应用该着，再调用模型的 `logits_value` 输出。
- 对 `logits_value` 的 loss / draw / win 做 softmax，并结合当前行棋方统一 WIN / LOSS 的显示方向；评估完成后立即撤销临时落子，不改变游戏实际局面。
- 使用 **kiero2** 定位 Direct3D 9 的 `Reset` / `Present`，再通过 **SafetyHook** 安装 hook。
- 使用 **Dear ImGui** 的 DX9 backend 在游戏左上角绘制评估浮窗：
  - 显示 `WIN`、`DRAW`、`LOSS` 百分比和进度条；
  - 尚无评估结果时显示 `Waiting for evaluation...`。
- 对局结束后，清空上一局评估结果，使浮窗恢复等待状态。
- 创建主游戏窗口时，将标题 `Chess Titans` 改为 `Chess Titans - Maia 3`。

## 依赖

第三方依赖以压缩包形式放在 Releases 中。当前源码使用到的主要依赖：

- SafetyHook
- Zydis
- ONNX Runtime static libraries
- kiero2
- Dear ImGui
- Dear ImGui DirectX 9 backend

## 模型来源

Maia3 ONNX 模型来自 Hugging Face 上的 **[bqrio/maia3-onnx](https://huggingface.co/bqrio/maia3-onnx)**。

本仓库默认附带 `maia3-5m.fp32.onnx`，用于开箱即用；DLL 本身并不限定于该模型。

当前版本会自动识别以下 Maia3 模型：

```text
maia3-5m.fp32.onnx
maia3-5m.fp16.onnx
maia3-23m.fp32.onnx
maia3-23m.fp16.onnx
maia3-79m.fp32.onnx
maia3-79m.fp16.onnx
maia3-3m-ablation.fp32.onnx
maia3-3m-ablation.fp16.onnx
```

可以根据需要从 `bqrio/maia3-onnx` 下载其他受支持的容量或精度版本，例如 `maia3-79m.fp32.onnx` 或 `maia3-79m.fp16.onnx`。

如果程序目录中同时存在多个受支持的模型，DLL 会按照上述顺序选择第一个找到的模型。

## 构建

需要 CMake 3.24+、MSVC x64，并支持 C++23。

```bat
cmake -S . -B build -A x64
cmake --build build --config Release
```

输出：

```text
build\Release\slc.dll
```

## 使用

将 `slc.dll` 和一个受支持的 Maia3 ONNX 模型放到国际象棋程序目录，例如：

```text
slc.dll
maia3-5m.fp32.onnx
```

启动游戏后，DLL 会自动在程序目录中查找并加载模型，无需修改配置。

加载成功后，游戏窗口标题会显示为 `Chess Titans - Maia 3`，左上角显示 Maia3 的 `WIN / DRAW / LOSS` 评估浮窗。开始产生落子评估前，浮窗会显示 `Waiting for evaluation...`。

`slc.dll` 导出游戏加载时需要的 `SLGetWindowsInformationDWORD`，如果程序目录中已有 `slc.dll`，直接覆盖即可。

## License

源码使用 MIT License。

`third_party` 中的第三方代码、静态库以及 Maia 模型文件分别遵循其自身许可证。
