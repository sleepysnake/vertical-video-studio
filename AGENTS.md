# 项目协作

默认使用简体中文交流。当前事实以代码、测试和 `docs/WORKFLOW.md` 为准。

- 项目是 Python / PySide6 的 Windows 桌面程序，入口 `app.py`；当前生产模型流程见 `studio/model_flow.py` 和 `studio/pipeline.py`。
- ASR 原词索引和时间戳是原声裁剪及字幕映射依据。模型负责语义决定，程序校验结构、源时间和覆盖集合。
- 模型流程包含初稿、独立复审及保留语音字幕；修改时保留检查点恢复和 ASR 缓存复用。
- 提示词位于 `config/*.txt`，运行时加载。界面不提供提示词编辑入口，样式预览使用通用文字。
- 使用 `.venv/Scripts/python.exe` 运行测试，相关媒体测试需 FFmpeg、FFprobe 和中文字体。
- `studio/version.py` 是应用版本唯一来源，内部分析版本独立维护。发布规则见 `docs/RELEASING.md`。
- 不提交本机设置、真实素材、转写、日志、Key、`analysis/`、构建和发布产物；新增测试使用匿名或合成数据。
- `build_windows.ps1` 保留 UTF-8 BOM，以兼容 Windows PowerShell 5.1。
