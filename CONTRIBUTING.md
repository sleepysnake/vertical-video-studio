# 参与开发

从 `main` 创建分支开发，例如 `feat/topic-style` 或 `fix/caption-timing`。修改后运行相关测试，通过 Pull Request 合并。提交说明描述实际行为变化。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

真实媒体测试需要本机 FFmpeg / FFprobe 和中文字体。缺失依赖的测试会跳过；发布测试环境应安装这些依赖。测试不需要真实 API Key，使用模拟返回验证程序行为，不能代表真实模型的语义效果。

不得提交 API Key、`.env`、用户设置、原视频、转写、处理日志或本地 `analysis` 目录。新增回归用合成素材或匿名文本。

应用版本统一维护在 `studio/version.py`。发布步骤见 [RELEASING.md](docs/RELEASING.md)。涉及缓存结构时单独更新相关内部分析版本，界面样式调整不应使文本模型缓存失效。
