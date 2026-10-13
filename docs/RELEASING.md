# 版本与发布

应用版本唯一来源是 `studio/version.py` 的 `__version__`。采用 `主版本.次版本.补丁版本`：不兼容变化递增主版本，新功能递增次版本，修复递增补丁版本。初期使用 `0.x.y`。

1. 更新应用版本与 `CHANGELOG.md`，完成相关测试。
2. 合并到 `main`，创建与应用版本一致的标签，例如 `v0.1.1`。
3. 推送标签，GitHub Actions 校验版本、运行 Windows 测试、构建并验证独立程序，成功后创建 GitHub Release 并上传 Windows 与源码包。

```powershell
git switch main
git pull --ff-only
git tag -a v0.1.1 -m "Release 0.1.1"
git push origin v0.1.1
```

本地构建：`powershell -ExecutionPolicy Bypass -File .\build_windows.ps1`。输出为 `release/VerticalVideoStudio-<版本>-Windows.zip` 与 `release/VerticalVideoStudio-<版本>-Source.zip`。发布包顶层目录带版本号，避免和旧版本文件混用。

`CAPTION_REVISION`、`flow_version` 和 Edit Plan 格式版本属于缓存/数据协议，与应用版本独立。需要变更模型分析或缓存格式时按实际影响递增；单纯更新界面无需重跑已有文本任务。
