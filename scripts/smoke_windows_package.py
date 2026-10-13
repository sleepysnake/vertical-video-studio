"""Launch the released EXE from a fresh extraction with no dev PATH entries."""
from __future__ import annotations

import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import shutil
from zipfile import ZipFile


def main(archive: Path) -> None:
    if sys.platform != "win32":
        raise RuntimeError("此验收脚本只支持 Windows")
    with tempfile.TemporaryDirectory(prefix="vvs-package-smoke-") as temporary:
        folder = Path(temporary)
        with ZipFile(archive) as package:
            bad = package.testzip()
            if bad:
                raise RuntimeError(f"ZIP 校验失败：{bad}")
            package.extractall(folder)
        matches = list(folder.glob("*/VerticalVideoStudio.exe"))
        if len(matches) != 1:
            raise RuntimeError("ZIP 中没有主程序")
        executable = matches[0]
        editor_cache = folder / "editor-smoke"
        editor_cache.mkdir()
        (editor_cache / "transcription.json").write_text(json.dumps({"transcripts": [{"sentences": [{"words": [
            {"begin_time": 0, "end_time": 200, "text": "咦"},
            {"begin_time": 200, "end_time": 1000, "text": "我吃得不多"}]}]}]}), encoding="utf-8")
        (editor_cache / "captions.json").write_text(json.dumps({"reviewed": True, "captions": [
            {"start": 0, "end": 1, "start_word": 0, "end_word": 2, "text": "咦我吃得不多"}],
            "edit_plan": {"cuts": [], "source_duration": 1}}), encoding="utf-8")
        ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        preview_checked = bool(ffmpeg and ffprobe)
        if preview_checked:
            from PIL import Image
            video, template = folder / "预览 样片.mp4", folder / "template.png"
            subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=640x360:r=25:d=1",
                            "-c:v", "libx264", str(video)], check=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
            Image.new("RGB", (540, 960), "#305F90").save(template)
            settings = folder / "isolated-profile" / "VerticalVideoStudio" / "settings.json"
            settings.parent.mkdir(parents=True)
            settings.write_text(json.dumps({"source": str(video), "template": str(template),
                "slot": "0,326,540,304", "topic": "健康饮食", "topic_box": "20,30,500,180",
                "topic_style": {"font_size": 40, "background": "rounded", "background_color": "#A03030"},
                "ffmpeg": ffmpeg, "ffprobe": ffprobe}, ensure_ascii=False), encoding="utf-8")
        for attempt in range(3):
            marker = folder / f"startup-{attempt}.txt"
            environment = {key: value for key, value in os.environ.items()
                           if key.upper() not in {"PYTHONPATH", "PYTHONHOME", "QT_PLUGIN_PATH",
                                                  "QT_QPA_PLATFORM_PLUGIN_PATH"}}
            windows = os.environ["SystemRoot"]
            environment["PATH"] = windows + r"\System32;" + windows
            environment["QT_QPA_PLATFORM"] = "offscreen"
            environment["VVS_STARTUP_MARKER"] = str(marker)
            environment["VVS_EDITOR_SMOKE"] = str(editor_cache)
            if preview_checked:
                environment["VVS_PREVIEW_SMOKE"] = "1"
            environment["APPDATA"] = str(folder / "isolated-profile")
            process = subprocess.Popen([str(executable)], cwd=folder, env=environment,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline and not marker.is_file():
                    if process.poll() is not None:
                        raise RuntimeError(f"启动尝试 {attempt + 1} 提前退出，代码 {process.returncode}")
                    time.sleep(.1)
                if not marker.is_file() or marker.read_text(encoding="utf-8") != "ready":
                    raise RuntimeError(f"启动尝试 {attempt + 1} 未能载入 Qt 界面")
                saved = json.loads((editor_cache / "caption-overrides.json").read_text(encoding="utf-8"))
                if saved["captions"][0]["text"] != "我吃得不多":
                    raise RuntimeError("发布包的字幕编辑与保存验证失败")
            finally:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=10)
        print("Windows ZIP 全新解压启动、字幕编辑及保存验证通过（3/3）" +
              ("；主题背景、视频帧与区域放大预览通过" if preview_checked else "；未安装媒体工具，未检查视频预览"))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
