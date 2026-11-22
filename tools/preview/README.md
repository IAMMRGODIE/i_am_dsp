# 前端预览

一个入口，四种模式。资源每次启动都从源码重建，所以浏览器里看到的就是插件当前会嵌入的页面。

```bash
node tools/preview/preview.mjs cards          # 卡片/英雄区画廊（默认）
node tools/preview/preview.mjs product gain   # 单个产品的真实面板
node tools/preview/preview.mjs rack           # I Am Multipass 页面
node tools/preview/preview.mjs patch          # patcher 的实时编辑器（它自己的服务器）
```

- `--port N`：端口，默认 8950。
- `--dir DIR`：把预览构建到别处（默认 `target/preview/...`）。
- `--no-build`：不重建，直接服务已生成的内容。

## 各模式是怎么来的

| 模式 | 数据从哪来 | 说明 |
| --- | --- | --- |
| `cards` | `.ui-preview` 的 HTML + 现场拼接的 `kit.*` / `card.*` | 卡片组件的视觉对照 |
| `product <name>` | `cargo run -p i_am_effects --example dump_face` | 真实的 `/face`、`/state` JSON |
| `rack` | `cargo run -p i_am_rack --example dump_rack` | 默认 RackState，静态看布局 |
| `patch` | `cargo run -p i_am_patch_product --example serve` | 页面要编辑实时图，所以跑的是产品自己的服务器（去掉宿主），页面和 agent 用的是同一套路由 |

静态模式下页面照常发送 `fetch("/face")` 这类请求，服务器把它们映射到 dump 出来的文件。
缺失的实时数据（scope、meter）会 404，页面按自己的方式降级，不影响看布局。
