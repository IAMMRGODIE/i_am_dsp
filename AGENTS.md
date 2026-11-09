# AGENTS.md — 给编码 agent 的工作说明

这是 `i_am_dsp` workspace：一个 Rust DSP 库 + CLAP/VST3 插件框架 + 网页面板 + 49 个产品。
第一次改代码前先读 [ARCHITECTURE.md](ARCHITECTURE.md)；本文件是它的「必须遵守」精简版。

## 先读什么

- [README.MD](README.MD) — 面向用户，中文，含构建与协议说明。
- [ARCHITECTURE.md](ARCHITECTURE.md) — 分层、线程模型、参数链路、10 条不变量。
- 仓库根目录之上的设计稿（`../patcher-design.md`、`../multipass-rack-design.md`、
  `../bundle-ui-ux-design.md`、`../bpm-sync-research.md`）描述正在做的事，读之前先确认是否已落地。

## 命令

```bash
cargo build                    # 默认 features：string + layout
cargo test                     # 全 workspace，约 700 个测试
cargo test -p i_am_dsp parameters_derive          # 单 crate / 单个测试
cargo clippy --all-targets     # 仓库要求零警告，交付前必须清
cargo fmt                      # rustfmt.toml 配置了 hard_tabs
cargo bundle -p i_am_effects_bundle --features vst3   # i_am_bundler 的别名
cargo run -p i_am_effects --example dump_face -- target/web-preview gain
cargo run -p i_am_dsp --features standalone           # egui + cpal 实时 demo
```

- `Cargo.lock` 被 gitignore（唯一例外是 `i_am_plugin/tests/set_library/Cargo.lock`）。
- `tools/` 下的 Python 脚本是校准/分析工具，不参与构建。
- 没有 CI；「跑过测试 + clippy 干净」就是本仓库的验证标准。

## 代码地图

| 目录 | 改什么来这里 |
| --- | --- |
| `i_am_dsp/src/lib.rs` | `Effect` / `Generator` / `ProcessContext` / `NoteEvent` |
| `i_am_dsp/src/processor.rs` | `Processor` / `Message` / `GeneratorEffect` 适配器 |
| `i_am_dsp/src/parameters/` | 参数值、`ParamMap`、稳定 id、布局元数据 |
| `i_am_dsp/src/effects/`、`generators/`、`tools/` | DSP 本体与可复用工具 |
| `i_am_plugin/src/` | 宿主包装、线程、params/state/latency/gui 扩展、`Editor` 接缝 |
| `i_am_web_gui/src/` | `Faceplate`、路由、IPC、`Scope`、`web_plugin!` 宏 |
| `i_am_effects/`、`i_am_synths/` | 产品注册与面板选项 |
| `i_am_patch/`、`i_am_patch_product/` | 图模型、编译、块引擎、patcher 页面 |
| `i_am_rack/` | Multipass：分频、模块、调制矩阵、页面 |
| `i_am_string/` | 物理建模弦（独立 crate，无依赖） |
| `*_bundle/` | 只放 `export_plugin!`，不放实现 |
| `i_am_bundler/` | `.clap` / `.vst3` 打包 |

## 约定

### 语言与注释

- 代码、注释、doc comment、提交信息：**英文**；README / 设计稿 / ARCHITECTURE：**中文**。
- 注释解释**为什么**，不复述代码。现有文档声音是「解释一个决定为什么是对的」，请保持一致。
- 公共项必须有 doc comment（`i_am_dsp` 和 `i_am_string` 开了 `#![warn(missing_docs)]`）。
- `unsafe` 块前写 `// SAFETY:` 说明为什么成立，照 `i_am_plugin` 里的写法。

### 格式与提交

- 缩进用 **tab**（`rustfmt.toml`），edition 2024。
- 提交信息格式：`<area>: <一句小写的陈述句>`，例如
  `patch: a port takes one wire`、`filter: a cutoff outside the band is clamped`。
  它描述「改完之后什么是真的」，不是「我做了什么」，句尾不加句号。
- 不要顺手格式化无关文件；不要新增依赖，除非现有依赖真的做不到。

### 测试

- 单元测试与实现同文件（`#[cfg(test)] mod tests`）；跨模块测试放 `src/tests.rs` 或 `src/tests/`。
- 集成测试放各 crate 的 `tests/`。页面路由测试在 `i_am_rack/src/tests/` 和 `i_am_web_gui` 里，
  新增路由时照抄它们的请求/响应辅助函数。
- 音频断言用确定性输入，避免依赖浮点边角；有阈值的测试写明为什么是这个阈值。

## 硬性约束（违反会静默坏掉宿主里的工程）

1. **`ParamMap` 与插件同寿**。采样率变化重建 DSP（`EffectProcessor::activate`），绝不重建参数表。
2. **稳定 host id 永不改变**。新参数追加或 `#[id(stable = N)]` 钉死；id 必须在 31 位内、不等于 `u32::MAX`。
3. **音频线程**：不分配、不加锁、不做 I/O、不写日志（`I_AM_PLUGIN_TRACE_PARAMS` 除外）。
   跨线程只用原子、环形缓冲、预分配通道。
4. **跨线程共享状态必须内部可变**：音频线程持 `&mut P` 时主线程仍能拿 `&P`。
5. **界面改参数直接写 `ParamMap` 原子**，不走消息通道；消息只承载动作。
6. **一个可加载库一个 `clap_entry`**。实现放普通 crate，入口放 `*_bundle`。
7. **构造时不知道采样率**（占位 44.1 kHz）；依赖采样率的系数在 `on_activate` 重建。
8. **`Value::Serialized` 是存档通道**（`rack_state`、`patch`），不是可编辑参数。

## 验证前端

只有一个入口，资源每次启动都从源码重建，不会看到过期的副本：

```bash
node tools/preview/preview.mjs cards          # 卡片/英雄区画廊（默认）
node tools/preview/preview.mjs product gain   # 单个产品的真实面板
node tools/preview/preview.mjs rack           # I Am Multipass 页面
node tools/preview/preview.mjs patch          # patcher 的实时编辑器
```

`--port` 换端口（默认 8950），`--dir` 换构建目录，`--no-build` 服务已有产物。
页面里的实时数据（scope/meter）在静态模式下会 404，页面按自己的方式降级；这不妨碍看布局。
`.ui-preview/serve.js` 是它的旧入口，仍然可用。

## 常见改动怎么下手

### 加一个效果器

1. DSP：`i_am_dsp/src/effects/<name>.rs` 实现 `Effect<2>` + `#[derive(Parameters)]`
   （或用 `i_am_effects` 里的产品级实现）。
2. 包装：在 `i_am_effects/src/lib.rs` 加一行
   `web_plugin!(NameWeb, DspType, |rate| DspType::new(rate), "iamdsp.effects.<name>.web", "I Am <Name>", (w, h), "#rrggbb");`
3. 参数上写 `#[layout(group, order, widget, size)]`、`#[unit(...)]`、`#[default(...)]`；
   面板由元数据生成，**不要**在页面里硬编码控件位置。
4. 导出：把名字加进 `i_am_effects_bundle/src/lib.rs` 的 `export_plugin!` 列表。
5. 验证：`cargo test -p i_am_effects`、`cargo clippy -p i_am_effects --all-targets`。

### 加一个合成器

同上，但实现 `Generator<2>`，产品体走 `i_am_synths` 的 `play_synth`（旁通=静音、scope 记录
输出而非输入），最后注册到 `i_am_synths_bundle`。两个只做转发的 trait 用 `synth_product!` 生成；
参数 `set_parameter` 有自定义逻辑时（鼓机会写 kit 的旋钮）用 `synth_processor!`，别手抄样板。
预设类参数在编辑器的 command 钩子里处理，不要混进通用 `id=value` 解析。

### 加一个 rack 模块

`i_am_rack/src/module.rs` 的 `ModuleKind` 加变体 + 实现；页面卡片在 `i_am_rack/src/page/`；
在 `i_am_rack/src/tests/modules.rs` 补一条「加到信号链后输出符合预期」的测试。

### 加一个 patcher 节点

`i_am_patch/src/nodes.rs` 里用稳定字符串 id 注册（不要用枚举序号当 id），在 `graph.rs` 的目录里
声明端口；需要在页面上画卡片的，再动 `i_am_patch/src/page/`。

### 动页面

页面 JS 是**按固定顺序拼接**的一个 IIFE：新增文件必须同时改
`i_am_web_gui/src/lib.rs` 的 `SCRIPT`（或产品自己的 `PAGE_JS`）里的 `concat!` 列表。
`card.css` / `card.js` 是共享资产，改之前先确认所有产品都还能看。

## 容易踩的坑

- **WebView2 仅 Windows**：其他平台能编译、只是打开编辑器会报错。不要在非 Windows 上加半套实现。
- **WebView2 需要可写 profile**：默认 `%LOCALAPPDATA%\i_am_plugin_wry\WebView2-v2`；
  打不开先看 `%TEMP%\i_am_plugin_wry\last-error.log` 与 `open.log`。
- **denormal**：包装层每个 `process` 都会调 `enable_flush_to_zero()`；自己接音频线程时要自己调。
- **`#[non_exhaustive]`**：`ProcessInfos`、`Unit`、`ParamWidget` 等对外枚举，加字段/变体时
  要检查所有 match 与页面 JSON 消费方。
- **对照 `vendor/wry`**：那是为了绕过窗口类全局注册的 vendored 补丁，不要同步上游覆盖它。
- **`I Am Multipass`** 直接 `export_plugin!(Rack)` 且 `crate-type = ["cdylib"]`；
  改动 `i_am_effects` 时它会一起链接，注意别把 `clap_entry` 带进来。

## 交付前检查

- [ ] `cargo clippy --all-targets` 对改动过的 crate 零警告
- [ ] `cargo test -p <改动的 crate>` 通过（动了共享层就跑全量）
- [ ] 新增公共项有 doc comment
- [ ] 没有重排/改变已有参数 id；新 id 只追加或钉死
- [ ] 音频路径没有新增分配、锁、I/O、日志
- [ ] 产品有增减时 `*_bundle` 的 `export_plugin!` 列表已同步，`cargo bundle` 仍能打包
- [ ] 结构变了就同步更新 [ARCHITECTURE.md](ARCHITECTURE.md)
