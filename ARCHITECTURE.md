# i_am_dsp — 架构文档

> 状态：workspace 14 个 crate、约 8.4 万行 Rust（另有约 1.5 万行网页面板 HTML/JS/CSS）、694 个 `#[test]`；
> CLAP ✅、VST3 ✅（经 `clap-wrapper`）；界面是网页面板（Windows 走 WebView2、Linux 走 WebKitGTK、
> macOS 走 WKWebView）；已产出 `I Am Effects Web`、
> `I Am Synths`、`I Am Patch` 三个可加载插件，`I Am Multipass`（`i_am_rack`）自己就带 `export_plugin!`，
> `cargo bundle -p i_am_rack --features vst3` 可单独打包。
>
> 本文描述代码**现在是什么样**。要做什么见根目录的 `patcher-design.md`、`multipass-rack-design.md`、
> `bundle-ui-ux-design.md`、`bpm-sync-research.md`、`drummer-roadmap.md`。

## 0. 一句话

一套自研的 Rust 音频 DSP 库 + CLAP/VST3 插件包装 + 「参数元数据即界面」的网页面板；
同一套积木拼出 49 个产品：38 个效果器、9 个合成器、2 个 patcher，外加 1 个 rack 宿主。

分层只有一条主线：

```
DSP 对象（Effect / Generator）
      ↑ 参数（Parameters / ParamMap）
音频逻辑（Processor：无界面、无宿主）
      ↑ 宿主协议（CLAP/VST3）
插件包装（i_am_plugin：线程、状态、参数扩展、Editor 接缝）
      ↑ 窗口与页面
界面（i_am_web_gui：元数据→JSON→网页）
      ↑ 产品配方（web_plugin! / 自定义 page）
产品（i_am_effects / i_am_synths / i_am_patch / i_am_rack）
```

## 1. 目标与非目标

**目标**

- 在 Rust 里写插件像写普通库：实现 `Effect`/`Generator` + `#[derive(Parameters)]`，其余（CLAP、状态、
  参数表、界面布局、打包）由框架承担。
- 同一个 DSP 对象既能被插件包装，也能被 standalone demo、被测试、被别的产品（rack、patcher）复用。
- 参数元数据是界面的唯一来源：改一个参数的 `#[layout]`，网页上的位置、控件、单位随之改变，不写第二份布局。

**非目标（当前）**

- 不做 DAW，不做插件管理器；standalone 只是开发用的 egui demo（`real_time_demo`）。
- 不做 SIMD（`wide` 已引入但只在局部用）、不做动态音频端口、不做 AU 的打包验证。
- 不追求 API 稳定：README 明确写了随时可能有破坏性改动。

## 2. workspace 全景

### 2.1 crate 表

| crate | 行数 | 职责 | 关键入口 |
| --- | ---: | --- | --- |
| `i_am_dsp_derive` | 1.4k | `#[derive(Parameters)]` 过程宏 | `i_am_dsp_derive/src/lib.rs` |
| `i_am_string` | 6.3k | 物理建模弦（拨/击/弓），波导与模态两种弦核 | `i_am_string/src/lib.rs` |
| `i_am_dsp` | 39.4k | DSP 核心：`Effect`/`Generator`/`Processor`、参数、时间线、效果与音源库、工具 | `i_am_dsp/src/lib.rs` |
| `i_am_plugin` | 5.1k | CLAP/VST3 包装、线程模型、宿主扩展（params/state/latency/gui/timer）、Editor 接缝、WebView 后端（WebView2 / WebKitGTK / WKWebView） | `i_am_plugin/src/lib.rs` |
| `i_am_web_gui` | 2.2k | 网页面板：faceplate、JSON 路由、IPC、Scope 遥测 | `i_am_web_gui/src/lib.rs` |
| `i_am_effects` | 8.4k | 38 个效果器产品 | `i_am_effects/src/lib.rs` |
| `i_am_synths` | 5.4k | 9 个合成器产品 | `i_am_synths/src/lib.rs` |
| `i_am_patch` | 5.1k | DAG 图模型、编译、块引擎、复音、调制源、行文本格式 | `i_am_patch/src/lib.rs` |
| `i_am_patch_product` | 1.0k | patcher 的两个产品（effect + generator）与页面路由 | `i_am_patch_product/src/patcher.rs` |
| `i_am_rack` | 7.8k | `I Am Multipass`：四通道 rack、模块、调制矩阵、RackState | `i_am_rack/src/lib.rs` |
| `i_am_effects_bundle` | 59 | 38 个效果的 `clap_entry` | 同名 `lib.rs` |
| `i_am_synths_bundle` | 29 | 9 个合成器的 `clap_entry` | 同名 `lib.rs` |
| `i_am_patch_bundle` | 7 | patcher 的 `clap_entry` | 同名 `lib.rs` |
| `i_am_bundler` | 1.5k | `cargo bundle`：把 dylib 变成 `.clap` / `.vst3` | `i_am_bundler/src/main.rs` |

### 2.2 依赖方向

```
            i_am_dsp_derive          i_am_string
                   │                     │ (feature "string")
                   └──────────┬──────────┘
                              ▼
                        ┌───────────┐
                        │ i_am_dsp  │  DSP 核心
                        └─────┬─────┘
                              ▼
                       ┌──────────────┐
                       │ i_am_plugin  │  CLAP/VST3 + 线程 + Editor 接缝
                       └───┬──────┬───┘
                           │      │
                     ┌─────▼──┐   │
                     │i_am_   │   │
                     │web_gui │◄──┼──────────────┐
                     └───┬────┘   │              │
                  ┌──────┴───────┐ │              │
                  ▼              ▼ │              ▼
            i_am_effects    i_am_synths      i_am_patch
                  │              │                │  (节点里嵌入 effects + synths)
                  └──────┬───────┘                │
                         ▼                        ▼
                   i_am_patch_product ◄────── i_am_rack
                         │  (也直接复用 effects)
                         ▼
              *_bundle（只有 export_plugin! 与 clap_entry）
```

### 2.3 为什么有 `*_bundle`

`clap_entry` 是**全局符号**：一个库里只能有一个，否则两个插件链接到一起就冲突。
所以「产品本体」放在 `i_am_effects` / `i_am_synths` / `i_am_patch_product`，可以被 rack 之类的
别的插件**复用同一份实现**；「可加载的宿主入口」单独放一个 bundle crate，只做：

```rust
pub use i_am_effects::*;
use i_am_plugin::export_plugin;
export_plugin!(GainWeb, DistortionWeb, /* ... 38 个 */);
```

每个 bundle 的 `[lib] crate-type = ["cdylib"]`，由 `i_am_bundler` 按
`[package.metadata.i_am_dsp]`（name / vendor / id）打包。

## 3. 核心模型（`i_am_dsp`）

### 3.1 三种处理形态

| trait | 位置 | 方向 | 用在 |
| --- | --- | --- | --- |
| `Effect<CHANNELS>` | `i_am_dsp/src/lib.rs` | 原地改写输入缓冲 | 效果器、rack 模块、patcher 节点 |
| `Generator<CHANNELS>` | 同上 | 每样本产出一个 frame | 合成器、采样器、鼓机 |
| `Processor` | `i_am_dsp/src/processor.rs` | **固定立体声** `[f32; 2]` 的宿主边界 | 插件产品 |

关键点：

- `Effect::process(&mut self, samples, other, ctx)` 原地处理，`other` 是额外输入（侧链）。
  附件钩子：`delay()`、`allowed_notes()`（autotune 的音名遮罩）、`gain_reduction()`、`band_meters()`、
  `clear_tail()`（PANIC / 清尾巴）。
- `GeneratorEffect<G>` 是唯一的适配器：把 generator 塞进 effect 的槽位（patcher 的 generator 节点、
  rack 的音源模块都靠它）。参数、音符、tempo 原样转发，只有 buffer 交换方向。
- `Processor` 是**插件边界**，故意只有双声道 + 一个 `Message` 关联类型：
  界面不直接碰 DSP，只通过 `ParamMap` 原子和 `on_message` 交流。
- `Vec<T>` / `Vec<Box<dyn Effect>>` 实现了链式 `Effect`，标识符加 `"{i}."` 前缀。

### 3.2 参数系统：五层结构

```
#[derive(Parameters)]  struct Foo { #[range(..)] cutoff: f32 }   ← 宏生成 get/set
        │
        ▼ get_parameters()
Value::Float { value, range, logarithmic }        ← 一次性的「带范围的快照」
        │
        ▼ to_atomic_value()
AtomicValue::Float { AtomicF32, range, ..., changed }  ← 跨线程共享值
        │
        ▼ ParamMap::build()
ParamMap { Arc<Vec<Arc<AtomicValue>>>, BiMap<String,usize>, stable_ids, metas }  ← 宿主/界面看的表
        │
        ▼ Paramed<T>
Paramed { value: T, params: ParamMap }            ← 「DSP 对象 + 它的参数表」
```

- `Paramed<T>` 实现 `Effect`/`Generator`/`Processor`，在每次 `process`/`generate` 开头调用
  `sync_params()`：把表里每个原子值 `SeqCst` 读出来，和上次写回 T 的值比较，变了才调
  `set_parameter`。**不能**用 `changed` 标志当判据：它是「界面向宿主上报」的标志，宿主写进来的值
  会把它清掉（`handle_param_value_event`），拿它做门就会断掉宿主→插件的同步。这个标志也**不能**
  在这里清：块尾的 `events::sync_params` 要靠它把界面改动上报给宿主，由上报来清。O(参数数) 每块，
  换来的是实现简单和永远一致。
- 标识符是唯一真相：字段名（或元组位置）→ 可选 `#[id(prefix/name)]` → 嵌套用点号
  （`sub.a`）。`set_parameter_by_index` 只是按表位置转发，宏内部从不依赖位置。
- `Value` 有五种：`Float` / `Int` / `Bool` / `Serialized(Vec<u8>)` / `Nothing`。
  **`Serialized` 是保留通道**：宿主把它当不可编辑的 blob 存进工程，rack 的 `rack_state`、
  patcher 的 `patch` 都靠它把结构化状态塞进宿主存档。
- 元数据 `ParameterMeta`：默认值、`Unit`（ms/Hz/beats/s/samples/dB/%）、`ParamLayout`
  （group/order/widget/size/bipolar/step/hero）。`Unit::format` 会把 `0.5` 渲染成 `1/8`，
  反过来也能解析。

### 3.3 稳定参数 id

```
identifier --FNV-1a(salt)--> 32bit --avalanche--> & 0x7FFF_FFFF --→ host id
```

- CLAP 要求 id 永不改变，且宿主把自动化/调制按 id 存。所以 id 由 **identifier 派生**，而不是位置：
  插入/删除/重排字段不会改别的参数的 id。
- 31 位上限是为了 VST3：Steinberg 占最高位，`clap-wrapper` 会剥掉它，超出的 id 在 VST3 里会指向
  另一个参数。
- 冲突（哈希撞车或与 `#[id(stable=N)]` 抢占）时用 salt 重算，而不是别名。
- 已发布、曾经用位置 id 的插件用 `StableIdScheme::Position` 复现旧 id。

### 3.4 时间与传输

`ProcessInfos` 每个块由包装层填好：`sample_rate`、`playing`、`tempo`、`position_beats: Option<f64>`、
`time_signature`、`current_time`。要点：

- `position_beats` 是**样本精确**的：每样本由 `ProcessContext::set_sample_offset` 推进；
  宿主报告传输变化时重新锚定，所以 seek / 循环回绕不会累积漂移。
- `BeatClock`（`i_am_dsp/src/tools/beat_clock.rs`）保存锚点 `(absolute_sample, beats, tempo, tempo_inc)`，
  `advance_to(n)` 用绝对样本数插值；能处理 tempo 渐变。旧的 `BpmSyncer` 已标记 deprecated，
  只在没有宿主时间线时用于自由运行。
- 每个效果自己决定是否跟 BPM（`synced_time.rs` 把「拍」与「毫秒」两种长度互转）。

### 3.5 参数表的所有权

```
PluginMain::new()  ── param_map = processor.param_map() ──► 宿主参数扩展 / 网页 /save
        │                                                     ▲
        │  同一组 Arc<AtomicValue>                           │
        ▼                                                     │
音频线程 process(): Paramed::sync_params()  ──读──> DSP 对象
```

**不变量**：这张表从插件创建到销毁必须是同一张。采样率变化时重建的是 `Paramed` 里的 DSP 对象
（`EffectProcessor::activate`），表本身不动——否则宿主的自动化会写进一张没人读的新表。

### 3.6 块间参数平滑（feature `smoothing`）

VST3 经 `clap-wrapper` 送进来的参数自动化**每块只有最后一个点**，画成曲线的自动化到了 DSP 就是
一级级台阶；台阶落在 gain 或截止频率上，就是咔哒声。规避放在包装层
（`i_am_plugin/src/smoothing.rs`），DSP 完全不知情：

- 插件在 `Plugin::SMOOTHED_PARAMETERS` 里点名参数和时长
  （`SmoothedParameter::new("gain", 20.0)`），`web_plugin!` 用 `smoothed = &[...]` 传进去；
  只有开了 `smoothing` feature 才生效。
- 每个样本、在这块的参数事件应用**之后**、DSP 读参数**之前**，包装层读一次原子值：它不等于自己
  上次写下的数，就说明有人（宿主或界面）写了新目标，于是把「交给 DSP 的数」在 N 个样本里线性走过去。
  N 只由时长和采样率算出：20 ms @ 48 kHz = 960 个样本。放在这个位置是因为 VST3 的目标正是在块首
  那一刻写进来的，早一步就看不到它，晚一步 DSP 已经读到跳变了。
- 平滑写回的是 `AtomicValue` 里的 `AtomicF32`，**不碰 changed 标志**——否则宿主自己的自动化会
  被回灌成每样本一个参数事件。
- 代价是那个参数最多滞后一个 ramp 时长，所以只点名连续参数；开关与波形选择写进去会表现得像卡顿。
- `reset()` 丢掉正在走的 ramp、直接对齐目标：重置是断点，不该从上次的位置滑过来。

## 4. 宿主层：`i_am_plugin`

### 4.1 `Plugin` trait

`Processor` + 宿主要知道的一切：`const DESCRIPTOR`（id/name/vendor/version/tags）、静态音频与 MIDI 端口、
`HAS_GUI`、`new()`、`param_map()`、`sync_params()`、`editor()`、`on_activate(sample_rate)`、
`on_reset` / `on_start_processing` / `on_stop_processing`、`reports_latency_changes()`。

注意 `new()` **不带采样率**：宿主先要插件，之后才说会话速率。所以 DSP 用占位 44.1 kHz 构造，
`on_activate` 时重建。`HAS_GUI` 是常量而不是从 `editor()` 推出来——宿主在创建任何编辑器之前就要知道。

### 4.2 线程模型

```
宿主主线程                          音频线程
┌────────────────────────────┐      ┌──────────────────────────────┐
│ PluginMain<P>              │      │ AudioProcessor<'a, P>        │
│  Pin<Box<P>>  ─────────────┼─────►│  processor: usize（裸地址）   │
│  param_map: ParamMap       │      │  clock / samples_processed   │
│  editor: Option<Editor>    │      │  event 通道 / latency 发布器  │
│  message_receivers         │      └──────────────────────────────┘
│  latency: LatencyWatch     │
└──────┬─────────────────────┘
       │ on_main_thread()：收 GUI 消息 → P::on_message(&self)
       │ 宿主 timer：跟窗口几何
       ▼
┌────────────────────────────┐
│ Editor<P>（WebView）        │  宿主给的 HWND / X11 window / NSView
│  页面 → ipc.postMessage     │
└────────────────────────────┘
```

- 音频线程用 `&mut P`，主线程同时可拿 `&P`（`on_message`、`delay`、载入状态时 `&mut P`）。
  两者重叠，所以**跨线程共享的状态只能是原子或内部可变**——这是写产品时最容易踩的一条。
- GUI 消息不直接进 DSP：页面 → channel → `on_main_thread` → `P::on_message`。
  参数**不走**这条路：页面直接写 `ParamMap` 的原子，音频线程下一个块自己读。
- WebView2 创建时会泵宿主的消息循环，可能重入 `show`/`hide`/`destroy`；`opening` 等标志把它们
  变成待办队列，防止开出第二个编辑器。GTK 版反过来：它不泵别人的循环，而是要别人泵它的
  （`Editor::pump`，从宿主的 GUI timer 里调）。

### 4.3 Editor 接缝

GUI 工具包完全在 `Editor` trait 后面：

```rust
trait Editor<P: Plugin> {
    fn open(&mut self, context: EditorContext<P>) -> Result<(), EditorError>;
    fn show(&mut self) -> Result<(), EditorError>;
    fn hide(&mut self) -> Result<(), EditorError>;
    fn close(&mut self);
    fn set_parent(&mut self, parent: ParentWindow) -> Result<(), EditorError> { Ok(()) }
    fn set_scale_factor(&mut self, f: f64) -> Result<(), EditorError> { Ok(()) }
    fn set_size(&mut self, size: WindowSize) -> Result<(), EditorError> { Ok(()) }
    fn pump(&mut self) {}
}
```

`EditorContext` 里只有跨工具包的东西：`ParentWindow`（`raw-window-handle` 裸句柄）、逻辑尺寸、
缩放、标题、可缩放、以及 `Sender<P::Message>`。`i_am_plugin::wry` 是唯一的实现，按平台分三个后端：

| 平台 | 引擎 | 窗口 | 谁来泵事件循环 |
| --- | --- | --- | --- |
| Windows | WebView2 | 自己建的子 HWND，宿主窗口是它的父窗口 | 宿主的消息循环（我们挂一个 Win32 timer 跟尺寸） |
| Linux | WebKitGTK | 嵌进宿主的 X11 窗口（Wayland 不支持嵌入，直接报错） | 我们，从宿主的 GUI timer 调 `Editor::pump` |
| macOS | WKWebView | 作为宿主 `NSView` 的 subview | 宿主的 NSApplication |

Windows 与 Linux 都在真机上跑过（Linux 是 WSL2 + WSLg 里的 Kali：X11 嵌入、页面加载、IPC 都通）；
macOS 的后端代码齐全，但**没有在真机上验证过**。其余平台仍是报错的 stub。

### 4.4 宿主扩展

| 扩展 | 实现 | 要点 |
| --- | --- | --- |
| params | `PluginMainThreadParams` | 用稳定 id；CLAP 没有 log 标志，所以对数参数对宿主呈现为 0..1，跨境范围用 symlog |
| state | `PluginStateImpl` | 保存 = `to_binary(Vec<Parameter>)`（ciborium）；载入时 `Serialized` 交给 `set_parameter`，普通值写回 map |
| latency | `LatencyPublisher` → `LatencyWatch` | 音频线程发布，主线程发现变化后调 `host_latency.changed()`（clack 的绑定没覆盖这个调用，故经 `clap-sys` 直接取 `clap_host` 的函数指针） |
| gui | `PluginGuiImpl` | 打开/关闭/缩放/重设父窗口，全部转发给 `Editor` |
| timer | `PluginTimerImpl` | 宿主支持 timer 时跟窗口尺寸；Linux 上还负责推进 GTK 循环（`Editor::pump`） |

### 4.5 导出

- 单插件：`export_plugin!(MyPlugin)` → `clap_entry`；开 `vst3` feature 时再导出
  `GetPluginFactory` / `InitDll` / `ExitDll`（链接 `clap-wrapper` + VST3 SDK）。
- 多插件：`PluginSetSource` + `SetEntry` + `SetFactory`，一个库列出多个插件。
- AU：实现 `PluginAuExt`（`AU_TYPE`/`AU_SUBTYPE`）只为导出符号，打包未验证。

### 4.6 `EffectProcessor`

绝大多数效果产品不需要自己写 `Processor`，用
`i_am_plugin::processor::EffectProcessor::new(|rate| MyEffect::new(rate))`：

- 构造时按占位 44.1 kHz 建 DSP，`activate(rate)` 时换掉 `Paramed.value` 并 `sync_params`，
  参数表保持同一张。
- `with_build` 允许捕获状态（例如已加载的 IR 要在重建后重新接上）。
- `EffectMessage` 是空消息：参数已经是界面本身，消息只承载动作。

## 5. 界面层：`i_am_web_gui`

### 5.1 元数据即布局

`Faceplate` 描述产品身份与页面的组织方式：

| 字段 | 作用 |
| --- | --- |
| `title` / `accent` / `size` | 标题、身份色、设计尺寸 |
| `mode` | 哪个参数在页眉当模式开关，而不是一个旋钮 |
| `visible` | 每个模式显示哪些参数（`(mode, &[names])`） |
| `header` | 哪些参数放页眉 |
| `switches` | 成对互斥参数（例如「毫秒」与「拍」） |
| `pages` | 标签页，每页列出要显示的 section |
| `hidden` | 宿主能看到、但页面不画的参数 |
| `no_hero` | 不画某个 section 的配图（继承来的音源 section） |
| `script` / `clear` | 是否显示脚本面板 / 清尾巴按钮 |

section 名、控件类型、大小全部来自参数的 `#[layout(...)]`；页面从不猜位置。

### 5.2 路由与 IPC

自定义协议（`WryEditor::with_custom_protocol`）提供：

| 路由 | 内容 |
| --- | --- |
| `GET /face` | 控件、布局、模式、身份（JSON） |
| `GET /state` | 当前值 + bypass |
| `GET /scope`、`/stereo` | 无锁环形缓冲的波形（f32 二进制） |
| `GET /note`、`/allowed`、`/meter` | 当前音符 / autotune 允许的音 / 电平 |
| `GET /kit.css`、`/kit.js`、`/card.css`、`/card.js` | 面板资产 |
| 产品自有 | `/clear`、`/script`、`/rack`、`/patch`、`/catalog`、`/module-ir` … |

写回通过 `window.ipc.postMessage`：

```
"cutoff=800"  → 按参数的 Value 类型解析 → 写进同一个 ParamMap 原子
"__bypass=1"  → 写 bypass 标志（不是参数，而是插件状态）
其它         → 产品的 command 钩子 → 否则变成 processor 的 Message
```

### 5.3 Scope（遥测）

音频线程写、页面读，全程无锁：8192 帧容量、页面读 4096；写索引用 Release 发布、Acquire 读，
所以看到新索引就一定能看到之前的样本。`Channels::None/Mono/Stereo` 决定分配哪些环——
一个 rack 可能有几十个模块，没人看的模块不分配。

### 5.4 `web_plugin!`：产品配方

一行宏就是完整产品：

```rust
web_plugin!(
    GainWeb, GainDsp,
    |_| GainDsp::new(1.0),
    "iamdsp.effects.gain.web", "I Am Gain", (360.0, 230.0), "#35c77a"
);
```

宏生成：`EffectProcessor` + bypass 原子 + `Scope` + `clear` 标志；`Parameters` 转发；
`Processor`（旁通=干声、scope 记录输入/输出/侧链、处理音符事件）；`Plugin`（`DESCRIPTOR`、
`window_options`、`param_map`、`on_activate`、`editor`）。
可选参数：`mode` / `visible` / `header` / `switches` / `pages` / `hidden` / `ports` / `midi` /
`tags` / `telemetry` / `clear`。

面板本身也共用一套工具，产品不再各抄一份：`window_options(size, title)` 给出 0.85 缩放、
两轴可缩放的宿主窗口（15 个产品共用；rack 和 patcher 版面不同，自己给尺寸）；
`json_escape` 与 `percent_decode` 是路由拼 JSON、解析 data URL 的唯一实现。

### 5.5 一份描述，两个页面壳

产品页（`page/kit/*`）和卡片页（`page/card/*`）不是两套 UI，它们共享下面三层：

- **描述**：`Faceplate` + 参数 `#[layout]` → `sections()` → `face_json`；patch 的节点目录
  （`i_am_patch/src/catalog.rs`）读的是同一份 sections/pages/heroes，不再各自拼字段。
- **控件**：两边都调用 `IAmCard.controlHandle / switchHandle / choiceHandle / sliderHandle`，
  没有第二套旋钮实现。
- **hero**：两边都调用 `IAmCard.hero(kind, values, context)`，产品页只负责把 live 数据
  （scope/spectrum/meter/IR…）放进 context 并轮询。

卡片布局本身也共享：`IAmCard.faceCard({ face, values, context, activePage, onPage, onEdit, wrapControl })`
把 face 的 sections/pages/heroes/header/visible/switches 画成一张卡（section 标题、分页标签、
hero 高亮正在编辑的控件）。patch 的产品卡就是它加上节点 chrome（调制环、拖放目标、缩放、端口）。

**边界**：产品页的 `.face` 壳和它的页面级交互（sampler 拖放/切片、IR 淡入淡出、wavetable、
kick/snare 编辑器、脚本面板）留在 kit 里，不进共享层——它们是产品页专属的编辑器，patch/rack
没有这些节点。要共享的是「描述 + 控件 + hero + 卡片布局语义」，不是「所有 DOM」。

## 6. 产品层

### 6.1 `i_am_effects`（38 个）

每个产品 = DSP 类型 + `web_plugin!`。DSP 本体在 `i_am_dsp::effects`（或本 crate 里的产品级实现），
`i_am_effects` 负责注册、faceplate 选项、以及需要页面配合的额外路由（IR 上传、Rhai 脚本）。

### 6.2 `i_am_synths`（9 个）

String / Fitted / Shapes / Additive / Spectrum / Script / Canvas / Drummer / Sampler。
与效果的差别：

- 没有音频输入，有音符输入；bypass 是静音而不是直通；scope 的输入永远是静音，画面画的是「做出来的声音」。
- 预设不是单个参数能表达的（选一个音色要同时改很多参数），所以编辑器在通用命令解析器之前先处理预设选择
  （`i_am_synths/src/lib.rs` 的 `play_synth`）。
- 参数表与音符播放的转发由 `synth_product!` 生成；只有鼓机的 `set_parameter` 要顺手写 kit 的旋钮，
  所以它用 `synth_processor!` 并保留自己的参数实现。

**Fitted（I Am Modal）** 是唯一表驱动的音源：`i_am_string::fitted` 的 `MODE` 表按音存一组衰减正弦
（每个分音带 prompt/tail 两段衰减、振幅和初相），播放就是把表查出来求和，不做弦模型。表有两个来源：

- 仓库自带的五张（Gentleman / Peel / Bass / Celeste / Harp），由 `gentleman-modal/` 的 numpy 脚本拟合；
- 插件自己拟合：`i_am_string::fit` 是同一套分析流程的 Rust 版（补零 FFT 找分音、解调测两段衰减、
  一次最小二乘定振幅相位），`i_am_dsp` 的 `fit_sample` 把 WAV 变成表，`/sample` 路由收下就换成 Custom。
  最小二乘的 Gram 矩阵是阻尼正弦的几何级数，有闭式解，所以一次拟合是几百毫秒而不是几十秒；
  `cargo run -p i_am_dsp --release --example fit_sample -- sample.wav out.bin [note]` 是同一段代码的命令行入口。

一个采样只提供自己那个音；其余键在播放时按对数音高从它派生（`derive`，导入时自动打开），
所以单采样也能弹音阶。`transpose`（-24..+24 半音）整体移调，用来修拟合时差一个八度的采样。

表**不走参数表**：`Value::Serialized` 在 `ParamMap` 里是 `AtomicValue::Nothing`，界面线程写不进去。
路由把表放进 `TableSlot`（与音频线程共享），乐器每弹一个音从槽里取；存档前 `sync_params` 再把槽
拷进 `imported` 参数，所以工程里仍然带着这张表。

### 6.3 `i_am_patch` + `i_am_patch_product`（2 个产品）

```
       图模型（纯数据）              编译（纯数据）            执行
  Graph { nodes, edges }  →  Compile: 校验端口/拓扑排序/算延迟  →  Engine: 每次 BLOCK=64
  节点 kind 是稳定字符串 id                                    节点/缓冲/scope/相位
        ▲                                                          │
        │ context.events / transport                               ▼
   Poly: 把 voice group 切成每个音符一份图，voice:out 汇总
```

- 节点目录（`nodes.rs`）：`mixer`、`delay`（Lookahead，对齐用）、`lfo`、`random`、`table`、`pitch`、
  `midi`、`envelope`，以及 9 个音源节点（`additive`…`strings`，直接包装 `i_am_synths` 的产品）。
- 拆分与合并成对出现，端口同名对得上：`split:ms`/`mix:ms`（mid/side）、`split:lr`/`mix:lr`（左右各复制成双声道）、
  `split:bands`/`mix:bands`（低/中/高，直接复用 rack 的 LR4 分频器 `crossover.rs`，两个分频点可调）。
- 调制是控制率的：引擎每 64 样本取一次值；每个源槽各有自己的相位/包络/随机状态。
- 调制深度是**参数自身范围的比例**（-1..1），引擎把它加在归一化空间里：`ParamDesc::normalize/denormalize`
  对 `logarithmic` 的参数取对数，所以 cutoff 这种旋钮按比例摆动（一圈一个八度），和 rack 的调制一致；
  没有被调制的参数原样下发，不会因为归一化被夹进范围。
- LFO 可以 one-shot：跑完一个周期停在末尾（相位停在 1，不再回绕），在 voice group 里就是每个音符扫一次；
  共享的 `SourceSettings` 里是 `lfo_one_shot`，patch 节点上是 `one_shot` 开关。
- `voice:in` 的 `voices` 控制决定同时能响几个音（1–8，默认 8）：它是图参数，改一下 generation 就变，
  引擎重建时 `Poly::voices_of` 读它。
- 一个音符只起一个 voice：`PatchContext` 按 `set_sample_offset` 把 `events()` 收窄到块的第一帧，
  和宿主交给产品的语义一致；否则音源节点会在一块的每一帧都看见同一个 note-on，一个音符叠 64 层。
- `Poly` 每个音符一份完整引擎（所有节点缓冲都各一份，demo 补丁上一个 voice ≈ 276 KB、1.9% 单核），
  上限是 `POLYPHONY`（8）。页面看的是"一个"voice 的值：控制量取最新那个，音频仍然相加。
- **改一个值不重建图**：页面的 `op` 分两类，只有结构操作（增删节点/连线/曲线/调制）才 `bump()`；
  `param`、`mod_depth`、`mod_mode` 走 `i_am_patch::edit::Edit`——页面把它推进 `PatcherShared.edits`，
  音频线程在每个块边界 `try_lock` 取走（和 `clear_requests` 同一个套路），直接改引擎里的 `base`/调制表，
  所以拖旋钮时正在响的音不会被重建掉。**`Poly::set_param` 必须同时改三处**：外层引擎、活着的 voices、
  以及 `voice_graph`（下一个音符的模板）；只改前两处时，下一个音符会从旧模板出生，听起来就是"参数自己跳回去了"。
  调制深度还要按 `Poly::voice_links` 的映射换算索引（voice 图只含组内的调制，行号和外层不一样）。
- **宿主侧只有 64 个 macro 参数**（`macro1..macro64`，0..1，`i_am_patch::MACRO_COUNT`）：宿主不能事后
  加参数，所以表在构造时定死，图里用多少算多少。图里的 `macro` 节点（Control）只带一个 `index`（1..64），
  输出一根 control 线，接调制就是普通源。宏值**不进图**：产品每块把宿主/页面写进 `ParamMap` 的原子
  拷进 `Slots::macros()`（`pull_macros`，只写动过的），节点直接读那块 pad——voice 的拷贝读同一份，
  所以不欠"三处一起改"那条账。
- 控制源只有一份实现：图节点（`nodes.rs` 的 `SourceNode`）是 `SourceRuntime` 的适配器，
  参数映射到 `SourceSettings`，值、相位、同步频率、随机流和平滑都与 rack 的 slot 一致；
  `source_dsp.rs` 放两者共用的底层形状与系数。
- 宿主存档 = 一个 `Serialized` 参数 `"patch"`，内容是**行文本格式**（`node 1 synth:additive 100 0 decay_time=0.5`），
  可读、可被 agent 编辑；JSON 是真相，文本是投影。
- 页面：`/catalog`、`/patch`、`GET|POST /text`、`/scope?node=N`、`/sample?node=N`；页面编辑 bump generation，
  音频线程在块边界重建引擎。宏走 `GET /macros` 与 `POST /macro?index=&value=`，读写的就是宿主那组原子，
  所以页面拖动在宿主眼里等同自动化。
- 页面状态走 `Slots`（按节点 id：拟合表、采样、波表、冲激响应、画布、频谱页、鼓手
  click/samples/曲线、脚本）。图参数只有 f32，序列化的东西进不了图，所以页面 POST 到
  `/sample?node=N`（Modal）、`/sampler?node=N`（Sampler）、`/waves?node=N`（Shapes）、
  `/ir?node=N`（Convolver）、`/canvas?node=N`、`/spectrum?node=N`、
  `/click?node=N`、`/samples?node=N`、`/pitch?node=N`、`/amp?node=N`、`/script?node=N`。
  节点在**建的时候**就拿的是页面那个槽/句柄，所以编辑下一块就生效，不需要重建；页面用同一路径 GET 回状态。
- 节点目录只有一份：`i_am_patch/src/catalog.rs` 的 `kinds_json()` 同时喂给编辑器 example 和
  产品的 `/catalog`；产品节点的 sections/pages/heroes/switches 从产品自己的元数据派生，
  所以产品页有的字段 patch 一定有。
- 产品卡由共享的 `IAmCard.faceCard` 画（sections/pages/heroes/header/visible/switches），
  patch 只负责节点 chrome：调制环、拖放目标、端口、缩放；内置控制源仍走 `genericCard`。
- 宿主侧链：CLAP 的第二个输入端口由 `i_am_plugin` 放进 `other`，`Body` 按块交给引擎，
  图里用 `input:sidechain` 节点取用；VST3 由 clap-wrapper 把非 main 输入映射成 sidechain bus。
- 源图形（LFO/包络/随机/表）与 rack 共用 `i_am_web_gui/page/card/sources.js` 一份实现，
  样式以 Multipass 的调制卡片为准；patch 只负责把节点数据喂进去。
  LFO 的点编辑也在同一份实现里：patch 直接在卡片 hero 上拖点/双击加点/右键删点，
  与 rack 的调制卡片是同一个编辑器。

#### Voice 状态：参数、槽、feed

一个音符会新建一份完整 `Engine`，所以“页面改了，voice 跟不跟”要分三类看：

1. **参数**：走 `Poly::set_param` 的三处（外层、活 voices、`voice_graph` 模板）。这一类已经由
   测试和 `patch: a fresh modulation starts at a musical depth` 一类工作覆盖。
2. **槽**：页面和 voice 持有**同一个** `Arc<Mutex<...>>`/`TableSlot`。新 voice 建的时候拿到句柄；
   已经响的 voice 分两种语义：
   - **连续读取**（Sampler、Shapes、Convolver）：播放时读当前槽，held 也跟着变；
   - **按音符快照**（Fitted、Drummer 的 click/samples）：一个音符用出生时的材料，held 保留它，改动只影响
     下一个音符。这是采样类乐器的正常语义，不是不同步；新 voice 必须取当前快照。
3. **feed**：页面写整份图形/页/脚本，voice 自己缓存一份。它必须遵守 `i_am_synths/src/feed.rs` 的
   `VoiceFeed` 契约：
   - 新 voice 用 `current()`/`snapshot()` 出生，不能用工厂默认值；
   - 活着的 voice 用 `generation()`/`changed()` 轮询，变了就重新加载；
   - **不许用一次性 `pending`**：队列只能被一个 voice 取走，其余 voice 会停在旧状态。
   Canvas、Spectrum、Script 目前都按 generation 广播；Canvas/Script 的一次性 `pending` 已删除。

所以“序列化对象不能进 `ParamMap`”本身不是同步问题的根因；即使塞进去，每个 voice 仍是独立实例。
真正要保证的是上面两条 voice 契约。每个 slot/feed 产品都应有两条测试：**改完，held voice 跟不跟**，
以及**改完，新 voice 从新状态出生吗**。

### 6.4 `i_am_rack`（I Am Multipass）

```
input ─► LR4 分频 ─┬─ Low  ─► [串行模块…] ─► gain/pan/mute/solo ─┐
                   ├─ Mid  ─► [串行模块…] ─► …                 ├─► sum ─► output
                   ├─ High ─► [串行模块…] ─► …                 │
                   └─ Full ─► [串行模块…] ─► …                 ┘
                          ▲
                 调制矩阵：LFO/包络/随机/表/音高/MIDI → 模块参数与 lane 控制
```

- 模块目录 38 种，几乎一个效果产品对应一种（Filter/Distortion/Compressor/Gate/Expander/Limiter/Gain/Delay/
  Comb/Allpass/Chorus/Flanger/Phaser/Crusher/Disperser/Width/Balance/Router/TapeStop/Transient/Formant/Eq/
  Reverb/Schroeder/ParticleWider/Pitch/Shifter/Autotune/…），实现直接复用 `i_am_effects`。
- 宿主只自动化 **8 个 macro + 32 个自动化槽**；其余全部住在序列化的 `RackState`（参数 `"rack_state"`）里，
  所以工程存档、预设、A/B 比较都不需要新增宿主参数。
- 页面：`/rack`、`/state`、`/modulation`、`POST /rack-command`、模块级 `/module-scope`、
  `/module-bands`、`/module-ir` 等。撤销/重做与预设导入放在状态旁边的共享内存里。
- 音频线程只在 dirty 时拿锁；稳态是一次原子交换，没有锁。

### 6.5 `i_am_string`（物理建模）

拨、击、弓的差别只在激励点的接触律（单边位移约束 / 非线性毡力 / 速度摩擦），其余（色散损耗弦、
端点、琴体）共享。弦本身有两种模型，`core::Core` 在两者间切换：

| 模型 | 原理 | 代价 | 音准 |
| --- | --- | --- | --- |
| 波导 | 两条延迟线 + 色散全通 + 损耗滤波 | 低 | 低音分准确，高音分最多偏 45 cent |
| 模态组 | 每个分音一个阻尼振子 | 约 8× 波导 | 任意劲度下 0.2 cent 以内 |

`StringEngine` 作为 generator 接入 `i_am_dsp`（`string_engine.rs`），String 产品就是它的面板。

### 6.6 打包：`i_am_bundler`

`cargo bundle -p <bundle-crate> [--features vst3] [--no-vst3]`（`.cargo/config.toml` 里的 alias）：

1. 读 `[package.metadata.i_am_dsp]`（name / vendor / id，可被命令行覆盖）；
2. 检查库是否真的导出了 VST3 入口（有 `vst3` feature 才有），没有就拒绝写 `.vst3`；
3. 写 `<name>.clap`（Windows/Linux 单文件）与 `<name>.vst3`（目录，或 `--single-file`）。

## 7. DSP 清单（`i_am_dsp`）

- **effects**：filter（SVF/comb）、multimode、eq、compressor、dynamics（gate/expander/limiter/transient）、
  distortion、crusher、phaser、delay（含 chorus/flanger）、tape_stop、reverb、schroeder、hybrid_reverb、
  reverb_ir_gen、convolver（FFT）、vocoder、phase_vocoder、stft_vocoder、pitch_shifter、psola、
  spectral_retune、autotune、freq_shifter、hilbert_transform、disperser、resonator、formant_filter、
  ott、particle_wider、stereo_control、router、unstablizer、visual。
- **generators**：wavetable（分帧/振荡器/表/平滑）、additive（含 bended/drawn/spectral）、
  drum（曲线/音频源）、sampler、adsr、stereo_generator、string_generator、string_engine、fitted_engine。
- **滤波器设计**：所有二阶滤波器的 RBJ 设计集中在 `i_am_dsp/src/effects/biquad_design.rs`；
  `filter`、`eq`、`ott` 共用同一组公式，只有 α 的参数化（Q/带宽/斜率）与钳制各自不同。
- **tools**：`ring_buffer`（含 WaveTable 插值）、`wsola`、`pitch`、`beat_clock`、`bpm_syncer`（deprecated）、
  `synced_time`、`smoother`、`matrix`、`interpolate`、`pcm_data`（WAV 编解码）、`audio_io_chooser`、
  `rhat_related`（Rhai 脚本产品用）。

## 8. 构建、测试与调试

```bash
cargo build                                  # 默认 features：string + layout
cargo test                                   # 694 个测试，含 rack/patcher 的页面路由测试
cargo clippy --all-targets                   # README 要求零警告
cargo bundle -p i_am_synths_bundle --features vst3
cargo run -p i_am_dsp --features standalone  # egui + cpal 的实时 demo
cargo run -p i_am_effects --example dump_face -- target/web-preview gain   # 把面板导出成静态页
node tools/preview/preview.mjs product gain   # 一个入口启动预览服务器（cards/product/rack/patch）
```

features（`i_am_dsp/Cargo.toml`）：`layout`（布局元数据）、`string`（物理建模）、`rhai`（脚本音源）、
`real_time_demo` / `standalone`（egui + cpal）、`room_gen`（parry3d）。`i_am_plugin` 的 `vst3` 会传递到 bundle。

调试钩子：

- `I_AM_PLUGIN_TRACE_PARAMS=1` 打开参数事件日志（音频线程，默认关；开了就是每块一次文件写）。
- `I_AM_PLUGIN_WRY_LOG_BYTES` 调 WebView2 日志上限；GUI 打不开时看 `%TEMP%\i_am_plugin_wry\last-error.log`
  与 `open.log`。
- WebView2 需要可写 profile：默认 `%LOCALAPPDATA%\i_am_plugin_wry\WebView2-v2`，
  `I_AM_PLUGIN_WRY_PER_PROCESS_PROFILE=1` 可恢复每进程目录。

## 9. 不变量（改代码前先读）

1. **参数表与插件同寿**。换 DSP 对象可以，换 `ParamMap` 不行——宿主的自动化跟着旧表走。
2. **稳定 id 永不变**。只追加、不重排；已发布的 id 用 `#[id(stable = N)]` 钉死。
   新 id 必须在 31 位内且不等于 `u32::MAX`。
3. **音频线程不做**：分配、加锁、I/O、日志（除显式 trace）、panic 展开。跨线程只走原子/环形缓冲/预分配通道。
4. **跨线程共享状态必须内部可变**。音频线程持 `&mut P` 的同时主线程能拿 `&P`（`on_message`）。
5. **界面写参数走 `ParamMap` 原子**，不走消息通道；消息只用于「动作」（预设、清尾巴、上传）。
   界面写完后宿主可能立刻存档，所以 `sync_params()` 在 `save` 前会被调一次。
6. **一个可加载库一个 `clap_entry`**。要复用实现就把产品放普通 crate，别把入口塞进去。
7. **构造时不知道采样率**。系数依赖采样率的效果必须在 `on_activate` 重建，且保住参数表。
8. **denormal 要关**。包装层在每次 `process` 入口都会调 `enable_flush_to_zero()`（x86 的 MXCSR /
   aarch64 的 FPCR，`i_am_plugin/src/context.rs`）；自己接宿主或自建音频线程时记得自己调一次，
   否则混响/卷积尾巴会慢一个数量级。
9. **`Value::Serialized` 是存档通道**，不是普通参数：界面不编辑它，宿主只负责搬运。
10. **identifier 是唯一真相**。位置只用于 `set_parameter_by_index`，宏内部不按位置派发。

## 10. 已知边界与待办

- **macOS 的后端没有真机验证**：Windows（WebView2）与 Linux（WebKitGTK，仅 X11，已在 WSLg 里跑通
  `cargo run -p i_am_plugin --example wry_standalone`：页面加载、IPC 都通）之外，macOS 的 WKWebView
  路径只是代码齐全；再往外的平台是编译得过、打开报错的 stub。Linux 上页面靠宿主的
  `clap_plugin_timer` 才有自己的事件循环，宿主不提供 timer 扩展时页面只会画一帧。
- **SIMD 只到 `wide` 那几处**（filter、convolver、additive、`i_am_string` 的模态与共鸣）；逐样本的 biquad 仍是标量，
  按时间 4 个样本一组的转置形式没有做（`effects/filter/process.rs` 里只有一行注释）。
- **无动态端口**、**无侧链自动声明**（侧链靠 `INPUT_PORTS` 手工列出，音频侧走 `other`）。
- **参数同步是 O(n)/块**，没有变化队列；参数特别多的产品（rack、patcher）靠「少量宿主参数 + 序列化状态」把 n 压住。
- **standalone 只是 demo**，不是完整应用；两条 standalone 线拿不到宿主 tempo（见 `bpm-sync-research.md`）。
- **AU 只导出符号**，打包与验证未做。
- **延迟补偿**只做到「变化时告诉宿主」，依赖宿主实现 PDC。
- **UI 共享到「卡片布局」为止**：产品页的页面级编辑器（sampler/IR/脚本等）是产品页专属，
  不抽进共享层；卡片页与产品页共享描述、控件、hero 和 `faceCard` 布局语义（见 5.5）。
- **无 CI**；测试是本地 `cargo test` + clippy。
- 上面这些是**产品级**缺口；库级别「要做什么」见 [README.MD](README.MD) 的 TODO，两边都写「现在缺什么」，不是愿望单。

---

*本文由代码阅读整理，行数与测试数为撰写时统计（2026-11）。改了结构记得回来改这里。*
