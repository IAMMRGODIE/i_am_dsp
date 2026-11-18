# I am DSP

一个简单的 rust Dsp 库，针对写 Vst/Clap 等插件场景编写，用起来或许好用吧。

## TODO

- [ ] **macOS 的面板验证**。Linux 已经验证过（WSL2 + WSLg 的 Kali：X11 嵌入、页面加载、IPC 都通），
  macOS 仍然只是代码齐全（WKWebView 作为宿主 `NSView` 的 subview）。Linux 上页面的事件循环靠宿主的
  `clap_plugin_timer`，没有 timer 的宿主页面只会画一帧；Wayland 不支持嵌入，后端会直接报错。
- [ ] **参数变化队列**。宿主参数每块 O(n) 扫一遍 `ParamMap`；现在靠「少量宿主参数 + 序列化状态」把 n 压住，
  参数再多一点的产品就会撞上这条。
- [ ] **AU 的打包与验证**。`PluginAuExt` 只导出符号，没打包过，也没在 Logic 里跑过。
- [ ] **页面拖动的手势事件**。页面改参数只写原子，不发 `ParamGestureBegin/End`，宿主不会为手动调整暂停自动化。
- [ ] **侧链的自动声明**。侧链端口靠 `INPUT_PORTS` 手工列出，宿主不知道哪条是侧链。
- [ ] **README**：最小可跑示例（现在只有构建与打包说明）、`i_am_effects` 里各产品的一句话功能表。

## 一个警告

`i_am_dsp` 还在开发中，随时可能有不经告知的破坏性改动（因为有的时候我自己也不知道）

另外我没有开一个 dev 分支，可以说习惯非常的坏了.

项目有 Agent 参与，如果你膈应的话可以去考虑隔壁 [nice-plug](https://codeberg.org/RustAudio/nice-plug). 或者 [Rust Audio](https://rust.audio/) 那边的，他们严格禁 AI（

## 关于编译

如果你想要编译 CLAP 插件的话，你可以用 `i_am_plugin` 这个库。
然后你需要为你的插件实现 `Plugin` trait，之后使用 `export_plugin` 这个宏创建 entry，然后就可以愉快 
`cargo build --release` 了，把输出的库文件换成 `.clap` 后缀名放到 CLAP 要求的地方应该就能使了 

哦对不要忘了加入下面这玩意

```toml
# Cargo.toml

[lib]
crate-type = ["cdylib"]
```

如果你想导出 VST3 的话，你可以试试 [clap_wrapper](https://docs.rs/clap-wrapper/latest/clap_wrapper/) 这个库，然后在你的插件里面加上

```rust
clap_wrapper::export_vst3!(); 
```

之后你需要参考对应 VST3 的文档去打包，不过目前我还没试过（

AU 的话你需要额外实现一个叫做 `PluginAuExt` 的 trait，然后你就可以通过

```rust
clap_wrapper::export_auv2!(); 
```

来导出 Au 的 Symbol 了，之后你还是要参考 Au 的文档去打包，不过目前我也没试过（

界面是网页面板：Windows 用 WebView2（系统自带），Linux 用 WebKitGTK（要装开发包：
`libwebkit2gtk-4.1-dev`、`libgtk-3-dev`、`pkg-config`），macOS 用系统自带的 WKWebView。
Linux 的嵌入只支持 X11，Wayland 下打开编辑器会报错。

哦对了现在有一个打包工具了在 [i_am_bundler](./i_am_bundler/README.md) 那边，还做了安装，应该会方便些

打包工具是基于前面提到的 [clap_wrapper](https://docs.rs/clap-wrapper/latest/clap_wrapper/) 的
不过由于事件约定的不同，`clap_wrapper` 导出的 vst3 的参数自动化似乎是每 block update 一次而不是
每 sample update 一次，可以说非常的坏了.

为此在包装层加了一个可选的 `smoothing` feature：插件点名几个连续参数（`Plugin::SMOOTHED_PARAMETERS`，
`web_plugin!` 里是 `smoothed = &[...]`），包装层就会把每块的跳变按样本摊平，按钮式的参数
不受影响。代价是被点名的参数最多滞后一个 ramp 时长。做法与边界见 [ARCHITECTURE.md](ARCHITECTURE.md) 的 3.6。

```bash
cargo bundle -p i_am_effects_bundle --features vst3,smoothing
```

## 贡献

开 PR 就行，顺便我们用的是 `cargo clippy`，所以交上来的时候记得清掉 clippy warning，还有 cargo doc 的 warning.

## 开源协议

MPL 2.0