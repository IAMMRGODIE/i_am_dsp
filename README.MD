# I am DSP

一个简单的 rust Dsp 库，针对写 Vst/Clap 等插件场景编写，用起来或许好用吧。

## TODO

- [x] 用 crates.io 上的 `iced_baseview` 替换本地的 `iced_baseview` 依赖
- [ ] 完善 README
- [ ] 完善 example
- [ ] 视觉效果器
- [x] BPM sync
- [ ] 更完善的 SIMD 支持
- [ ] 多写一点例子
- [ ] 物理建模（如果我有精力研究的话）
- [ ] 做完 `i_am_effects`

## 一个警告

`i_am_dsp` 还在开发中，随时可能有不经告知的破坏性改动（因为有的时候我自己也不知道）

另外我没有开一个 dev 分支，可以说习惯非常的坏了.

项目有 Agent 参与，如果你膈应的话可以去考虑隔壁 [nice-plug](https://codeberg.org/RustAudio/nice-plug). 或者 [Rust Audio](https://rust.audio/) 那边的，他们严格禁 AI（

## 关于编译

如果你想要编译 CLAP 插件的话，你可以用 `i_am_plugin` 这个库。
然后你需要为你的插件实现 `Plugin` trait，之后使用 `export_clap` 这个宏创建 entry，然后就可以愉快 
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

哦对了现在有一个打包工具了在 [i_am_bundler](./i_am_bundler/README.md) 那边，还做了安装，应该会方便些

## 贡献

开 PR 就行，顺便我们用的是 `cargo clippy`，所以交上来的时候记得清掉 clippy warning.

## 开源协议

MPL 2.0