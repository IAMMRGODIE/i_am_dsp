# I am DSP

一个简单的 rust Dsp 库，针对写 Vst/Clap 等插件场景编写，用起来或许好用吧。

## TODO

- [ ] 用 crates.io 上的 `iced_baseview` 替换本地的 `iced_baseview` 依赖
- [ ] 完善 README
- [ ] 完善 example
- [ ] 视觉效果器
- [ ] BPM sync
- [ ] 更完善的 SIMD 支持
- [ ] 多写一点例子
- [ ] 物理建模（如果我有精力研究的话）

## 关于编译

如果你想要编译 CLAP 插件的话，你可以用 `i_am_plugin` 这个库。

但是因为开发方便，目前 `i_am_plugin` 里的 `Cargo.toml` 写的是本地的依赖，你需要手动将将 `iced_baseview` 换成[这里](https://github.com/IAMMRGODIE/iced_baseview)的对应实现

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

不过需要注意一下开源协议问题，因为我在这里并没有分发一份 `clap_wrapper`, 所以我无法固定 `clap_wrapper` 的协议

## 开源协议

MPL 2.0