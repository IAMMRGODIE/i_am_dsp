# I am DSP

一个简单的 rust Dsp 库，针对写 Vst/Clap 等插件场景编写，用起来或许好用吧。

## TODO

- [ ] 完善 README
- [ ] 完善 example
- [ ] 视觉效果器
- [ ] BPM sync
- [ ] 更完善的 SIMD 支持
- [ ] 多写一点例子

## 关于编译

如果你想要编译 CLAP 插件的话，你可以用 `i_am_plugin` 这个库。

但是因为开发方便，目前 `i_am_plugin` 里的 `Cargo.toml` 写的是本地的依赖，你需要手动将 `clack` 相关的依赖换成
[这里](https://github.com/prokopyl/clack)的对应库，以及将 `iced_baseview` 换成[这里](https://github.com/IAMMRGODIE/iced_baseview)的对应实现

然后你需要为你的插件实现 `Plugin` trait，之后使用 `export_clap` 这个宏创建 entry，然后就可以愉快 
`cargo build --release` 了，把输出的库文件换成 `.clap` 后缀名放到 CLAP 要求的地方应该就能使了 