# i_am_bundler

Turns a built `i_am_plugin` library into the on-disk layouts audio hosts load.

A CLAP plug-in is just the built cdylib with a `.clap` name. A **VST3** plug-in is a
directory bundle, and it only works at all if the library exports the VST3 entry points
`GetPluginFactory` / `InitDll` / `ExitDll` - which `i_am_plugin` adds when its `vst3`
feature is on (it links [clap-wrapper](https://crates.io/crates/clap-wrapper) and the
VST3 SDK into your library). This tool does the second half: it builds the plug-in,
then writes the bundle.

## Usage

From the workspace root, through the `cargo bundle` alias (`.cargo/config.toml`):

```sh
cargo bundle -p plugin_example --features vst3
cargo bundle -p plugin_example --features vst3 --install
```

Or directly:

```sh
cargo run -p i_am_bundler -- --manifest-path plugin_example/Cargo.toml --features vst3
```

`i_am_bundler --help` lists every option. The useful ones:

| Option | What it does |
| --- | --- |
| `-p, --package NAME` | Cargo package to build and bundle |
| `--features vst3` | Passed to `cargo build`; required to get VST3 entry points |
| `--debug` | Build the debug profile instead of release |
| `--no-build` | Bundle the library that is already there |
| `--single-file` | Write `NAME.vst3` as one renamed library (Windows only) |
| `--install` / `--install-system` | Copy into the user / system plug-in directory |
| `--force` | Write the VST3 bundle even without `GetPluginFactory` |
| `--name`, `--vendor`, `--version`, `--id` | Override the manifest metadata |

## Metadata

Bundle names come from the plug-in crate's manifest, and every field can be overridden
on the command line:

```toml
[package.metadata.i_am_dsp]
name = "I Am Table Synth"                  # defaults to the package name
vendor = "iamplugins"                      # optional, used for the default bundle id
version = "0.1.0"                          # defaults to the package version
id = "iamdsp.example.wavetable.synth"      # used by the macOS Info.plist
```

## What gets written

| Platform | VST3 | CLAP |
| --- | --- | --- |
| Windows | `Name.vst3/Contents/x86_64-win/Name.vst3` | `Name.clap` |
| Linux | `Name.vst3/Contents/x86_64-linux/Name.so` | `Name.clap` |
| macOS | `Name.vst3/Contents/MacOS/Name` + `Info.plist` | `Name.clap/Contents/MacOS/Name` + `Info.plist` |

`--single-file` writes `Name.vst3` as a single renamed library instead of a bundle.
Steinberg deprecates that form, but plenty of hosts still load it.

Before writing anything the tool reads the library's export table (PE on Windows) and
stops with an explanation if `GetPluginFactory` is missing, because a `.vst3` bundle
around a CLAP-only library silently does nothing in every DAW.

## Install locations

| Platform | VST3 | CLAP |
| --- | --- | --- |
| Windows (user) | `%LOCALAPPDATA%\Programs\Common\VST3` | `%LOCALAPPDATA%\Programs\Common\CLAP` |
| Windows (system) | `%CommonProgramFiles%\VST3` | `%CommonProgramFiles%\CLAP` |
| macOS (user) | `~/Library/Audio/Plug-Ins/VST3` | `~/Library/Audio/Plug-Ins/CLAP` |
| macOS (system) | `/Library/Audio/Plug-Ins/VST3` | `/Library/Audio/Plug-Ins/CLAP` |
| Linux (user) | `~/.vst3` | `~/.clap` |
| Linux (system) | `/usr/lib/vst3` | `/usr/lib/clap` |
