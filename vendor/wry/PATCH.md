# Local wry patch

This is wry 0.57.1, copied from crates.io and patched for use inside audio
plug-ins.

## Why

wry 0.57.1 registers a single process-global window class named WRY_WEBVIEW,
with a window procedure that lives in the plug-in DLL. A host that unloads the
plug-in DLL while the process stays alive leaves the class registered with a
dangling procedure. The next WebView (from another plug-in, or from a reloaded
copy of the same one) then crashes on its first window message with
0xc000041d.

This is exactly what FL Studio does through its Fruity Wrapper: it can unload
and reload a plug-in DLL inside FL64.exe.

## Change

In src/webview2/mod.rs, every container window gets a unique class name built
from the module base, a time nonce and a sequence number. The class is
registered under the module that owns the window procedure and is unregistered
when the WebView is dropped. The public wry API is unchanged.

The rest of the crate is untouched. See PATCH.md in the repository root for how
to update the patch when wry is upgraded.
