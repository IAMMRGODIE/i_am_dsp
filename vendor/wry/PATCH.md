# Local wry patch

This is wry 0.57.2, copied from crates.io and patched for use inside audio
plug-ins.

## Why

wry registers a single process-global window class named WRY_WEBVIEW,
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

## Updating

To take a later wry, copy the crates.io source over this directory and re-apply
the class-name change (the `class_name` fields, the unique class in
`create_container`, and the `UnregisterClassW` calls in its cleanup paths and
in `Drop`). The child-window behaviour 0.57.2 introduced is upstream now, so
nothing else has to be carried by hand.
