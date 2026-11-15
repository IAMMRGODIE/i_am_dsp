# Local wry patch

This is wry 0.57.1, copied from crates.io and patched for use inside audio
plug-ins. Its Windows backend also carries the two fixes wry 0.57.2 made to the
same file, so the WebView2 side behaves like 0.57.2 while the rest of the crate
is still 0.57.1.

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

## The 0.57.2 fixes, ported

0.57.2 changed the same file in two places, both about a webview built as a
child of a host's window, which is exactly how a plug-in's editor is built:

- `Drop` only detaches the parent's window subclass when the webview is not a
  child. A child webview never attached one, and detaching the host's subclass
  took the host's own handling away with it.
- `WM_ENTERSIZEMOVE` no longer moves focus. The message arrives while the user
  drags a window, and a child webview should not steal focus in the middle of
  it.

Both are applied here verbatim. The rest of 0.57.2 is Android, macOS and iOS
work that this crate never compiles.

## Updating

To take a later wry, copy the crates.io source over this directory and re-apply
the class-name change above (three places in src/webview2/mod.rs) plus whatever
of the child-window behaviour the new version already has; do not copy 0.57.2's
two hunks onto a version that has them.
