"""Observed actions through Browser Harness; one CDP session, no per-step subprocess."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

from browser_harness.admin import ensure_daemon
from browser_harness.helpers import cdp

# Atomically read visible content and controls, preserving actual DOM node identity.
READ_STATE = Path(__file__).with_name("snapshot.js").read_text()
MARKER = f"(() => {{ const state={READ_STATE}; return state?.marker ?? null; }})()"
# The marker plus the page's own word on whether it is still loading: WAI-ARIA `aria-busy="true"`
# marks a region whose content is being updated (skeletons, streamed panels, pending saves).
STILLNESS = (
    f"(() => {{ const state={READ_STATE}; "
    "return state ? [state.marker, !!document.querySelector('[aria-busy=\"true\"]')] : null; })()"
)
# How long an interaction may take to show its effect before the page is judged unchanged.
# Client-side routers and streamed content typically land a few hundred milliseconds after the
# input; an unchanged marker read sooner than that records a false `page_changed: False` and
# invites a repeat of the same action, each one restarting the same navigation.
SETTLE_MS = int(os.environ.get("JEV_SETTLE_MS", "1500"))
# Emulated viewport, "WIDTHxHEIGHT". The default is a laptop pane; a narrow value exercises
# responsive layouts (collapsed navigation, stacked forms) with the same observed action space.
VIEWPORT = tuple(int(v) for v in os.environ.get("JEV_VIEWPORT", "1120x780").lower().split("x"))

class StalePage(ValueError):
    """A decision no longer refers to the observed page."""


class Browser:
    def __init__(self, url):
        ensure_daemon()
        self.target = cdp("Target.createTarget", url="about:blank", background=True)["targetId"]
        self.session = cdp("Target.attachToTarget", targetId=self.target, flatten=True)["sessionId"]
        width, height = VIEWPORT
        self.call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=False)
        # Keep rAF/menus rendering in an owned background tab, without activating the user's Chrome tab.
        self.call("Emulation.setFocusEmulationEnabled", enabled=True)
        # A native file picker would stall the tab. Files are attached through DOM.setFileInputFiles instead.
        self.call("Page.setInterceptFileChooserDialog", enabled=True)
        self.call("Page.navigate", url=url)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if self.evaluate("document.readyState") == "complete":
                break
            time.sleep(0.02)
        self.quiesce()

    def call(self, method, **params):
        return cdp(method, session_id=self.session, **params)

    def evaluate(self, expression):
        response = self.call("Runtime.evaluate", expression=expression, returnByValue=True)
        if response.get("exceptionDetails"):
            raise StalePage("Document changed during evaluation")
        return response.get("result", {}).get("value")

    def observe(self, screenshot=True):
        if getattr(self, "after_input", None):
            (action, marker), self.after_input = self.after_input, None
            if action["kind"] != "wait":
                self.render(action)
            # Every action, WAIT included, settles the same way: WAIT means "until the page moves".
            # The page is then given the chance to finish moving: a committed client-side
            # navigation is an empty shell until its content streams in. A page that did not
            # move is still by definition, so this costs it one more read.
            self.settle(marker)
            self.quiesce()
        for attempt in range(10):
            try:
                return browser_operation(
                    {"operation": "observe", "session": self.session, "screenshot": screenshot}
                )
            except StalePage:
                if attempt == 9:
                    raise
                time.sleep(0.02)
        raise StalePage("Page did not settle")

    def render(self, action):
        """Give the input up to two frames (or visible autocomplete options) to paint.

        This is read-only and happens after execution was logged, even if navigation interrupts it."""
        try:
            self.call(
                "Runtime.evaluate",
                expression="""(action => new Promise(resolve => {
                  const field=window.__jevFast?.nodes.get(action.node);
                  const autocomplete=action.kind==='fill' && field?.getAttribute('role')==='combobox';
                  let frames=0, stopped=false;
                  const finish=()=>{stopped=true;resolve()};
                  setTimeout(finish,autocomplete ? 200 : 50);
                  const ready=()=>{
                    if (stopped) return;
                    const ids=(field?.getAttribute('aria-controls')||field?.getAttribute('aria-owns')||'')
                      .split(/\\s+/).filter(Boolean);
                    const roots=ids.length ? ids.map(id=>document.getElementById(id)).filter(Boolean) : [document];
                    const options=roots.flatMap(root=>[...root.querySelectorAll('[role="option"]')]);
                    if (++frames>=2 && (!autocomplete || options.some(e=>{
                      const r=e.getBoundingClientRect();
                      return r.width && r.height && r.bottom>0 && r.top<innerHeight &&
                        e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true});
                    }))) finish();
                    else requestAnimationFrame(ready);
                  };
                  requestAnimationFrame(ready);
                }))(""" + json.dumps(action) + ")",
                awaitPromise=True,
                returnByValue=True,
            )
        except RuntimeError:
            pass

    def quiesce(self):
        """Wait for a loaded or changed page to stop changing before it is observed.

        `readyState === 'complete'` and a committed navigation both precede streamed content and
        hydration on most modern sites; an observation taken then sees the shell, not the page.
        Two identical marker reads 100 ms apart count as still — unless the page shows nothing
        yet or declares a region `aria-busy`, which is a shell until the budget (SETTLE_MS) says
        otherwise."""
        deadline = time.monotonic() + SETTLE_MS / 1000
        previous = None
        while time.monotonic() < deadline:
            try:
                current, busy = self.evaluate(STILLNESS) or (None, True)
            except StalePage:
                current, busy = None, True
            # marker = [timeOrigin, href, scrollX, scrollY, w, h, title, text, semantics, forms, box scroll]
            if not busy and current is not None and current == previous and (current[7] or current[8]):
                return True
            previous = current
            time.sleep(0.1)
        return False

    def settle(self, marker):
        """Return as soon as the page differs from `marker`, or once SETTLE_MS pass without a change.

        The fast path (the action already changed the page) costs one read. Only an action with no
        visible effect pays the full budget, which is exactly when a truthful `page_changed: False`
        matters. A document swap mid-poll reads as a change."""
        deadline = time.monotonic() + SETTLE_MS / 1000
        while time.monotonic() < deadline:
            try:
                if self.evaluate(MARKER) != marker:
                    return True
            except StalePage:
                return True
            time.sleep(0.05)
        return False

    def fresh(self, page, action=None):
        if action is not None and action["kind"] in {"click", "select", "upload", "enter"}:
            node = action["node"]
            if type(node) is not int:
                return False
            current = self.evaluate(
                "(() => { const c=window.__jevFast; "
                f"return c ? [c.pageKey(),c.guard(c.nodes.get({node}))] : null; }})()"
            )
            return current == [page["page_key"], page["guards"].get(str(node))]
        return self.evaluate(MARKER) == page["marker"]

    def act(self, action, page, text=None):
        if not self.fresh(page, action):
            raise StalePage("Page changed since this decision. Observe again.")
        result = browser_operation({"operation": "act", "session": self.session, "action": action, "text": text})
        # Every action, WAIT included, is followed by the settle read: WAIT means "until the page moves".
        self.after_input = (action, page["marker"])
        return result

    def close(self):
        if self.target:
            cdp("Target.closeTarget", targetId=self.target)
            self.target = None


def fingerprint(state):
    content = {k: state[k] for k in ("url", "text", "actions", "scroll")}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def browser_operation(request):
    operation = request["operation"]
    session = request["session"]

    def call(method, **params):
        return cdp(method, session_id=session, **params)

    def evaluate(expression):
        result = call("Runtime.evaluate", expression=expression, returnByValue=True)
        if result.get("exceptionDetails"):
            if operation == "act" and request["action"]["kind"] == "select":
                raise RuntimeError("Dropdown execution was interrupted; inspect before retrying.")
            raise StalePage("Document changed during evaluation")
        return result.get("result", {}).get("value")

    if operation == "act":
        action = request["action"]
        kind = action["kind"]
        if kind == "scroll":
            # The wheel lands where the snapshot put it: over the overflow container hiding the
            # most content, or the middle of the viewport when the document itself scrolls.
            call(
                "Input.dispatchMouseEvent",
                type="mouseWheel",
                x=action["x"],
                y=action["y"],
                deltaX=0,
                deltaY=action["delta"],
            )
        elif kind == "upload":
            if type(action["node"]) is not int or not action.get("files"):
                raise ValueError("Invalid observed node")
            # Resolve the observed input to a live object handle; the paths come from the caller, never the model.
            result = call("Runtime.evaluate", expression=f"""(() => {{
              const e=window.__jevFast?.nodes.get({action["node"]});
              return e?.isConnected && e.tagName==='INPUT' && e.type==='file' && !e.matches(':disabled') &&
                !e.closest('[aria-disabled="true"],[inert]') ? e : null;
            }})()""")
            handle = result.get("result", {}).get("objectId")
            if result.get("exceptionDetails") or not handle:
                raise StalePage("File input changed or is unavailable. Observe again.")
            try:
                call("DOM.setFileInputFiles", files=action["files"], objectId=handle)
            except RuntimeError as error:
                # The change event may already have fired; do not let transport recovery attach twice.
                raise RuntimeError(f"Upload was not confirmed ({error}); inspect before retrying.") from None
            call("Runtime.releaseObject", objectId=handle)
        elif kind != "wait":
            if type(action["node"]) is not int:
                raise ValueError("Invalid observed node")
            # Code-owned node IDs refer to actual observed elements, never model-generated selectors.
            target = evaluate("""(action => {
              const e=window.__jevFast?.nodes.get(action.node);
              if (!e?.isConnected || e.matches(':disabled') || e.closest('[aria-disabled="true"],[inert]') ||
                  !e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true})) return null;
              if (action.kind==='fill' && (e.readOnly || e.getAttribute('aria-readonly')==='true')) return null;
              const r=e.getBoundingClientRect(), x=r.x+r.width/2, y=r.y+r.height/2;
              if (!r.width || !r.height || x<0 || y<0 || x>=innerWidth || y>=innerHeight) return null;
              if (!e.contains(document.elementFromPoint(x,y))) return null;
              if (action.kind==='select') {
                if (e.tagName!=='SELECT' || ![...e.options].some(o=>o.value===action.value &&
                    !o.disabled && !o.closest('optgroup[disabled]'))) return null;
                e.value=action.value;
                e.dispatchEvent(new Event('input',{bubbles:true}));
                e.dispatchEvent(new Event('change',{bubbles:true}));
              }
              return {x,y};
            })(""" + json.dumps(action) + ")")
            if target is None:
                if kind == "select":
                    raise RuntimeError("Dropdown execution was not confirmed; inspect before retrying.")
                raise StalePage("Target changed or is covered. Observe again.")
            if kind != "select":
                x, y = target["x"], target["y"]
                for event in ("mousePressed", "mouseReleased"):
                    call("Input.dispatchMouseEvent", type=event, x=x, y=y, button="left", clickCount=1)
                if kind == "enter":
                    # The click above focused the field; Enter submits it like a keyboard user would.
                    call(
                        "Input.dispatchKeyEvent",
                        type="keyDown",
                        key="Enter",
                        code="Enter",
                        windowsVirtualKeyCode=13,
                        text="\r",
                    )
                    call("Input.dispatchKeyEvent", type="keyUp", key="Enter", code="Enter", windowsVirtualKeyCode=13)
                if kind == "fill":
                    call(
                        "Input.dispatchKeyEvent",
                        type="keyDown",
                        key="a",
                        code="KeyA",
                        modifiers=4 if sys.platform == "darwin" else 2,
                        commands=["selectAll"],
                    )
                    call(
                        "Input.dispatchKeyEvent",
                        type="keyUp",
                        key="a",
                        code="KeyA",
                        modifiers=4 if sys.platform == "darwin" else 2,
                    )
                    if request["text"]:
                        call("Input.insertText", text=request["text"])
                    else:
                        # An empty replacement clears the field: insertText("") leaves the selection as is.
                        for event in ("keyDown", "keyUp"):
                            call(
                                "Input.dispatchKeyEvent",
                                type=event,
                                key="Backspace",
                                code="Backspace",
                                windowsVirtualKeyCode=8,
                            )
        return {"executed": action["id"]}

    info = evaluate(READ_STATE)
    if info is None:
        raise StalePage("Document is navigating")
    info["fingerprint"] = fingerprint(info)
    if request.get("screenshot", True):
        info["screenshot"] = call("Page.captureScreenshot", format="jpeg", quality=72)["data"]
    return info
