"""Local-browser freshness/execution regressions. No model calls or external websites."""

import tempfile
import time
from pathlib import Path
from urllib.parse import quote

from jev_ultrafast.browser import Browser, StalePage

HTML = """<!doctype html><title>Guard checks</title>
<style>body{margin:30px}button{width:180px;height:50px}#outside{position:absolute;top:3000px}</style>
<p id="context">Cart total: $10</p>
<button id="target" onclick="window.clicks=(window.clicks||0)+1">Continue</button>
<label>City<input id="field" value="Zurich"></label>
<label><input id="toggle" type="checkbox">Refundable</label>
<select aria-label="Category"><option>All</option><option>Design</option></select>
<p id="outside">Unrelated offscreen text</p>"""


def main():
    browser = Browser("data:text/html," + quote(HTML))
    passed = []
    try:
        page = browser.observe(screenshot=False)
        action = next(a for a in page["actions"] if a["label"] == "Continue")
        browser.evaluate("document.querySelector('#target').style.transform='translateX(200px)'")
        assert browser.fresh(page), "Movement should use fresh geometry, not another model call"
        browser.act(action, page)
        assert browser.evaluate("window.clicks") == 1
        passed.append("moving target clicked at its current location")

        browser.evaluate("document.querySelector('#outside').textContent='Updated outside the viewport'")
        assert browser.fresh(page)
        passed.append("unrelated offscreen text does not invalidate")

        mutations = {
            "visible context": "document.querySelector('#context').textContent='Cart total: $100'",
            "accessible label": "document.querySelector('#target').setAttribute('aria-label','Delete account')",
            "field property": "document.querySelector('#field').value='London'",
            "checkbox property": "document.querySelector('#toggle').checked=true",
            "disabled target": "document.querySelector('#target').disabled=true",
            "read-only field": "document.querySelector('#field').readOnly=true",
            "hidden target": "document.querySelector('#target').style.display='none'",
            "replaced node": "document.querySelector('#target').outerHTML=document.querySelector('#target').outerHTML",
            "dropdown option": "document.querySelector('select').options[1].text='Coastal'",
        }
        for label, expression in mutations.items():
            browser.evaluate("document.querySelector('#target').style.display='block'; "
                             "document.querySelector('#target').disabled=false")
            page = browser.observe(screenshot=False)
            browser.evaluate(expression)
            assert not browser.fresh(page), label
            passed.append(label + " invalidates")

        browser.evaluate("document.querySelector('#target').disabled=false; "
                         "document.querySelector('#target').style.display='block'")
        page = browser.observe(screenshot=False)
        action = next(a for a in page["actions"] if a["label"] == "Delete account")
        # A textless overlay does not alter the model's semantic state, but must block a click.
        browser.evaluate("const cover=document.createElement('div'); "
                         "cover.style.cssText='position:fixed;inset:0;z-index:9999;background:white'; "
                         "document.body.append(cover)")
        assert browser.fresh(page)
        try:
            browser.act(action, page)
        except (RuntimeError, StalePage):
            pass
        else:
            raise AssertionError("Covered target was clicked")
        assert browser.evaluate("window.clicks") == 1
        passed.append("overlay blocked before input")

        browser.evaluate("document.body.innerHTML=" + repr("""
          <form><p id="price">Total $10</p>
          <button type="button" id="buy">Buy</button>
          <label>Search <input id="query" role="combobox" aria-controls="suggestions"></label>
          <div role="listbox" id="suggestions"></div>
          <label><input id="check" type="checkbox">Enabled</label>
          <label><input id="radio" type="radio">Choice</label>
          <input id="readonly" aria-label="Read only" readonly>
          <input id="secret" type="password" value="never expose this">
          <button id="off" disabled>Disabled</button>
          <select id="category" aria-label="Category">
            <option>All</option><option>Design</option><option disabled>Unavailable</option>
          </select></form><aside id="unrelated">News</aside>
        """))
        page = browser.observe(screenshot=False)
        buy = next(a for a in page["actions"] if a["label"] == "Buy")
        browser.evaluate("document.querySelector('#unrelated').textContent='New unrelated news'")
        assert browser.fresh(page, buy)
        assert not browser.fresh(page)
        passed.append("click guard accepts unrelated visible updates; terminal guard rejects them")
        for label, expression in {
            "nearby price": "document.querySelector('#price').textContent='Total $100'",
            "form value": "document.querySelector('#query').value='changed'",
            "form toggle": "document.querySelector('#check').checked=true",
            "target replacement": "document.querySelector('#buy').outerHTML=document.querySelector('#buy').outerHTML",
        }.items():
            page = browser.observe(screenshot=False)
            buy = next(a for a in page["actions"] if a["label"] == "Buy")
            browser.evaluate(expression)
            assert not browser.fresh(page, buy), label
            passed.append(label + " invalidates action-specific guard")

        page = browser.observe(screenshot=False)
        actions = page["actions"]
        for role in ("checkbox", "radio"):
            assert {a["kind"] for a in actions if a.get("role") == role} == {"click"}
        assert {a["kind"] for a in actions if a["label"] == "Read only"} == {"click"}
        assert not any(a["label"] == "Disabled" or a.get("value") == "never expose this" for a in actions)
        assert [a["value"] for a in actions if a["kind"] == "select"] == ["Design"]
        passed.append("native controls expose only supported operations and safe values")

        select = next(a for a in actions if a["kind"] == "select")
        browser.act(select, page)
        assert browser.evaluate("document.querySelector('#category').value") == "Design"
        passed.append("native dropdown selects an observed option")

        browser.evaluate("document.querySelector('#query').addEventListener('input',()=>setTimeout(()=>{"
                         "document.querySelector('#suggestions').innerHTML='<div role=option>Generated</div>'"
                         "},60))")
        page = browser.observe(screenshot=False)
        field = next(a for a in page["actions"] if a["kind"] == "fill")
        browser.act(field, page, text="Generated")
        page = browser.observe(screenshot=False)
        value = browser.evaluate("document.querySelector('#query').value")
        assert value == "Generated", repr(value)
        assert any(a.get("role") == "option" for a in page["actions"])
        passed.append("real text input waits for asynchronous combobox suggestions")

        # Uploads: a hidden input behind a styled label is the common case, and a stray click on that
        # label must not open a native file picker that would stall the tab.
        browser.evaluate("document.body.innerHTML=" + repr("""
          <form><label style="display:inline-block;padding:8px;border:1px solid">Choose a file
          <input id="cv" type="file" accept=".pdf" style="display:none"></label>
          <button type="button" id="browse" onclick="document.querySelector('#cv').click()">Browse</button>
          <input id="visible" type="file" aria-label="Attachments" multiple>
          <button type="button" id="submit">Upload</button></form>
        """))
        browser.evaluate("window.changes=0;document.querySelector('#cv').addEventListener('change',()=>window.changes++)")
        page = browser.observe(screenshot=False)
        uploads = {a["label"]: a for a in page["actions"] if a["kind"] == "upload"}
        assert set(uploads) == {"Choose a file", "Attachments"}, uploads
        assert uploads["Choose a file"]["accept"] == ".pdf" and uploads["Attachments"]["multiple"] is True
        assert "rect" in uploads["Choose a file"], "hidden input borrows its label's box for the inspector"
        assert not any(a["kind"] == "click" and a.get("role") == "file" for a in page["actions"])
        passed.append("hidden and visible file inputs are observed as upload targets, never as clicks")

        browse = next(a for a in page["actions"] if a["label"] == "Browse")
        browser.act(browse, page)
        page = browser.observe(screenshot=False)
        passed.append("a Browse button that opens the picker does not stall the tab")

        with tempfile.TemporaryDirectory() as folder:
            cv = Path(folder) / "cv.pdf"
            cv.write_bytes(b"%PDF-1.4 guard check")
            target = next(a for a in page["actions"] if a["label"] == "Choose a file" and a["kind"] == "upload")
            browser.act({**target, "files": [str(cv)]}, page)
            assert browser.evaluate("document.querySelector('#cv').files[0].name") == "cv.pdf"
            assert browser.evaluate("window.changes") == 1
            assert not browser.fresh(page, target), "an attached file changes the input's state"
            page = browser.observe(screenshot=False)
            assert next(a for a in page["actions"] if a["kind"] == "upload" and a["node"] == target["node"])[
                "value"
            ] == "cv.pdf"
            passed.append("hidden file input receives the caller's file and reports its name")

            browser.evaluate("document.querySelector('#cv').disabled=true")
            try:
                browser.act({**target, "files": [str(cv)]}, page)
            except StalePage:
                pass
            else:
                raise AssertionError("Disabled file input accepted a file")
            passed.append("disabled file input rejects an attach before any browser input")

        # Settle: a click whose effect lands later (client-side routing, streamed content) is
        # observed as a change; a click with no effect is judged unchanged only after the budget.
        browser.evaluate("document.body.innerHTML=" + repr("""
          <button id="slow" onclick="setTimeout(()=>{document.title='Loaded';
            document.body.insertAdjacentHTML('beforeend','<p>Loaded later</p>')},300)">Load later</button>
          <button id="noop">Nothing</button>"""))
        page = browser.observe(screenshot=False)
        started = time.monotonic()
        browser.act(next(a for a in page["actions"] if a["label"] == "Load later"), page)
        after = browser.observe(screenshot=False)
        elapsed = time.monotonic() - started
        assert "Loaded later" in after["text"] and 0.25 < elapsed < 1.2, elapsed
        passed.append("a delayed page change is observed instead of a false page_changed=False")
        started = time.monotonic()
        browser.act(next(a for a in after["actions"] if a["label"] == "Nothing"), after)
        unchanged = browser.observe(screenshot=False)
        elapsed = time.monotonic() - started
        assert unchanged["marker"] == after["marker"] and elapsed >= 1.4, elapsed
        passed.append("a no-op click pays the settle budget once and is then truthfully unchanged")

        # Duplicate labels carry the nearest distinguishing text as context.
        browser.evaluate(
            """document.body.innerHTML=
              '<dl><div><dt>Loan amount</dt><dd>$5,000,000 <button aria-label="Edit">e</button></dd></div>'+
              '<div><dt>Closing date</dt><dd>2026-06-01 <button aria-label="Edit">e</button></dd></div></dl>'"""
        )
        dup = browser.observe(screenshot=False)
        contexts = [a.get("context") for a in dup["actions"] if a["label"] == "Edit"]
        assert contexts == ["$5,000,000 e", "2026-06-01 e"], contexts
        passed.append("duplicate labels are told apart by their nearest enclosing text")

        # PRESS_ENTER submits a field that has no button, the way a keyboard user would.
        browser.evaluate(
            """document.body.innerHTML='<form onsubmit="event.preventDefault();window.submitted=(this.q.value)">'+
              '<label>Query<input name="q" value="hello"></label></form>'"""
        )
        form_page = browser.observe(screenshot=False)
        enter = next(a for a in form_page["actions"] if a["kind"] == "enter")
        assert enter["label"] == "Press Enter in Query", enter
        browser.act(enter, form_page)
        assert browser.evaluate("window.submitted") == "hello"
        passed.append("PRESS_ENTER submits the field it targets")

        # An empty TYPE_TEXT clears the field instead of leaving the old value selected.
        browser.evaluate("document.body.innerHTML='<label>City<input id=\"city\" value=\"Zurich\"></label>'")
        typed = browser.observe(screenshot=False)
        browser.act(next(a for a in typed["actions"] if a["kind"] == "fill"), typed, text="")
        assert browser.evaluate("document.querySelector('#city').value") == ""
        passed.append("an empty text replacement clears the field")

        # Stillness: a region that declares itself aria-busy is a shell, not a page.
        browser.evaluate("""document.body.innerHTML='<div id="panel" aria-busy="true">Loading…</div>';
          setTimeout(()=>{const p=document.querySelector('#panel');p.removeAttribute('aria-busy');
            p.textContent='3 deals'},400)""")
        started = time.monotonic()
        assert browser.quiesce() and "3 deals" in browser.observe(screenshot=False)["text"]
        assert 0.35 < time.monotonic() - started < 1.2
        passed.append("an aria-busy region holds observation until it is done")

        browser.call("Page.navigate", url="about:blank")
        assert not browser.fresh(page, field)
        passed.append("navigation invalidates the old document")
    finally:
        browser.close()
    print("\n".join(passed))
    print(f"PASS: {len(passed)} browser guard checks; no model calls")


if __name__ == "__main__":
    main()
