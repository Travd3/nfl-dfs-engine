"""Inline optimizer_output.json into the static page. Netlify-ready, one file."""
import os, sys

def build(data="tables/optimizer_output.json",
          template="web/index_template.html", out="web/index.html"):
    html = open(template).read()
    payload = open(data).read()
    assert "__DATA__" in html, "template lost its data placeholder"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(html.replace("__DATA__", payload))
    print(f"wrote {out}  ({os.path.getsize(out)/1024:.0f} KB, self-contained)")

if __name__ == "__main__":
    build(*sys.argv[1:])
