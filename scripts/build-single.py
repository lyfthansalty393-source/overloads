"""Build single.html from the reviewed split sources, using only Python stdlib.

Photos and finale assets remain shared files in assets/. No CDN is required.
"""
from pathlib import Path
import base64

root = Path(__file__).resolve().parents[1]
html = (root / 'index.html').read_text()
css = (root / 'styles.css').read_text()
js = (root / 'app.js').read_text()
three = base64.b64encode((root / 'vendor/three.module.min.js').read_bytes()).decode()
js = js.replace("'./vendor/three.module.min.js'", f"'data:text/javascript;base64,{three}'")
html = html.replace('<link rel="stylesheet" href="styles.css">', f'<style>\n{css}\n</style>')
for element, filename in [('archive-data', 'archive.json'), ('projection-data', 'projections.json')]:
    data = (root / filename).read_text().replace('</', '<\\/')
    html = html.replace(f'<script id="{element}" type="application/json">null</script>',
                        f'<script id="{element}" type="application/json">{data}</script>')
html = html.replace('<script type="module" src="app.js"></script>',
                    '<script type="module">\n' + js.replace('</script', '<\\/script') + '\n</script>')
(root / 'single.html').write_text(html)
print('Built single.html; retain the full assets/ directory alongside it.')
