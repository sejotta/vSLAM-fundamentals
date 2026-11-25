# vSLAM Fundamentals — Public Showcase

Public website for the vSLAM Fundamentals project: incremental 3D
reconstruction and camera localisation with PnP.

- **Live site:** https://sejotta.github.io/vSLAM-fundamentals/
- **Contact:** https://wa.me/qr/PXTMUZ4FCBSRP1
- **Code:** private; shared on request via WhatsApp.

This repository contains only the static public site:

```text
index.html
assets/          images, GIFs and the WhatsApp QR card
LICENSE
.nojekyll
```

There are **no `.py` files** and no private implementation here. The Python
package, tests, examples and configuration live in a separate private
repository.

## Local preview

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/`.

## Deploy

GitHub Pages serves `main` from the repository root. Every push to `main`
rebuilds the static site. The `.nojekyll` file disables Jekyll processing.

## Licence

MIT. See [LICENSE](LICENSE).
