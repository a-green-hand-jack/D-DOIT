# D-DOIT project page

This directory is a self-contained static project page for
**D-DOIT: Training-free Adaptation of Discrete Diffusion via Doob's
`h`-Transform**.

It follows the clean academic-project-page pattern used by the SENTINEL page,
with no build step or third-party JavaScript dependency. The current visual direction
also borrows the information-rich, image-first narrative rhythm used by the NeurIPS
2026 MetaCanvas and CO₂Jump project pages; only the layout conventions are reused.
The visual tokens use named colors from [Chinese Traditional Colors](https://github.com/nevertoday/zhongguo-traditional-colors): 月白, 乳白, 玉色, 玄青, 鸦青, 青黛, 石青, 赭石, 淡桃红, 墨色, 黛蓝, and 金瓜黄.

The page exposes the practical D-DOIT sampler in two pseudocode panels, the main
DRAKES HepG2 comparison table, cross-task Ctrl-DNA/protein results, and an evaluation
protocol map. Values and caveats follow `PAPER.md` and the active manuscript sources. Open `index.html`
directly for a quick preview, or serve the directory locally:

```bash
python3 -m http.server 8000 --directory website
```

Live page: <https://a-green-hand-jack.github.io/D-DOIT/>

The workflow at `.github/workflows/deploy-pages.yml` publishes this directory
from the public `a-green-hand-jack/D-DOIT` repository on pushes to `main` that
touch the site. Pages is enabled with GitHub Actions as its source.

The private paper repository retains the website source; its Pages job is
skipped. To publish future edits made here, copy `website/` and the Pages
workflow to the public repository and commit and push there as well.

The arXiv button is intentionally disabled until a preprint URL exists. The
OpenReview and code buttons already point to the supplied public resources.
