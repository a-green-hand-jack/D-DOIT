# D-DOIT project page

This directory is a self-contained static project page for
**D-DOIT: Training-free Adaptation of Discrete Diffusion via Doob's
`h`-Transform**.

It follows the clean academic-project-page pattern used by the SENTINEL page,
with no build step or third-party JavaScript dependency. Open `index.html`
directly for a quick preview, or serve the directory locally:

```bash
python3 -m http.server 8000 --directory website
```

The repository workflow at `.github/workflows/deploy-pages.yml` publishes this
directory to GitHub Pages on pushes to `main` that touch the site. The first
deployment still requires GitHub Pages to be enabled for the repository; after
that, the workflow owns the deployment and does not require a build toolchain.

The arXiv button is intentionally disabled until a preprint URL exists. The
OpenReview and code buttons already point to the supplied public resources.
