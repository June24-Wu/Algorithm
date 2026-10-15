# Algorithm Atlas

A responsive study website for the content in this repository. It includes collection cards, a folder sidebar, question-and-answer pages, code highlighting and copying, cross-collection search, persistent light/dark themes, and English/Chinese AI infrastructure answers.

## Project layout

```text
study_site/
  frontend/          HTML, CSS, JavaScript, and favicon
  backend/           Python content builder, preview server, and tests
  content/zh/        Chinese translations of AI infrastructure answers
  collections.json   Published folders, labels, and translation mappings
  requirements.txt   Pinned build dependencies
  dist/              Generated website; ignored by Git
```

English answers are read directly from the repository's existing Markdown. Python practice files are displayed and downloadable. Only the three configured study folders are published; repository metadata and unrelated directories are excluded.

## Local preview

Use Python 3.10 or newer. From the repository root:

```bash
python3 -m venv study_site/.venv
source study_site/.venv/bin/activate
python -m pip install -r study_site/requirements.txt
python study_site/backend/serve.py
```

Open **http://127.0.0.1:8000/**. Stop the server with `Ctrl+C`. Restart it after editing source content or website files to rebuild the preview. A different port can be selected with `--port 8080`.

The preview server also provides `GET /api/catalog` and `GET /api/document?id=<document-id>`. The frontend reads equivalent static JSON files, so the deployed website does not need a running Python server. GitHub Pages hosts HTML, CSS, JavaScript, and static assets, not a persistent application backend. [GitHub Pages documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages).

## Deploy to GitHub Pages

The repository is `June24-Wu/Algorithm`, and its deployment branch is `master`. The workflow is [deploy-study-site.yml](../.github/workflows/deploy-study-site.yml).

1. In the [repository's Pages settings](https://github.com/June24-Wu/Algorithm/settings/pages), set **Build and deployment → Source → GitHub Actions**.
2. Commit and push the website and workflow. Review the selected files before committing:

   ```bash
   git add study_site .github/workflows/deploy-study-site.yml
   git diff --cached --stat
   git commit -m "Add bilingual Algorithm Atlas study website"
   git push origin master
   ```

3. Open [Actions](https://github.com/June24-Wu/Algorithm/actions) and wait for **Deploy Algorithm Atlas** to finish. You can also start it using **Run workflow** on `master`.
4. After the deployment succeeds, visit **https://June24-Wu.github.io/Algorithm/**. The deploy job and Pages settings show the authoritative published URL.

The workflow tests and builds pull requests without publishing them. Pushes to `master` that change the configured content, website, or workflow rebuild and publish the site. The workflow follows GitHub's [custom Pages workflow](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages) and [publishing-source configuration](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

Relative asset URLs and hash-based page routes support the `/Algorithm/` project-site path, direct links, and page refreshes without server rewrite rules. If you rename the repository or change its deployment branch, update `collections.json`, the workflow branch filters, and these URLs.

## Add or update content

- Edit an existing Markdown/Python file in a configured study folder, or add a new subfolder with a README containing `##` sections. Rebuilding regenerates the sidebar and search index.
- Short README files without `##` sections act as directory indexes and are not separate lessons.
- To add a collection, register its repository directory and metadata in `collections.json` and include its path in the workflow triggers.
- Add a Chinese Markdown translation under `content/zh/`, then map its English repository path in `collections.json`. Keep the same question order and section count. Python blocks may use `{{python-block-1}}`, `{{python-block-2}}`, etc. to reuse the canonical English code.
- The bilingual buttons appear for AI Infra Guide. Pages without a Chinese translation show an explicit availability notice rather than claiming their English content was translated.

The seven existing AI infrastructure questions have complete Chinese and English explanations. Additional interview pages are included only after their answers have been added to the repository.

## Build and test

```bash
python -m unittest discover -s study_site/backend -p 'test_*.py' -v
python study_site/backend/build.py
```

Publish only `study_site/dist/` when using a different static host. Static Markdown is sanitized during the build; no API credentials or browser-side service keys are needed.
