# Prompt

For each article in the series, directly generate a cover-image SVG and save it inside that article's folder. Discover the article folders by reading [README.md](README.md) — it documents the article numbering, the slug for each article, and where the per-article folders live under [articles/](articles/).

The image filename is derived from the article markdown filename: replace the `.md` extension with `.poster.svg`. For example, `0-presentation-of-the-project/0-presentation-of-the-project.md` produces `0-presentation-of-the-project/0-presentation-of-the-project.poster.svg`.

Once the SVG is written, generate the corresponding `.poster.png` from it automatically (same folder, same basename), so each article folder ends up with both `<article-slug>.poster.svg` and `<article-slug>.poster.png`.

## SVG technical requirements

- Canvas: exactly `1200` by `627` pixels (`viewBox="0 0 1200 627" width="1200" height="627"`), optimized for LinkedIn feed and mobile preview.
- Self-contained: no external fonts, images, or network resources — use system font stacks (e.g. `'Inter Tight','Helvetica Neue',Arial,sans-serif`) and inline SVG primitives only.
- Must rasterize cleanly with `rsvg-convert` (no foreignObject, no JavaScript, no CSS animations).
- Every word in the title and subtitle must remain legible when the image is rendered at 360px wide on a phone.

## Design guidelines (shared across the series)

- Keep typography as the primary focus — large, legible, left-aligned.
- Any graph, chart, or method motif must be subtle and secondary; treat it as a watermark, never competing with the headline.
- Hold the color palette, layout direction, and type system constant across all articles so the covers read as a single set.
- Clean, editorial, professional aesthetic suitable for a technical LinkedIn audience.

## Per-article content the SVG must include

- The exact article title (from the article folder name / outline) rendered as the dominant text.
- A short subtitle or tagline tied to the article's specific topic.
- A small uppercase tag above the title identifying the series and article number (e.g. `BITCOIN ML SERIES • ARTICLE N OF 7`).
- A concrete visual motif that reflects the article's method, kept subtle and secondary to the typography.

## Generating the PNG

After writing the `.poster.svg` in an article folder, render the matching `.poster.png` in the same folder at the native `1200x627` resolution, keeping the same basename as the SVG. Use whichever converter is available on the system (`rsvg-convert -w 1200 -h 627 <slug>.poster.svg -o <slug>.poster.png` is the preferred default). Do this automatically for every article — do not leave the PNG step to the user.
