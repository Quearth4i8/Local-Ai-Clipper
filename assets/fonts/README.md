# Caption fonts

Drop `.ttf` / `.otf` files in this folder to use them for burned-in captions.
**No system install needed** — FFmpeg/libass loads them straight from here, and
they show up in Settings → Animated captions → Font.

After adding a file, set `captions.font` to the font's **family name** (usually
the filename without the weight suffix, e.g. `Poppins-ExtraBold.ttf` →
`Poppins ExtraBold`). The dropdown lists what it finds.

## Good rounded/heavy faces for this style

All free (SIL Open Font License), all downloadable from Google Fonts:

| Font | Why |
|---|---|
| **Poppins** (ExtraBold/Black) | geometric, round counters — the classic Shorts caption look |
| **Nunito** (ExtraBold/Black) | genuinely rounded terminals, softer |
| **Baloo 2** (ExtraBold) | chunky and very rounded |
| **Fredoka** (SemiBold/Bold) | playful, heavily rounded |
| **Montserrat** (ExtraBold/Black) | geometric, slightly less round |

## Already installed on Windows

These need nothing dropped here — just pick them in the dropdown:

`Arial Rounded MT Bold` (the default, the roundest one Windows ships),
`Segoe UI Black`, `Arial Black`, `Impact`, `Bahnschrift`, `Cooper Black`.

> Font files are gitignored, so anything you add here stays local and no
> licensed font ends up in the repository.
