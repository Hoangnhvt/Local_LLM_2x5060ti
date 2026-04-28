# Research

Drop-zone for emerging tech, papers, blog posts, and vendor docs that might
feed into the stack. Anything here is **input material**, not yet decisions.

## Workflow

1. **Drop sources** — PDFs, markdown, URL lists, screenshots, model cards into
   `research/_inbox/` (or directly into a topic folder).
2. **Triage** — for each source, produce a `NOTES.md` answering:
   - What is it? (1-2 lines)
   - Applies to 2× 5060 Ti / Blackwell? (yes / no / partially + why)
   - Stack impact — exact files/configs that would change
   - Maturity — experimental / beta / prod-ready
   - Recommendation — adopt / pilot / watch / skip
3. **Promote** — once a tech earns a spot in the stack, distil findings into
   [../docs/](../docs/) and update the relevant configs. Keep the raw source
   here for traceability.

## Layout

```
research/
├── README.md
├── _inbox/                  # unsorted drops (gitignored)
├── nvfp4/                   # example topic folder
│   ├── sources/             # raw PDFs / saved HTML / screenshots
│   ├── links.md             # URLs + one-line summaries
│   └── NOTES.md             # triage notes (created on demand)
└── <topic>/...
```

## How to feed sources

Any of these work:

- **URLs** — paste in chat; I'll fetch and summarise on request.
- **PDFs / markdown / code** — drop the file in `_inbox/` or a topic folder.
- **Screenshots** — drop image files; I can read them.
- **Tweets / forum threads** — paste the text.

Then say e.g. *"triage research/_inbox/foo.pdf into research/nvfp4/"* and I'll
produce `NOTES.md`, propose stack changes, and (on approval) wire them in.

## Open topics

- [ ] **nvfp4** — native FP4 on Blackwell tensor cores; candidate default for `coder-lg`.
- [ ] **quantization-30b** — survey of quant formats (AWQ / GPTQ / NVFP4 / MXFP4 / FP8 / GGUF / EXL2 / HQQ / AQLM) for ~30B-class models on 2× 5060 Ti.
- _add more as you drop sources_
