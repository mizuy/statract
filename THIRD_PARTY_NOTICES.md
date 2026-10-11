# Third-party notices

statract is released under the MIT License (see `LICENSE`). The file below is
included under its own license and is not covered by the MIT License.

## tools/csl/vancouver.csl

- What: the "NLM/Vancouver: Citing Medicine 2nd edition (citation-sequence)"
  Citation Style Language style, used only by the documentation build to
  format references.
- Origin: the Citation Style Language styles repository
  (<https://github.com/citation-style-language/styles>), style id
  `http://www.zotero.org/styles/nlm-citation-sequence`. Authors and
  contributors are listed in the file's `<info>` block.
- License: Creative Commons Attribution-ShareAlike 3.0
  (<https://creativecommons.org/licenses/by-sa/3.0/>), as stated in the file's
  `<rights>` element, which is kept unchanged.
- Modification: statract added an empty `<name/>` element to each `<names>`
  element so that older pandoc citeproc versions apply the style-level name
  options (for example "Austin PC" instead of "Peter C Austin"). The change is
  noted in a comment at the top of the file. The modified file is distributed
  under the same CC BY-SA 3.0 license.
