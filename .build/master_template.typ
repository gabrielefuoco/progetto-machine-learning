#import "title_page.typ": title_page

#let project(
  title: "",
  author: "",
  university: "",
  department: "",
  degree: "",
  supervisor: "",
  academic_year: "",
  degree_label: "Corso di laurea magistrale in",
  academic_year_label: "Anno Accademico",
  logo_path: none,
  numbered_chapters: true,
  heading_numbering: "1.1",
  date: none,
  version: "",
  body,
) = {
  // Metadati PDF
  set document(title: title, author: author)

  // 1. IMPOSTAZIONI PAGINA (Simil-LaTeX geometry)
  set page(
    paper: "a4",
    margin: (left: 3.5cm, right: 2.5cm, top: 3cm, bottom: 3cm), 
    numbering: "1",
  )

  // 2. FONT E TESTO (New Computer Modern - Modern Standard)
  set text(
    font: "New Computer Modern",
    size: 12pt, 
    lang: "it"
  )

  // 3. PARAGRAFI (Stile moderno senza rientro prima riga)
  set par(
    justify: true,
    leading: 0.65em,         // Interlinea standard
    first-line-indent: 0em,  // No rientro - evita problemi con line break markdown
    spacing: 1.2em           // Spazio tra paragrafi (aumentato per chiarezza)
  )

  // --- Frontespizio Modulare ---
  title_page(
    university: university,
    department: department,
    degree: degree,
    title: title,
    author: author,
    supervisor: supervisor,
    academic_year: academic_year,
    degree_label: degree_label,
    academic_year_label: academic_year_label,
    logo_path: logo_path
  )
  counter(page).update(1)

  // Enable dynamic numbering from config
  set heading(numbering: heading_numbering)

  // --- Indice (Unnumbered) ---
  show outline.entry: it => {
    it
  }
  outline(title: "Indice", depth: 3, indent: auto)
  pagebreak()

  // --- Heading Design (LaTeX-like with Hanging Indent) ---
  
  // Chapter (Level 1)
  show heading.where(level: 1): it => {
    set text(size: 1.8em, weight: "bold")
    set par(justify: false)
    block(above: 2em, below: 1.5em)[
      #if numbered_chapters and it.numbering != none {
        grid(
            columns: (auto, 1fr),
            gutter: 0.5em,
            counter(heading).display(),
            it.body
        )
      } else {
        it.body
      }
    ]
  }
  
  // Section (Level 2)
  show heading.where(level: 2): it => {
    set text(size: 1.3em, weight: "bold")
    set par(justify: false)
    block(above: 1.5em, below: 1em)[
      #if numbered_chapters and it.numbering != none {
        grid(
            columns: (auto, 1fr),
            gutter: 0.5em,
            counter(heading).display(),
            it.body
        )
      } else {
        it.body
      }
    ]
  }
  
  // Subinterion (Level 3)
  show heading.where(level: 3): it => {
    set text(size: 1.1em, weight: "bold")
    set par(justify: false)
    block(above: 1em, below: 0.8em)[
      #if numbered_chapters and it.numbering != none {
        grid(
            columns: (auto, 1fr),
            gutter: 0.5em,
            counter(heading).display(),
            it.body
        )
      } else {
        it.body
      }
    ]
  }

  // --- Content ---
  body
}
