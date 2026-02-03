// Frontespizio Standard Tesi Italiana (Sleek Style - Reference Match)
#let title_page(
  university: "",
  department: "",
  degree: "Corso di Laurea Magistrale in...",
  degree_label: "Corso di laurea magistrale in",
  title: "Titolo della Tesi",
  author: "Candidato",
  supervisor: "Relatore",
  academic_year: "2024/2025",
  academic_year_label: "Anno Accademico",
  logo_path: none
) = {
  set page(
    margin: (top: 2.5cm, bottom: 2.5cm, left: 2.5cm, right: 2.5cm),
    numbering: none
  )
  
  // All content centered
  align(center)[
    // Logo
    #if logo_path != none {
      image(logo_path, width: 70%)
    } else {
      text(1.4em, weight: "bold")[#university]
      v(0.3em)
      text(1.1em)[#department]
    }
    
    #v(4em)  // Slightly lower
    
    // Corso di laurea (10% larger: 1.1 -> 1.21)
    #text(1.21em)[#degree_label]
    #v(0.3em)
    #text(1.21em, style: "italic")[#degree]
    
    #v(1fr)
    
    // Titolo (10% smaller: 1.8 -> 1.62)
    #text(1.62em, weight: "bold")[#title]
    
    #v(1fr)
    
    // Autore
    #text(1.3em, weight: "bold")[#author]
    #v(1em)
    
    // Barra e Anno Accademico
    #line(length: 60%, stroke: 0.5pt)
    #v(0.5em)
    #text(1em)[#academic_year_label #academic_year]
  ]
  
  pagebreak()
}
